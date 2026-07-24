"""
grape_baseline.py
------------------
Gradient-based (GRAPE-style) open-loop pulse optimizers, used as baselines
against the trained RL agent.

Two GRAPE variants are implemented, because comparing RL (trained across a
distribution of miscalibration) against a GRAPE pulse optimized for only the
nominal point is not a fair fight -- a real GRAPE practitioner facing a
miscalibrated device would optimize for robustness too:

  - `run_grape`          : standard GRAPE, optimized against the single
                            nominal (zero-miscalibration) qubit, with a
                            genuine multi-start search (full [-1,1]-range
                            random restarts, not a narrow band).
  - `run_ensemble_grape`  : robust/ensemble GRAPE, optimized against the
                            *average* reward over a deterministic 3x3
                            quadrature grid over (detuning, amplitude) at
                            {-1,0,+1} standard deviations -- 9 points that
                            cover the 2-D uncertainty distribution far more
                            evenly than a handful of random draws, which is
                            what a sample-average approximation of an
                            expectation actually needs. Still open-loop, no
                            mid-episode feedback.

Both optimize the exact same (I, Q) piecewise-constant parameterization the
RL agent controls, subject to the same physics (3-level Lindblad dynamics,
AWG bandwidth filter) and the same reward (energy + leakage penalty, final
fidelity bonus).

A note on "closed-loop", and the mechanism test that settles it: the RL
policy observes the exact simulator density matrix at every step, which is
*not* something a real qubit readout can provide mid-gate (readout is
destructive/perturbing and takes hundreds of ns to us, versus a 10 ns gate
here). Whether RL's measured robustness comes from a genuinely more robust
*waveform* or from exploiting that (unrealistic) mid-episode information is
directly testable: `distill_policy()` freezes one nominal rollout of the
trained policy into a fixed, open-loop pulse and evaluates it exactly like
GRAPE. See SCIENCE.md, Section 7, for the full results -- freezing the
policy collapses its robustness below both GRAPE variants, which is the
answer.

Usage
-----
    python grape_baseline.py --model models/best_model.zip
"""

import os
import sys
import argparse
import numpy as np
from scipy.optimize import minimize

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_env import QuantumPulseEnv
from utils import QUBIT_PARAMS


def parse_args():
    p = argparse.ArgumentParser(description="GRAPE baselines vs trained RL agent")
    p.add_argument("--model", type=str, default="models/best_model.zip")
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=0.35)
    p.add_argument("--leakage-pen", type=float, default=1.0)
    p.add_argument("--energy-pen", type=float, default=0.01)
    p.add_argument("--bandwidth-tau", type=float, default=0.5)
    p.add_argument("--detuning-std", type=float, default=0.03)
    p.add_argument("--amp-error-std", type=float, default=0.05)
    p.add_argument("--n-restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--ensemble-size", type=int, default=6,
                   help="Number of fixed miscalibration draws ensemble-GRAPE optimizes against")
    p.add_argument("--restart-seed", type=int, default=0,
                   help="Seed for GRAPE's random-restart initial guesses")
    p.add_argument("--n-eval", type=int, default=200,
                   help="Number of random-miscalibration episodes for the robustness test")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def make_env(args, detuning_std, amp_error_std):
    return QuantumPulseEnv(
        n_steps=args.n_steps, dt=args.dt, omega_max=args.omega_max,
        energy_penalty=args.energy_pen, leakage_penalty=args.leakage_pen,
        bandwidth_tau=args.bandwidth_tau,
        detuning_std=detuning_std, amp_error_std=amp_error_std,
        params=QUBIT_PARAMS,
    )


def rollout_fixed_pulse(env, action_seq, seed=None, options=None):
    """Replay a FIXED (open-loop) sequence of (ax, ay) actions -- ignoring
    whatever the environment observes -- and return (fidelity, leakage,
    total_reward). `options` can force a specific (detuning, amp_gain) via
    {'detuning':.., 'amp_gain':..} instead of letting the env sample one."""
    obs, _ = env.reset(seed=seed, options=options)
    total_reward = 0.0
    info = {}
    for a in action_seq:
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        total_reward += r
        if term or trunc:
            break
    return info["fidelity"], info["leakage"], total_reward


def naive_pulse(args):
    """The analytic, leakage-blind, exactly-resonant constant I-only
    pi-pulse -- the baseline a two-level-only analysis would recommend.
    Returns a (n_steps, 2) action array (normalized to [-1,1])."""
    amp = np.pi / (args.n_steps * args.dt)
    a_norm = np.clip(amp / args.omega_max, -1.0, 1.0)
    return np.stack([np.full(args.n_steps, a_norm), np.zeros(args.n_steps)], axis=1)


def sample_draws(n, detuning_std, amp_error_std, seed):
    """A fixed, reusable set of (detuning, amp_gain) pairs, so every
    controller (naive / GRAPE / ensemble-GRAPE / RL) can be evaluated on
    the *identical* paired draws -- removing evaluation-noise confounds
    from the comparison between controllers."""
    rng = np.random.default_rng(seed)
    detunings = rng.normal(0.0, detuning_std, size=n) if detuning_std > 0 else np.zeros(n)
    amp_gains = 1.0 + (rng.normal(0.0, amp_error_std, size=n) if amp_error_std > 0 else np.zeros(n))
    return list(zip(detunings.tolist(), amp_gains.tolist()))


def quadrature_draws(detuning_std, amp_error_std):
    """Deterministic 3x3 grid over (detuning, amp_gain) at {-1,0,+1} standard
    deviations each -- 9 points that cover a 2-D Gaussian's mass far more
    evenly than a handful of random draws, at fixed, reproducible cost. The
    standard way robust/ensemble optimal control approximates an expectation
    over a *small*-dimensional uncertainty distribution (a sample-average
    approximation with random draws needs many more samples than a
    quadrature rule to reach the same estimation variance)."""
    detunings = [-detuning_std, 0.0, detuning_std]
    amp_gains = [1.0 - amp_error_std, 1.0, 1.0 + amp_error_std]
    return [(d, g) for d in detunings for g in amp_gains]


def distill_policy(model, args, seed=12345):
    """Roll the trained RL policy out ONCE on the nominal (zero-miscalibration)
    qubit and freeze the exact sequence of raw actions it chose into a fixed,
    open-loop (n_steps,2) action array -- suitable for evaluate_on_draws()
    exactly like a GRAPE pulse.

    This is the mechanism test: if RL's measured robustness advantage (Section
    7) comes from finding a genuinely more robust *waveform*, the frozen pulse
    should hold up nearly as well as the live, reactive policy. If it instead
    comes from exploiting mid-episode state information no real qubit readout
    can provide at gate timescales, the frozen pulse should collapse toward
    (or below) the open-loop GRAPE baselines once it can no longer react."""
    env = make_env(args, detuning_std=0.0, amp_error_std=0.0)
    obs, _ = env.reset(seed=seed)
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    return np.array(actions)


def _best_fit_scale_r2(y_true, y_pred):
    """Least-squares scale c minimizing ||y_true - c*y_pred||^2 (a one-
    parameter fit, since DRAG only predicts a shape, not an amplitude), and
    the R^2 of that fit against y_true."""
    denom = np.dot(y_pred, y_pred)
    c = np.dot(y_true, y_pred) / denom if denom > 0 else 0.0
    resid = y_true - c * y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(c), float(r2)


def drag_correlation_check(actions, dt, omega_max, alpha):
    """Compares a pulse's actual Q-quadrature Omega_y(t) against the
    first-order DRAG prediction Omega_y(t) ~= -dOmega_x(t)/dt / alpha
    (Motzoi et al. 2009). `actions` is a (n_steps,2) array of normalized
    [-1,1] (ax, ay) values, e.g. from distill_policy() or a GRAPE action
    sequence. Returns (correlation, r_squared, best_fit_scale)."""
    omega_x = actions[:, 0] * omega_max
    omega_y = actions[:, 1] * omega_max
    domega_x = np.gradient(omega_x, dt)
    drag_pred = -domega_x / alpha
    if np.std(omega_y) > 0 and np.std(drag_pred) > 0:
        corr = float(np.corrcoef(omega_y, drag_pred)[0, 1])
    else:
        corr = float("nan")
    scale, r2 = _best_fit_scale_r2(omega_y, drag_pred)
    return corr, r2, scale


def validate_drag_detector(alpha, n_steps=10, dt=1.0, omega_max=0.35):
    """Self-test for drag_correlation_check(): builds a synthetic Gaussian
    Omega_x(t) envelope together with its EXACT analytic DRAG-derived
    Omega_y(t) = -dOmega_x/dt / alpha, then runs that pair through
    drag_correlation_check(). A negative or weak result on a real policy
    (SCIENCE.md, Section 7.5) is only trustworthy if this self-test comes
    back near-perfect (corr~1, R^2~1) -- otherwise the detector, not the
    policy, would be the thing that's broken.

    The peak amplitude is deliberately kept well under omega_max: this is a
    detector self-test, not a physical pi-pulse, and if the synthetic
    Omega_x(t) or Omega_y(t) clipped against the [-1,1] action bounds the
    clipped signal would no longer be an exact derivative pair, defeating
    the point of the test."""
    t = np.arange(n_steps) * dt
    center = (n_steps - 1) * dt / 2.0
    sigma = n_steps * dt / 4.0
    peak = 0.3 * omega_max
    omega_x = peak * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    omega_y = -np.gradient(omega_x, dt) / alpha
    assert np.max(np.abs(omega_x)) <= omega_max and np.max(np.abs(omega_y)) <= omega_max, \
        "synthetic DRAG self-test pulse exceeds omega_max -- would be clipped, invalidating the test"
    actions = np.stack([omega_x / omega_max, omega_y / omega_max], axis=1)
    return drag_correlation_check(actions, dt, omega_max, alpha)


def evaluate_on_draws(controller, args, draws, is_rl: bool):
    """Evaluate a controller (fixed (n_steps,2) action array, or an SB3
    model if is_rl) on a fixed, pre-sampled list of (detuning, amp_gain)
    pairs from sample_draws(). Returns an array of per-draw fidelities.
    See evaluate_on_draws_full() to also get leakage from the same rollouts
    instead of re-simulating twice."""
    fids, _ = evaluate_on_draws_full(controller, args, draws, is_rl)
    return fids


def evaluate_on_draws_full(controller, args, draws, is_rl: bool):
    """Same as evaluate_on_draws, but returns (fidelities, leakages) from a
    single pass over the draws -- both are read off the same deterministic
    rollout's final info dict, so computing them separately would just
    re-simulate identical physics twice."""
    env = make_env(args, detuning_std=args.detuning_std, amp_error_std=args.amp_error_std)
    fids, leaks = [], []
    for detuning, amp_gain in draws:
        options = {"detuning": detuning, "amp_gain": amp_gain}
        obs, _ = env.reset(options=options)
        info = {}
        if is_rl:
            done = False
            while not done:
                action, _ = controller.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
        else:
            for a in controller:
                obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        fids.append(info["fidelity"])
        leaks.append(info["leakage"])
    return np.array(fids), np.array(leaks)


def _optimize(objective, args, extra_args, n_params, restart_seed, tag_prefix):
    pi_pulse_amp = np.clip(np.pi / (args.n_steps * args.dt) / args.omega_max, -1.0, 1.0)
    init_guesses = [np.stack(
        [np.full(args.n_steps, pi_pulse_amp), np.zeros(args.n_steps)], axis=1
    ).flatten()]
    rng = np.random.default_rng(restart_seed)
    # Full [-1,1] range, not a narrow band around zero -- a genuine multi-start
    # search needs restarts that can land anywhere in the feasible region, not
    # just near the origin, or they under-explore the landscape by construction.
    for _ in range(args.n_restarts - 1):
        init_guesses.append(rng.uniform(-1.0, 1.0, size=n_params))

    best_result = None
    for i, x0 in enumerate(init_guesses):
        res = minimize(
            objective, x0, args=extra_args,
            method="L-BFGS-B", bounds=[(-1.0, 1.0)] * n_params,
            options={"maxiter": args.maxiter, "ftol": 1e-12, "gtol": 1e-10},
        )
        tag = "seeded" if i == 0 else f"random-{i}"
        print(f"  [{tag_prefix}:{tag}] final objective (-reward) = {res.fun:.6f}")
        if best_result is None or res.fun < best_result.fun:
            best_result = res
    return best_result.x.reshape(args.n_steps, 2)


def grape_objective(flat_params, env, n_steps):
    action_seq = flat_params.reshape(n_steps, 2)
    _, _, total_reward = rollout_fixed_pulse(env, action_seq, seed=12345)
    return -total_reward


def run_grape(args, restart_seed=None):
    """Standard GRAPE: optimize against the single nominal (zero-miscalibration) qubit."""
    if restart_seed is None:
        restart_seed = args.restart_seed
    nominal_env = make_env(args, detuning_std=0.0, amp_error_std=0.0)
    n_params = args.n_steps * 2
    print(f"[grape] Optimizing {n_params} params vs. NOMINAL qubit "
          f"({args.n_restarts} restarts, maxiter={args.maxiter})...")
    action_seq = _optimize(grape_objective, args, (nominal_env, args.n_steps),
                            n_params, restart_seed, "grape")
    fidelity, leakage, reward = rollout_fixed_pulse(nominal_env, action_seq, seed=12345)
    print(f"[grape] Best nominal fidelity: {fidelity:.4f}  leakage: {leakage:.4f}  reward: {reward:.4f}")
    return action_seq, fidelity, leakage


def ensemble_grape_objective(flat_params, env, n_steps, draws):
    action_seq = flat_params.reshape(n_steps, 2)
    total = 0.0
    for detuning, amp_gain in draws:
        options = {"detuning": detuning, "amp_gain": amp_gain}
        _, _, reward = rollout_fixed_pulse(env, action_seq, options=options)
        total += reward
    return -total / len(draws)


def run_ensemble_grape(args, restart_seed=None, ensemble_seed=None, draws=None):
    """Robust/ensemble GRAPE: optimize against the AVERAGE reward over a
    fixed sample of miscalibration draws (sample-average approximation of
    robust optimal control) -- the standard way a real GRAPE practitioner
    would handle a known-miscalibrated device. Still an open-loop, fixed
    pulse: no mid-episode feedback.

    By default uses a deterministic 3x3 quadrature grid (quadrature_draws) --
    9 points that cover the 2-D uncertainty distribution far more evenly than
    a handful of random draws, which is what the sample-average approximation
    actually needs to be a good approximation. Pass `draws` explicitly to
    override (e.g. random sample_draws(), for comparison)."""
    if restart_seed is None:
        restart_seed = args.restart_seed
    if ensemble_seed is None:
        ensemble_seed = args.restart_seed
    env = make_env(args, detuning_std=0.0, amp_error_std=0.0)  # draws forced via options
    if draws is None:
        draws = quadrature_draws(args.detuning_std, args.amp_error_std)
    n_params = args.n_steps * 2
    print(f"[ensemble-grape] Optimizing {n_params} params vs. {len(draws)}-point "
          f"ensemble ({args.n_restarts} restarts, maxiter={args.maxiter})...")
    action_seq = _optimize(ensemble_grape_objective, args, (env, args.n_steps, draws),
                            n_params, restart_seed, "ensemble-grape")
    # Report fidelity/leakage averaged over the same ensemble it trained on.
    fids, leaks = [], []
    for detuning, amp_gain in draws:
        f, l, _ = rollout_fixed_pulse(env, action_seq, options={"detuning": detuning, "amp_gain": amp_gain})
        fids.append(f); leaks.append(l)
    print(f"[ensemble-grape] Ensemble-mean fidelity: {np.mean(fids):.4f}  leakage: {np.mean(leaks):.4f}")
    return action_seq, float(np.mean(fids)), float(np.mean(leaks))


def main():
    """Quick single-seed, 5-controller comparison (naive / GRAPE / ensemble-GRAPE /
    RL / frozen-RL) for interactive use (e.g. the tutorial notebook). For the full
    5-seed study with proper mean+-std across training/optimization seeds, see
    seed_study.py."""
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("  GRAPE Baselines vs Trained RL Agent (single-seed quick check)")
    print("=" * 60)

    naive_actions = naive_pulse(args)
    grape_actions, grape_nominal_fid, grape_nominal_leak = run_grape(args)
    ens_actions, ens_nominal_fid, ens_nominal_leak = run_ensemble_grape(args)

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    nominal_env = make_env(args, detuning_std=0.0, amp_error_std=0.0)
    obs, _ = nominal_env.reset(seed=12345)
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = nominal_env.step(action)
        done = term or trunc
    rl_nominal_fid, rl_nominal_leak = info["fidelity"], info["leakage"]

    frozen_actions = distill_policy(model, args)

    print(f"\n[compare] Evaluating all 5 controllers on {args.n_eval} PAIRED miscalibration draws...")
    draws = sample_draws(args.n_eval, args.detuning_std, args.amp_error_std, seed=999)
    naive_fids = evaluate_on_draws(naive_actions, args, draws, is_rl=False)
    grape_fids = evaluate_on_draws(grape_actions, args, draws, is_rl=False)
    ens_fids = evaluate_on_draws(ens_actions, args, draws, is_rl=False)
    rl_fids = evaluate_on_draws(model, args, draws, is_rl=True)
    frozen_fids = evaluate_on_draws(frozen_actions, args, draws, is_rl=False)

    rows = [
        ("Naive constant pulse", None, naive_fids),
        ("GRAPE (nominal-only)", grape_nominal_fid, grape_fids),
        ("Ensemble-GRAPE (robust)", ens_nominal_fid, ens_fids),
        ("RL (live/closed-loop)", rl_nominal_fid, rl_fids),
        ("RL (frozen/open-loop)", None, frozen_fids),
    ]
    print("\n" + "-" * 66)
    print(f"{'':24s}{'Nominal':>12s}{'Robust mean':>15s}{'Robust std':>12s}")
    for name, nominal, fids in rows:
        nominal_str = f"{nominal:12.4f}" if nominal is not None else f"{'--':>12s}"
        print(f"{name:24s}{nominal_str}{fids.mean():15.4f}{fids.std():12.4f}")
    print("-" * 66)

    summary_path = os.path.join(args.results_dir, "grape_comparison.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GRAPE Baselines vs Trained RL Agent (single-seed)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Detuning std / Amp error std: {args.detuning_std} / {args.amp_error_std}\n")
        f.write(f"n_eval (paired draws)       : {args.n_eval}\n\n")
        for name, nominal, fids in rows:
            f.write(f"{name}\n")
            if nominal is not None:
                f.write(f"  nominal fidelity : {nominal:.6f}\n")
            f.write(f"  robust mean/std  : {fids.mean():.6f} / {fids.std():.6f}\n\n")
    print(f"\n[compare] Summary saved -> {summary_path}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"Naive constant pulse": "#9E9E9E", "GRAPE (nominal-only)": "#E53935",
              "Ensemble-GRAPE (robust)": "#8E24AA", "RL (live/closed-loop)": "#1E88E5",
              "RL (frozen/open-loop)": "#43A047"}
    lo = min(f.min() for _, _, f in rows) - 0.01
    bins = np.linspace(lo, 1.0, 40)
    for name, nominal, fids in rows:
        ax.hist(fids, bins=bins, alpha=0.5, color=colors[name],
                label=f"{name}, mean={fids.mean():.4f}")
    ax.set_xlabel("Gate fidelity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Fidelity under paired random miscalibration (n={args.n_eval})\n"
                 f"detuning~N(0,{args.detuning_std}), amp gain~N(1,{args.amp_error_std})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "grape_vs_rl_robustness.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[compare] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
