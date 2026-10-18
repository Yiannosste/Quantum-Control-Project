"""
evaluate.py
-----------
Load a trained PPO model and evaluate it on the 3-level quantum pulse
environment, including its behavior under per-episode miscalibration.

Usage
-----
    python evaluate.py                           # load best_model.zip
    python evaluate.py --model models/ppo_quantum_pulse.zip
    python evaluate.py --n-episodes 20

Outputs
-------
    results/pulse_bloch_trajectory.png   : I/Q pulse + populations + Bloch plot
    results/eval_summary.txt             : fidelity/leakage statistics
"""

import os
import sys
import argparse
import numpy as np

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# encode the unicode symbols (π, →, …) used in this script's log output.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from stable_baselines3 import PPO

from quantum_env import QuantumPulseEnv
from utils import QUBIT_PARAMS, plot_pulse_and_bloch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained quantum pulse agent")
    p.add_argument("--model",       type=str, default="models/best_model.zip",
                   help="Path to the saved SB3 model (.zip)")
    p.add_argument("--n-episodes",  type=int, default=10,
                   help="Number of evaluation episodes (default: 10)")
    p.add_argument("--n-steps",     type=int, default=10)
    p.add_argument("--dt",          type=float, default=1.0)
    p.add_argument("--omega-max",   type=float, default=0.35)
    p.add_argument("--leakage-pen", type=float, default=1.0)
    p.add_argument("--bandwidth-tau", type=float, default=0.5)
    p.add_argument("--detuning-std", type=float, default=0.03,
                   help="Per-episode detuning stddev used during evaluation (default: 0.03)")
    p.add_argument("--amp-error-std", type=float, default=0.05,
                   help="Per-episode amplitude-error stddev used during evaluation (default: 0.05)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic policy (recommended for eval)")
    return p.parse_args()


def run_episode(model, env, deterministic: bool = True, seed=None):
    """Run a single episode and return (fidelity, reward, info)."""
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    fidelity = info.get("fidelity", 0.0)
    return fidelity, total_reward, info


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Check model path ──────────────────────────────────────────────────
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = "models/ppo_quantum_pulse.zip"
        if os.path.exists(fallback):
            model_path = fallback
            print(f"[eval] best_model.zip not found, using {fallback}")
        else:
            raise FileNotFoundError(
                f"No model found at '{args.model}' or '{fallback}'.\n"
                "Run  python train.py  first."
            )

    print("=" * 60)
    print("  RL-Driven Quantum Pulse Optimization: Evaluation")
    print("=" * 60)
    print(f"  Model        : {model_path}")
    print(f"  Episodes     : {args.n_episodes}")
    print(f"  Gate time    : {args.n_steps * args.dt:.0f} ns")
    print(f"  Miscalibration: detuning_std={args.detuning_std}, amp_error_std={args.amp_error_std}")
    print("=" * 60)

    # ── Load model & environment ──────────────────────────────────────────
    model = PPO.load(model_path)

    env_kwargs = dict(
        n_steps=args.n_steps,
        dt=args.dt,
        omega_max=args.omega_max,
        leakage_penalty=args.leakage_pen,
        bandwidth_tau=args.bandwidth_tau,
        detuning_std=args.detuning_std,
        amp_error_std=args.amp_error_std,
        params=QUBIT_PARAMS,
    )
    env = QuantumPulseEnv(**env_kwargs)

    # ── Run evaluation episodes (each with a fresh random miscalibration) ──
    fidelities, leakages = [], []
    best_fidelity = -1.0
    best_info = None

    for ep in range(args.n_episodes):
        fidelity, reward, info = run_episode(model, env, deterministic=args.deterministic, seed=1000 + ep)
        fidelities.append(fidelity)
        leakages.append(info["leakage"])
        print(f"  Episode {ep+1:>3d} | Fidelity: {fidelity:.4f} | Leakage: {info['leakage']:.4f} | "
              f"Detuning: {info['detuning']:+.4f} | AmpGain: {info['amp_gain']:.4f} | Reward: {reward:.4f}")

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_info = info

    # ── Summary statistics ────────────────────────────────────────────────
    fid_arr = np.array(fidelities)
    leak_arr = np.array(leakages)
    print("\n" + "─" * 50)
    print(f"  Mean fidelity  : {fid_arr.mean():.4f} ± {fid_arr.std():.4f}")
    print(f"  Best fidelity  : {fid_arr.max():.4f}")
    print(f"  Worst fidelity : {fid_arr.min():.4f}")
    print(f"  Mean leakage   : {leak_arr.mean():.4f}")
    print(f"  Episodes ≥ 0.99 fidelity: {(fid_arr >= 0.99).sum()}/{args.n_episodes}")
    print("─" * 50)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, "eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RL Quantum Pulse Optimization: Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model       : {model_path}\n")
        f.write(f"Episodes    : {args.n_episodes}\n")
        f.write(f"Gate time   : {args.n_steps * args.dt:.0f} ns\n")
        f.write(f"T1 / T2     : {QUBIT_PARAMS['T1']:.0f} / {QUBIT_PARAMS['T2']:.0f} ns\n")
        f.write(f"Anharmonicity: {QUBIT_PARAMS['alpha']:.4f} rad/ns\n")
        f.write(f"Detuning std / Amp error std: {args.detuning_std} / {args.amp_error_std}\n\n")
        f.write(f"Mean fidelity  : {fid_arr.mean():.6f}\n")
        f.write(f"Std fidelity   : {fid_arr.std():.6f}\n")
        f.write(f"Best fidelity  : {fid_arr.max():.6f}\n")
        f.write(f"Worst fidelity : {fid_arr.min():.6f}\n")
        f.write(f"Mean leakage   : {leak_arr.mean():.6f}\n")
        f.write(f"Episodes ≥0.99 : {(fid_arr >= 0.99).sum()}/{args.n_episodes}\n\n")
        f.write("Per-episode fidelities / leakage:\n")
        for i, (fid, lk) in enumerate(zip(fidelities, leakages)):
            f.write(f"  Ep {i+1:>3d}: fidelity={fid:.6f}  leakage={lk:.6f}\n")
    print(f"\n[eval] Summary saved → {summary_path}")

    # ── Visualise best episode ────────────────────────────────────────────
    if best_info is not None:
        print(f"\n[eval] Plotting best episode (fidelity={best_fidelity:.4f}) ...")
        pulse_x = np.array(best_info["pulse_x"])
        pulse_y = np.array(best_info["pulse_y"])
        populations, bloch01 = env.get_full_trajectory(
            pulse_x, pulse_y,
            detuning=best_info["detuning"], amp_gain=best_info["amp_gain"],
        )
        plot_path = os.path.join(args.results_dir, "pulse_bloch_trajectory.png")
        plot_pulse_and_bloch(
            amplitudes_x=pulse_x,
            amplitudes_y=pulse_y,
            populations=populations,
            bloch01=bloch01,
            dt=args.dt,
            fidelity=best_fidelity,
            save_path=plot_path,
        )

    # ── Print ideal (noise-free, nominal-calibration) benchmark ───────────
    print("\n[eval] Computing noise-free, nominal-calibration benchmark ...")
    env_ideal = QuantumPulseEnv(
        n_steps=args.n_steps, dt=args.dt, omega_max=args.omega_max,
        leakage_penalty=0.0, bandwidth_tau=args.bandwidth_tau,
        detuning_std=0.0, amp_error_std=0.0,
        params={**QUBIT_PARAMS, "T1": 1e9, "T2": 1e9},
    )
    fid_ideal, _, _ = run_episode(model, env_ideal, deterministic=True, seed=0)
    print(f"  Ideal fidelity (no decoherence, no miscalibration): {fid_ideal:.4f}")
    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
