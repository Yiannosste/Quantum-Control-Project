"""
seed_study.py
-------------
Multi-seed robustness + mechanism study across five controllers:

  1. Naive constant pulse:         analytic, leakage-blind floor.
  2. GRAPE (nominal-only):         open-loop, optimized for the known qubit.
  3. Ensemble GRAPE (robust):      open-loop, optimized for the average
                                    reward over a 3x3 quadrature grid spanning
                                    the miscalibration distribution.
  4. RL (live/closed-loop):        the trained policy, observing exact
                                    simulator state every step.
  5. RL (frozen/open-loop):        one nominal rollout of that SAME policy,
                                    frozen into a fixed pulse and replayed
                                    open-loop. This is the mechanism test.

RL is trained across 5 independent seeds (200,000 timesteps each); GRAPE and
ensemble GRAPE are each optimized across 5 independent random-restart seeds,
with a genuine multi-start search (full [-1,1]-range random restarts).  Every
controller, every seed, is evaluated on an IDENTICAL set of paired
miscalibration draws, and results are reported as mean +/- std ACROSS SEEDS
(the hierarchical statistic that actually bounds whether a gap is real or
seed noise), plus paired significance tests against the two GRAPE
baselines.

Usage
-----
    python seed_study.py --n-seeds 5 --timesteps 200000
"""

import os
import sys
import json
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from grape_baseline import (
    make_env, naive_pulse, sample_draws, evaluate_on_draws_full,
    run_grape, run_ensemble_grape, distill_policy,
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed GRAPE/ensemble-GRAPE/RL robustness + mechanism study")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=0.35)
    p.add_argument("--leakage-pen", type=float, default=1.0)
    p.add_argument("--energy-pen", type=float, default=0.01)
    p.add_argument("--bandwidth-tau", type=float, default=0.5)
    p.add_argument("--detuning-std", type=float, default=0.03)
    p.add_argument("--amp-error-std", type=float, default=0.05)
    p.add_argument("--grape-restarts", type=int, default=6)
    p.add_argument("--grape-maxiter", type=int, default=50)
    p.add_argument("--ensemble-restarts", type=int, default=2)
    p.add_argument("--ensemble-maxiter", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=150)
    p.add_argument("--rl-parallel", type=int, default=3, help="Max concurrent RL training subprocesses")
    p.add_argument("--model-root", type=str, default="models_seedstudy")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


class GrapeArgsShim:
    """Duck-types the subset of grape_baseline's argparse.Namespace that
    make_env/naive_pulse/run_grape/run_ensemble_grape/evaluate_on_draws need."""
    def __init__(self, a, n_restarts, maxiter, restart_seed):
        self.n_steps = a.n_steps
        self.dt = a.dt
        self.omega_max = a.omega_max
        self.leakage_pen = a.leakage_pen
        self.energy_pen = a.energy_pen
        self.bandwidth_tau = a.bandwidth_tau
        self.detuning_std = a.detuning_std
        self.amp_error_std = a.amp_error_std
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.restart_seed = restart_seed


def train_one_seed(args, seed):
    model_dir = os.path.join(args.model_root, f"seed{seed}")
    log_dir = os.path.join(model_dir, "logs")
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"[seed_study] seed {seed}: model already exists, skipping training")
        return seed, True
    cmd = [
        sys.executable, "train.py",
        "--timesteps", str(args.timesteps),
        "--n-steps", str(args.n_steps),
        "--dt", str(args.dt),
        "--omega-max", str(args.omega_max),
        "--leakage-pen", str(args.leakage_pen),
        "--energy-pen", str(args.energy_pen),
        "--bandwidth-tau", str(args.bandwidth_tau),
        "--detuning-std", str(args.detuning_std),
        "--amp-error-std", str(args.amp_error_std),
        "--seed", str(seed),
        "--model-dir", model_dir,
        "--log-dir", log_dir,
        "--results-dir", results_dir,
    ]
    print(f"[seed_study] seed {seed}: starting training ({args.timesteps} timesteps)...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[seed_study] seed {seed}: TRAINING FAILED after {dt:.0f}s")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        return seed, False
    print(f"[seed_study] seed {seed}: training done in {dt:.0f}s")
    return seed, True


def nominal_fidelity_leakage(controller, grape_args, is_rl):
    env = make_env(grape_args, detuning_std=0.0, amp_error_std=0.0)
    obs, _ = env.reset(seed=12345)
    done = False
    info = {}
    if is_rl:
        while not done:
            action, _ = controller.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
    else:
        for a in controller:
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
    return info["fidelity"], info["leakage"]


def summarize(nominal_vals, fids_per_seed, leaks_per_seed=None):
    """fids_per_seed: list of per-seed arrays of per-draw fidelities.
    Returns nominal mean/std across seeds, and robust mean/std across
    seeds (of each seed's own mean-over-draws), the correct hierarchical
    statistic for 'variance across training/optimization seeds'."""
    seed_means = np.array([f.mean() for f in fids_per_seed])
    all_fids = np.concatenate(fids_per_seed)
    out = {
        "nominal_mean": float(np.mean(nominal_vals)) if nominal_vals is not None else None,
        "nominal_std": float(np.std(nominal_vals)) if nominal_vals is not None else None,
        "robust_mean_across_seeds": float(seed_means.mean()),
        "robust_std_across_seeds": float(seed_means.std()),
        "robust_mean_all_draws": float(all_fids.mean()),
        "robust_std_all_draws": float(all_fids.std()),
        "per_seed_nominal": [float(v) for v in nominal_vals] if nominal_vals is not None else None,
        "per_seed_robust_mean": [float(v) for v in seed_means],
    }
    if leaks_per_seed is not None:
        leak_seed_means = np.array([l.mean() for l in leaks_per_seed])
        out["leakage_mean_across_seeds"] = float(leak_seed_means.mean())
        out["leakage_std_across_seeds"] = float(leak_seed_means.std())
        out["per_seed_leakage_mean"] = [float(v) for v in leak_seed_means]
    return out


def paired_stats(a_per_seed_means, b_per_seed_means, label_a, label_b):
    """Paired t-test and sign test between two controllers' per-seed robust
    means (same seeds, same paired draws -> a genuine paired comparison)."""
    a = np.array(a_per_seed_means)
    b = np.array(b_per_seed_means)
    diff = a - b
    n = len(diff)
    if np.allclose(diff.std(ddof=1), 0):
        t_stat, p_val = float("nan"), float("nan")
    else:
        t_stat, p_val = stats.ttest_rel(a, b)
    wins = int(np.sum(diff > 0))
    sign_p = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue if n > 0 else float("nan")
    return {
        "comparison": f"{label_a} vs {label_b}",
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if n > 1 else 0.0,
        "t_stat": float(t_stat) if t_stat == t_stat else None,
        "p_value_paired_t": float(p_val) if p_val == p_val else None,
        "wins": wins,
        "n": n,
        "p_value_sign_test": float(sign_p),
    }


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_root, exist_ok=True)
    seeds = list(range(args.n_seeds))

    # ---- 1. Train RL across seeds (parallel pool; skips seeds already trained) ----
    print("=" * 70)
    print(f"  Seed study: {args.n_seeds} seeds, {args.timesteps} timesteps each")
    print("=" * 70)
    with ThreadPoolExecutor(max_workers=args.rl_parallel) as ex:
        results = list(ex.map(lambda s: train_one_seed(args, s), seeds))
    failed = [s for s, ok in results if not ok]
    if failed:
        raise RuntimeError(f"Training failed for seeds {failed}; aborting.")

    # ---- 2. Fixed evaluation draw set (identical for every controller) --
    grape_args = GrapeArgsShim(args, args.grape_restarts, args.grape_maxiter, restart_seed=0)
    eval_draws = sample_draws(args.n_eval, args.detuning_std, args.amp_error_std, seed=999)
    print(f"\n[seed_study] Built {args.n_eval} fixed paired evaluation draws (seed=999)")

    # ---- 3. Naive constant-pulse baseline (deterministic, no seed) ------
    naive_actions = naive_pulse(grape_args)
    naive_nominal_fid, naive_nominal_leak = nominal_fidelity_leakage(naive_actions, grape_args, is_rl=False)
    naive_fids, naive_leaks = evaluate_on_draws_full(naive_actions, grape_args, eval_draws, is_rl=False)
    print(f"[seed_study] Naive pulse: nominal={naive_nominal_fid:.4f} robust_mean={naive_fids.mean():.4f} "
          f"leak={naive_leaks.mean():.4f}")

    # ---- 4. GRAPE / ensemble-GRAPE / RL (live + frozen) across seeds ----
    from stable_baselines3 import PPO
    grape_nominals, grape_leaks_nom, grape_fids_per_seed, grape_leaks_per_seed = [], [], [], []
    ens_nominals, ens_leaks_nom, ens_fids_per_seed, ens_leaks_per_seed = [], [], [], []
    rl_nominals, rl_leaks_nom, rl_fids_per_seed, rl_leaks_per_seed = [], [], [], []
    frozen_fids_per_seed, frozen_leaks_per_seed = [], []

    for s in seeds:
        print(f"\n[seed_study] === seed {s} ===")
        g_args = GrapeArgsShim(args, args.grape_restarts, args.grape_maxiter, restart_seed=s)
        grape_actions, gfid, gleak = run_grape(g_args, restart_seed=s)
        g_fids, g_leaks = evaluate_on_draws_full(grape_actions, g_args, eval_draws, is_rl=False)
        grape_nominals.append(gfid); grape_leaks_nom.append(gleak)
        grape_fids_per_seed.append(g_fids); grape_leaks_per_seed.append(g_leaks)

        e_args = GrapeArgsShim(args, args.ensemble_restarts, args.ensemble_maxiter, restart_seed=s)
        ens_actions, efid, eleak = run_ensemble_grape(e_args, restart_seed=s, ensemble_seed=s)
        ens_nominal_fid, ens_nominal_leak = nominal_fidelity_leakage(ens_actions, e_args, is_rl=False)
        e_fids, e_leaks = evaluate_on_draws_full(ens_actions, e_args, eval_draws, is_rl=False)
        ens_nominals.append(ens_nominal_fid); ens_leaks_nom.append(ens_nominal_leak)
        ens_fids_per_seed.append(e_fids); ens_leaks_per_seed.append(e_leaks)

        model_path = os.path.join(args.model_root, f"seed{s}", "best_model.zip")
        model = PPO.load(model_path)
        rl_nominal_fid, rl_nominal_leak = nominal_fidelity_leakage(model, grape_args, is_rl=True)
        rl_fids, rl_leaks = evaluate_on_draws_full(model, grape_args, eval_draws, is_rl=True)
        rl_nominals.append(rl_nominal_fid); rl_leaks_nom.append(rl_nominal_leak)
        rl_fids_per_seed.append(rl_fids); rl_leaks_per_seed.append(rl_leaks)

        # Mechanism test: freeze this seed's policy into a fixed open-loop pulse
        frozen_actions = distill_policy(model, grape_args)
        frozen_fids, frozen_leaks = evaluate_on_draws_full(frozen_actions, grape_args, eval_draws, is_rl=False)
        frozen_fids_per_seed.append(frozen_fids); frozen_leaks_per_seed.append(frozen_leaks)

        print(f"  GRAPE        nominal={gfid:.4f}  robust_mean={g_fids.mean():.4f}  leak={g_leaks.mean():.4f}")
        print(f"  EnsGRAPE     nominal={ens_nominal_fid:.4f}  robust_mean={e_fids.mean():.4f}  leak={e_leaks.mean():.4f}")
        print(f"  RL (live)    nominal={rl_nominal_fid:.4f}  robust_mean={rl_fids.mean():.4f}  leak={rl_leaks.mean():.4f}")
        print(f"  RL (frozen)  robust_mean={frozen_fids.mean():.4f}  leak={frozen_leaks.mean():.4f}")

    # ---- 5. Aggregate + statistics ---------------------------------------
    summary = {
        "config": vars(args),
        "naive": {"nominal": naive_nominal_fid, "leak": naive_nominal_leak,
                   "robust_mean": float(naive_fids.mean()), "robust_std": float(naive_fids.std()),
                   "leak_mean": float(naive_leaks.mean())},
        "grape": summarize(grape_nominals, grape_fids_per_seed, grape_leaks_per_seed),
        "ensemble_grape": summarize(ens_nominals, ens_fids_per_seed, ens_leaks_per_seed),
        "rl": summarize(rl_nominals, rl_fids_per_seed, rl_leaks_per_seed),
        "rl_frozen": summarize(None, frozen_fids_per_seed, frozen_leaks_per_seed),
    }
    summary["stats"] = {
        "rl_vs_grape": paired_stats(summary["rl"]["per_seed_robust_mean"], summary["grape"]["per_seed_robust_mean"],
                                     "RL(live)", "GRAPE"),
        "rl_vs_ensemble_grape": paired_stats(summary["rl"]["per_seed_robust_mean"], summary["ensemble_grape"]["per_seed_robust_mean"],
                                              "RL(live)", "Ensemble-GRAPE"),
        "rl_live_vs_frozen": paired_stats(summary["rl"]["per_seed_robust_mean"], summary["rl_frozen"]["per_seed_robust_mean"],
                                           "RL(live)", "RL(frozen)"),
        "rl_frozen_vs_grape": paired_stats(summary["rl_frozen"]["per_seed_robust_mean"], summary["grape"]["per_seed_robust_mean"],
                                            "RL(frozen)", "GRAPE"),
        # Fidelity's live-vs-frozen gap (rl_live_vs_frozen above) is large and
        # significant. But leakage is a genuinely separate question: does
        # THAT advantage also depend on live state access, or does it survive
        # freezing? Checked the same way, on leakage instead of fidelity.
        "rl_live_vs_frozen_leakage": paired_stats(summary["rl"]["per_seed_leakage_mean"], summary["rl_frozen"]["per_seed_leakage_mean"],
                                                    "RL(live) leakage", "RL(frozen) leakage"),
        "rl_frozen_vs_grape_leakage": paired_stats(summary["rl_frozen"]["per_seed_leakage_mean"], summary["grape"]["per_seed_leakage_mean"],
                                                     "RL(frozen) leakage", "GRAPE leakage"),
    }

    print("\n" + "=" * 100)
    print(f"{'Controller':24s}{'Nominal':>14s}{'Robust mean':>16s}{'Robust std':>12s}{'Leakage (robust)':>18s}")
    print("-" * 100)
    print(f"{'Naive constant pulse':24s}{naive_nominal_fid:>14.4f}{naive_fids.mean():>16.4f}{naive_fids.std():>12.4f}{naive_leaks.mean():>18.4f}")
    for name, key in [("GRAPE (nominal-only)", "grape"), ("Ensemble-GRAPE (robust)", "ensemble_grape"),
                       ("RL (live/closed-loop)", "rl"), ("RL (frozen/open-loop)", "rl_frozen")]:
        d = summary[key]
        nom_str = f"{d['nominal_mean']:.4f}" if d["nominal_mean"] is not None else "--"
        print(f"{name:24s}{nom_str:>14s}{d['robust_mean_across_seeds']:>16.4f}"
              f"{d['robust_std_across_seeds']:>12.4f}{d['leakage_mean_across_seeds']:>18.4f}")
    print("=" * 100)
    print("\nPaired significance (per-seed robust means, n=%d):" % args.n_seeds)
    for key in ["rl_vs_grape", "rl_vs_ensemble_grape", "rl_live_vs_frozen", "rl_frozen_vs_grape"]:
        st = summary["stats"][key]
        print(f"  {st['comparison']:28s} mean_diff={st['mean_diff']:+.4f}  "
              f"paired-t p={st['p_value_paired_t']}  wins={st['wins']}/{st['n']}  sign-test p={st['p_value_sign_test']:.4f}")
    print("\nPaired significance (per-seed mean leakage, n=%d):" % args.n_seeds)
    for key in ["rl_live_vs_frozen_leakage", "rl_frozen_vs_grape_leakage"]:
        st = summary["stats"][key]
        print(f"  {st['comparison']:28s} mean_diff={st['mean_diff']:+.5f}  "
              f"paired-t p={st['p_value_paired_t']}  wins={st['wins']}/{st['n']}  sign-test p={st['p_value_sign_test']:.4f}")

    with open(os.path.join(args.results_dir, "seed_study.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.results_dir, "seed_study.txt"), "w", encoding="utf-8") as f:
        f.write(f"Multi-seed study: {args.n_seeds} seeds, {args.timesteps} RL timesteps each\n")
        f.write(f"n_eval paired draws: {args.n_eval} (identical across all controllers/seeds)\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Controller':24s}{'Nominal':>14s}{'Robust mean':>16s}{'Robust std':>12s}{'Leakage':>14s}\n")
        f.write(f"{'Naive constant pulse':24s}{naive_nominal_fid:>14.4f}{naive_fids.mean():>16.4f}"
                f"{naive_fids.std():>12.4f}{naive_leaks.mean():>14.4f}\n")
        for name, key in [("GRAPE (nominal-only)", "grape"), ("Ensemble-GRAPE (robust)", "ensemble_grape"),
                           ("RL (live/closed-loop)", "rl"), ("RL (frozen/open-loop)", "rl_frozen")]:
            d = summary[key]
            nom_str = f"{d['nominal_mean']:.4f}" if d["nominal_mean"] is not None else "--"
            f.write(f"{name:24s}{nom_str:>14s}{d['robust_mean_across_seeds']:>16.4f}"
                    f"{d['robust_std_across_seeds']:>12.4f}{d['leakage_mean_across_seeds']:>14.4f}\n")
        f.write("\nPaired significance (per-seed robust means):\n")
        for key in ["rl_vs_grape", "rl_vs_ensemble_grape", "rl_live_vs_frozen", "rl_frozen_vs_grape"]:
            st = summary["stats"][key]
            f.write(f"  {st['comparison']:28s} mean_diff={st['mean_diff']:+.4f}  "
                    f"paired-t p={st['p_value_paired_t']}  wins={st['wins']}/{st['n']}  sign-test p={st['p_value_sign_test']:.4f}\n")
        f.write("\nPaired significance (per-seed mean leakage):\n")
        for key in ["rl_live_vs_frozen_leakage", "rl_frozen_vs_grape_leakage"]:
            st = summary["stats"][key]
            f.write(f"  {st['comparison']:28s} mean_diff={st['mean_diff']:+.5f}  "
                    f"paired-t p={st['p_value_paired_t']}  wins={st['wins']}/{st['n']}  sign-test p={st['p_value_sign_test']:.4f}\n")
        f.write("\nPer-seed detail:\n")
        for name, key in [("GRAPE", "grape"), ("Ensemble-GRAPE", "ensemble_grape"),
                           ("RL (live)", "rl"), ("RL (frozen)", "rl_frozen")]:
            d = summary[key]
            f.write(f"\n{name}:\n")
            for s in seeds:
                nom = f"{d['per_seed_nominal'][s]:.4f}" if d["per_seed_nominal"] is not None else "--"
                f.write(f"  seed {s}: nominal={nom}  robust_mean={d['per_seed_robust_mean'][s]:.4f}  "
                        f"leak={d['per_seed_leakage_mean'][s]:.4f}\n")

    print(f"\n[seed_study] Full results saved -> {args.results_dir}/seed_study.json, seed_study.txt")

    # ---- 6. Plot: nominal & robust fidelity per seed, per controller ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))
    controllers = [("GRAPE", grape_nominals, grape_fids_per_seed, grape_leaks_per_seed, "#E53935"),
                   ("Ens-GRAPE", ens_nominals, ens_fids_per_seed, ens_leaks_per_seed, "#8E24AA"),
                   ("RL(live)", rl_nominals, rl_fids_per_seed, rl_leaks_per_seed, "#1E88E5"),
                   ("RL(frozen)", None, frozen_fids_per_seed, frozen_leaks_per_seed, "#43A047")]
    x = np.arange(len(controllers))
    for ax, mode, title in [(ax1, "nominal", "Nominal fidelity by seed"),
                             (ax2, "robust", "Robust mean fidelity by seed"),
                             (ax3, "leakage", "Robust mean leakage by seed")]:
        for i, (name, noms, fids_list, leaks_list, color) in enumerate(controllers):
            if mode == "nominal":
                if noms is None:
                    continue
                vals = noms
            elif mode == "robust":
                vals = [f.mean() for f in fids_list]
            else:
                vals = [l.mean() for l in leaks_list]
            ax.scatter([i] * len(vals), vals, color=color, alpha=0.7, s=60, zorder=3)
            ax.errorbar([i], [np.mean(vals)], yerr=[np.std(vals)], fmt="_", color="black",
                        markersize=30, capsize=6, zorder=4)
        if mode == "nominal":
            ax.axhline(naive_nominal_fid, color="gray", linestyle="--", alpha=0.6, label="naive pulse")
        elif mode == "robust":
            ax.axhline(naive_fids.mean(), color="gray", linestyle="--", alpha=0.6, label="naive pulse")
        else:
            ax.axhline(naive_leaks.mean(), color="gray", linestyle="--", alpha=0.6, label="naive pulse")
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in controllers])
        ax.set_ylabel("Leakage P2" if mode == "leakage" else "Fidelity")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "seed_study.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[seed_study] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
