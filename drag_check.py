"""
drag_check.py
--------------
Validates the DRAG-correlation detector against a known analytic DRAG pulse
(a synthetic Gaussian Omega_x(t) with its exact DRAG-derived Omega_y(t)),
then runs the same check on trained RL policies' frozen, nominal-rollout
pulses -- so the negative result reported in SCIENCE.md, Section 7.5 (the
policy does not rediscover DRAG) can be trusted rather than attributed to a
broken detector.

With --model-root, runs across all 5 seed_study.py models (matching the
project's multi-seed convention elsewhere) and reports mean/std/range, not
just a single model's number -- this check is cheap (one nominal rollout
per seed, no optimization) so there's no reason to settle for n=1.

Usage
-----
    python drag_check.py --model models/best_model.zip
    python drag_check.py --model-root models_seedstudy --n-seeds 5
"""

import os
import sys
import argparse

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from grape_baseline import distill_policy, drag_correlation_check, validate_drag_detector
from utils import QUBIT_PARAMS


def parse_args():
    p = argparse.ArgumentParser(description="Validate the DRAG-correlation detector, then run it on trained policies")
    p.add_argument("--model", type=str, default="models/best_model.zip",
                    help="Single model path (ignored if --model-root is given)")
    p.add_argument("--model-root", type=str, default=None,
                    help="Directory of seed{i}/best_model.zip models (e.g. models_seedstudy); overrides --model")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=0.35)
    p.add_argument("--leakage-pen", type=float, default=1.0)
    p.add_argument("--energy-pen", type=float, default=0.01)
    p.add_argument("--bandwidth-tau", type=float, default=0.5)
    p.add_argument("--detuning-std", type=float, default=0.03)
    p.add_argument("--amp-error-std", type=float, default=0.05)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    alpha = QUBIT_PARAMS["alpha"]

    val_corr, val_r2, val_scale = validate_drag_detector(alpha, args.n_steps, args.dt, args.omega_max)
    print(f"[drag-check] Detector self-test (synthetic analytic DRAG pulse): "
          f"corr={val_corr:.4f}  R^2={val_r2:.4f}  fit_scale={val_scale:.4f}")

    from stable_baselines3 import PPO

    if args.model_root:
        model_paths = [(s, os.path.join(args.model_root, f"seed{s}", "best_model.zip")) for s in range(args.n_seeds)]
    else:
        model_paths = [(None, args.model)]

    per_seed = []
    for seed, path in model_paths:
        model = PPO.load(path)
        frozen_actions = distill_policy(model, args)
        corr, r2, scale = drag_correlation_check(frozen_actions, args.dt, args.omega_max, alpha)
        label = f"seed {seed}" if seed is not None else path
        print(f"[drag-check] {label}: corr={corr:.4f}  R^2={r2:.4f}  fit_scale={scale:.4f}")
        per_seed.append((seed, path, corr, r2, scale))

    out_path = os.path.join(args.results_dir, "drag_check.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("DRAG-correlation detector: self-test + trained-policy check\n")
        f.write("=" * 62 + "\n")
        f.write("Detector self-test (synthetic Gaussian pulse + its EXACT analytic\n")
        f.write("DRAG-derived Omega_y(t)) -- should be near-perfect if the detector works:\n")
        f.write(f"  correlation = {val_corr:.4f}\n  R^2         = {val_r2:.4f}\n  fit_scale   = {val_scale:.4f}\n\n")
        f.write("Trained policy/policies (frozen, one nominal rollout each):\n")
        corrs = np.array([c for _, _, c, _, _ in per_seed])
        r2s = np.array([r for _, _, _, r, _ in per_seed])
        for seed, path, corr, r2, scale in per_seed:
            label = f"seed {seed}" if seed is not None else path
            f.write(f"  {label:12s}  correlation={corr:+.4f}  R^2={r2:+.4f}  fit_scale={scale:+.4f}\n")
        if len(per_seed) > 1:
            f.write(f"\n  Across {len(per_seed)} seeds: correlation mean={corrs.mean():+.4f} std={corrs.std():.4f} "
                    f"range=[{corrs.min():+.4f}, {corrs.max():+.4f}]\n")
            f.write(f"  Across {len(per_seed)} seeds: R^2 mean={r2s.mean():+.4f} std={r2s.std():.4f} "
                    f"range=[{r2s.min():+.4f}, {r2s.max():+.4f}]\n")
    print(f"[drag-check] Saved -> {out_path}")


if __name__ == "__main__":
    main()
