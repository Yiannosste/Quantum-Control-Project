"""
evaluate.py
-----------
Load a trained PPO model and evaluate it on the quantum pulse environment.

Usage
-----
    python evaluate.py                           # load best_model.zip
    python evaluate.py --model models/ppo_quantum_pulse.zip
    python evaluate.py --n-episodes 20

Outputs
-------
    results/pulse_bloch_trajectory.png   : pulse shape + Bloch sphere plot
    results/eval_summary.txt             : fidelity statistics
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO

from quantum_env import QuantumPulseEnv
from utils import QUBIT_PARAMS, plot_pulse_and_bloch, compute_fidelity
import qutip as qt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained quantum pulse agent")
    p.add_argument("--model",       type=str, default="models/best_model.zip",
                   help="Path to the saved SB3 model (.zip)")
    p.add_argument("--n-episodes",  type=int, default=10,
                   help="Number of evaluation episodes (default: 10)")
    p.add_argument("--n-steps",     type=int, default=50)
    p.add_argument("--dt",          type=float, default=1.0)
    p.add_argument("--omega-max",   type=float, default=0.1)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic policy (recommended for eval)")
    return p.parse_args()


def run_episode(model, env, deterministic: bool = True):
    """Run a single episode and return fidelity + pulse amplitudes."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    fidelity = info.get("fidelity", 0.0)
    amplitudes = info.get("pulse_amplitudes", [])
    return fidelity, total_reward, amplitudes


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Check model path ──────────────────────────────────────────────────
    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback: try the final model
        fallback = "models/ppo_quantum_pulse.zip"
        if os.path.exists(fallback):
            model_path = fallback
            print(f"[eval] best_model.zip not found — using {fallback}")
        else:
            raise FileNotFoundError(
                f"No model found at '{args.model}' or '{fallback}'.\n"
                "Run  python train.py  first."
            )

    print("=" * 60)
    print("  RL-Driven Quantum Pulse Optimization — Evaluation")
    print("=" * 60)
    print(f"  Model        : {model_path}")
    print(f"  Episodes     : {args.n_episodes}")
    print(f"  Gate time    : {args.n_steps * args.dt:.0f} ns")
    print("=" * 60)

    # ── Load model & environment ──────────────────────────────────────────
    model = PPO.load(model_path)

    env = QuantumPulseEnv(
        n_steps=args.n_steps,
        dt=args.dt,
        omega_max=args.omega_max,
        params=QUBIT_PARAMS,
    )

    # ── Run evaluation episodes ───────────────────────────────────────────
    fidelities = []
    all_rewards = []
    best_fidelity = -1.0
    best_amplitudes = None

    for ep in range(args.n_episodes):
        fidelity, reward, amplitudes = run_episode(
            model, env, deterministic=args.deterministic
        )
        fidelities.append(fidelity)
        all_rewards.append(reward)
        print(f"  Episode {ep+1:>3d} | Fidelity: {fidelity:.4f} | "
              f"Reward: {reward:.4f}")

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_amplitudes = amplitudes

    # ── Summary statistics ────────────────────────────────────────────────
    fid_arr = np.array(fidelities)
    print("\n" + "─" * 50)
    print(f"  Mean fidelity  : {fid_arr.mean():.4f} ± {fid_arr.std():.4f}")
    print(f"  Best fidelity  : {fid_arr.max():.4f}")
    print(f"  Worst fidelity : {fid_arr.min():.4f}")
    print(f"  Episodes ≥ 0.99 fidelity: "
          f"{(fid_arr >= 0.99).sum()}/{args.n_episodes}")
    print("─" * 50)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(args.results_dir, "eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RL Quantum Pulse Optimization — Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model       : {model_path}\n")
        f.write(f"Episodes    : {args.n_episodes}\n")
        f.write(f"Gate time   : {args.n_steps * args.dt:.0f} ns\n")
        f.write(f"T1 / T2     : {QUBIT_PARAMS['T1']:.0f} / {QUBIT_PARAMS['T2']:.0f} ns\n\n")
        f.write(f"Mean fidelity  : {fid_arr.mean():.6f}\n")
        f.write(f"Std fidelity   : {fid_arr.std():.6f}\n")
        f.write(f"Best fidelity  : {fid_arr.max():.6f}\n")
        f.write(f"Worst fidelity : {fid_arr.min():.6f}\n")
        f.write(f"Episodes ≥0.99 : {(fid_arr >= 0.99).sum()}/{args.n_episodes}\n\n")
        f.write("Per-episode fidelities:\n")
        for i, fid in enumerate(fidelities):
            f.write(f"  Ep {i+1:>3d}: {fid:.6f}\n")
    print(f"\n[eval] Summary saved → {summary_path}")

    # ── Visualise best episode ────────────────────────────────────────────
    if best_amplitudes:
        print(f"\n[eval] Plotting best episode (fidelity={best_fidelity:.4f}) ...")
        trajectory = env.get_bloch_trajectory(best_amplitudes)
        plot_path = os.path.join(args.results_dir, "pulse_bloch_trajectory.png")
        plot_pulse_and_bloch(
            amplitudes=np.array(best_amplitudes),
            trajectory=trajectory,
            dt=args.dt,
            fidelity=best_fidelity,
            save_path=plot_path,
        )

    # ── Print ideal (noise-free) benchmark ───────────────────────────────
    print("\n[eval] Computing noise-free (T1=T2=∞) benchmark ...")
    env_ideal = QuantumPulseEnv(
        n_steps=args.n_steps,
        dt=args.dt,
        omega_max=args.omega_max,
        params={"omega_q": 0.0, "T1": 1e9, "T2": 1e9},
    )
    fid_ideal, _, _ = run_episode(model, env_ideal, deterministic=True)
    print(f"  Ideal fidelity (no noise): {fid_ideal:.4f}")
    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
