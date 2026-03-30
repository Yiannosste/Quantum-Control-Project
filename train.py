"""
train.py
--------
Train a PPO agent to optimize quantum control pulses for an X-gate.

Usage
-----
    python train.py                      # default hyperparameters
    python train.py --timesteps 500000   # longer training
    python train.py --help               # all options

Outputs
-------
    models/ppo_quantum_pulse.zip         : trained SB3 model
    logs/                                : TensorBoard logs
    results/training_rewards.png         : reward curve
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for training

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from quantum_env import QuantumPulseEnv
from utils import QUBIT_PARAMS, plot_training_rewards


# ── Reward-tracking callback ──────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Collects per-episode rewards for post-training plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self._current_reward: float = 0.0

    def _on_step(self) -> bool:
        self._current_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO for quantum pulse optimization")
    p.add_argument("--timesteps",    type=int,   default=200_000,
                   help="Total training timesteps (default: 200 000)")
    p.add_argument("--n-steps",      type=int,   default=50,
                   help="Episode length / time steps per episode (default: 50)")
    p.add_argument("--dt",           type=float, default=1.0,
                   help="Timestep duration in ns (default: 1.0)")
    p.add_argument("--omega-max",    type=float, default=0.1,
                   help="Max drive amplitude 2pi*MHz (default: 0.1)")
    p.add_argument("--energy-pen",   type=float, default=0.01,
                   help="Pulse-energy penalty coefficient (default: 0.01)")
    p.add_argument("--lr",           type=float, default=3e-4,
                   help="PPO learning rate (default: 3e-4)")
    p.add_argument("--batch-size",   type=int,   default=64,
                   help="PPO mini-batch size (default: 64)")
    p.add_argument("--n-epochs",     type=int,   default=10,
                   help="PPO update epochs (default: 10)")
    p.add_argument("--n-envs",       type=int,   default=1,
                   help="Parallel environments (default: 1)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--model-dir",    type=str,   default="models")
    p.add_argument("--log-dir",      type=str,   default="logs")
    p.add_argument("--results-dir",  type=str,   default="results")
    return p.parse_args()


# ── Main training routine ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create output directories
    for d in [args.model_dir, args.log_dir, args.results_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  RL-Driven Quantum Pulse Optimization — Training")
    print("=" * 60)
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  Episode len  : {args.n_steps} steps × {args.dt} ns = "
          f"{args.n_steps * args.dt:.0f} ns gate time")
    print(f"  Omega max    : {args.omega_max:.3f} × 2π MHz")
    print(f"  T1 / T2      : {QUBIT_PARAMS['T1']:.0f} ns / {QUBIT_PARAMS['T2']:.0f} ns")
    print("=" * 60)

    # ── Environment factory ───────────────────────────────────────────────
    env_kwargs = dict(
        n_steps=args.n_steps,
        dt=args.dt,
        omega_max=args.omega_max,
        energy_penalty=args.energy_pen,
        params=QUBIT_PARAMS,
    )

    def make_env():
        env = QuantumPulseEnv(**env_kwargs)
        env = Monitor(env)
        return env

    # Training environment (vectorised for future multi-env support)
    train_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    # Separate evaluation environment
    eval_env = Monitor(QuantumPulseEnv(**env_kwargs))

    # ── PPO agent ─────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps * 4,          # rollout buffer length
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,                     # mild entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=args.seed,
        verbose=1,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    reward_logger = RewardLogger()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=max(args.n_steps * 50, 2000),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix="ppo_qpulse",
    )

    # ── Training ──────────────────────────────────────────────────────────
    print("\n[train] Starting PPO training ...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[reward_logger, eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(args.model_dir, "ppo_quantum_pulse")
    model.save(final_path)
    print(f"\n[train] Model saved → {final_path}.zip")

    # ── Plot reward curve ─────────────────────────────────────────────────
    rewards = reward_logger.episode_rewards
    if rewards:
        np.save(os.path.join(args.results_dir, "episode_rewards.npy"), rewards)
        plot_training_rewards(
            rewards,
            window=50,
            save_path=os.path.join(args.results_dir, "training_rewards.png"),
        )
        print(f"[train] Final 100-ep mean reward: "
              f"{np.mean(rewards[-100:]):.4f}")

    print("\n[train] Done. Run  python evaluate.py  to test the trained agent.")


if __name__ == "__main__":
    main()
