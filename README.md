# RL-Driven Quantum Pulse Optimization

> **Reinforcement Learning for minimum-time, high-fidelity X-gate synthesis on a superconducting qubit under realistic decoherence.**

---

## Overview

This project demonstrates how a **Proximal Policy Optimization (PPO)** agent can learn to shape a microwave control pulse that steers a superconducting qubit from the ground state |0⟩ to the excited state |1⟩ (an X-gate), while:

- maximizing **gate fidelity**,
- minimizing **pulse energy** (leakage & heating concern), and
- operating under **open-system decoherence** (T1 relaxation + T2 dephasing).

The qubit dynamics are simulated with **QuTiP** using the Lindblad master equation. The RL agent is trained with **Stable-Baselines3** through a custom **Gymnasium** environment.

---

## Physics Background

### Two-Level System Hamiltonian

In the rotating-wave approximation (RWA) at resonance, the system Hamiltonian splits into:

```
H(t) = H₀ + Ω(t) · Hc

H₀ = (ω_q / 2) σ_z          [drift — qubit frequency]
Hc = (1/2) σ_x               [control — microwave coupling]
```

- **σ_x, σ_y, σ_z** are the Pauli matrices.
- **ω_q** is the qubit transition frequency (set to 0 in the rotating frame for resonant drive).
- **Ω(t)** is the time-varying drive amplitude learned by the RL agent.

### Lindblad Master Equation (Open System)

To model real superconducting qubits, we include two decoherence channels:

```
dρ/dt = -i[H(t), ρ] + Σ_k ( L_k ρ L_k† - ½{L_k†L_k, ρ} )
```

| Channel | Collapse operator | Physical meaning |
|---------|-------------------|-----------------|
| T1 (energy relaxation) | `√(Γ₁) · σ₋` | Excited state decays to ground |
| T2* (pure dephasing) | `√(Γ_φ) · σ_z / 2` | Phase randomization |

Rates: `Γ₁ = 1/T1`, `Γ_φ = 1/T2 − 1/(2·T1)`

Typical transmon values used: **T1 = 1 µs, T2 = 500 ns**.

### Gate Fidelity

```
F(ρ, |ψ_target⟩) = ⟨ψ_target| ρ |ψ_target⟩  ∈ [0, 1]
```

For the X-gate: `|ψ_target⟩ = |1⟩`.

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | Bloch vector `[⟨X⟩, ⟨Y⟩, ⟨Z⟩]` ∈ ℝ³ |
| **Action** | Normalized drive amplitude `a(t) ∈ [−1, 1]` → `Ω(t) = a(t) · Ω_max` |
| **Reward** | `r_t = −λ · Ω(t)²` (energy penalty) + `F(ρ, |1⟩)` at final step |
| **Episode** | 50 time steps × 1 ns = 50 ns gate |
| **Termination** | After `n_steps` steps |

The reward structure encourages the agent to reach |1⟩ with high fidelity **without wasting pulse energy**.

---

## Why RL for Quantum Pulse Shaping?

Classical approaches (GRAPE, CRAB) solve pulse optimization as a gradient-based optimal control problem. RL adds key advantages relevant to near-term quantum hardware:

1. **Noise robustness** — The RL agent implicitly learns policies that are robust to the stochastic decoherence environment, as each episode samples a different noise trajectory.
2. **No analytical gradient needed** — Critical when the simulator is replaced by a real QPU with hardware-in-the-loop feedback.
3. **Adaptability** — A trained policy can be fine-tuned online as qubit parameters drift.
4. **Minimum-time gates** — RL naturally discovers shorter, high-bandwidth pulses that minimise exposure to decoherence (quantum speed limit optimization).

This paradigm is actively used at leading quantum hardware companies including IQM, Alice & Bob, IBM, and Google.

---

## Repository Structure

```
Quantum-Control-Project/
│
├── quantum_env.py       # Custom Gymnasium environment (MDP)
├── utils.py             # Hamiltonians, collapse ops, fidelity, plotting
├── train.py             # PPO training script (SB3)
├── evaluate.py          # Evaluation & visualization script
│
├── tutorial.ipynb       # Educational Jupyter notebook
│
├── models/              # Saved model checkpoints (created at runtime)
│   ├── best_model.zip
│   ├── ppo_quantum_pulse.zip
│   └── checkpoints/
│
├── results/             # Output plots & statistics (created at runtime)
│   ├── training_rewards.png
│   ├── pulse_bloch_trajectory.png
│   └── eval_summary.txt
│
├── logs/                # TensorBoard logs (created at runtime)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9 – 3.11
- (Recommended) NVIDIA GPU with CUDA for faster SB3 training

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Yiannosste/Quantum-Control-Project.git
cd Quantum-Control-Project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the PPO Agent

```bash
python train.py
```

Key options:
```
--timesteps   200000    Total training timesteps
--n-steps     50        Episode length (time steps)
--dt          1.0       Step duration [ns]
--omega-max   0.1       Max drive amplitude [2π·MHz]
--energy-pen  0.01      Pulse energy penalty weight
--lr          3e-4      Learning rate
```

Monitor training live with TensorBoard:
```bash
tensorboard --logdir logs/
```

### 2. Evaluate the Trained Agent

```bash
python evaluate.py
```

Generates:
- `results/pulse_bloch_trajectory.png` — optimized pulse shape + Bloch sphere
- `results/eval_summary.txt` — fidelity statistics over 10 episodes

### 3. Interactive Tutorial

```bash
jupyter notebook tutorial.ipynb
```

---

## Results (Expected)

After ~200 k timesteps of PPO training:

| Metric | Value |
|--------|-------|
| Mean gate fidelity | ~ 0.97 |
| Best gate fidelity | ~ 0.97 |
| Gate time | 50 ns |
| Policy architecture | MLP [64, 64] |

> **Note:** Fidelity will be lower than the noise-free benchmark due to T1/T2 decoherence. The gap between noisy and ideal fidelity quantifies the decoherence impact.

---

## Technical Stack

| Library | Role |
|---------|------|
| [QuTiP](https://qutip.org) | Quantum simulation (Lindblad master equation) |
| [Gymnasium](https://gymnasium.farama.org) | RL environment interface |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io) | PPO implementation |
| [NumPy](https://numpy.org) | Numerical computation |
| [Matplotlib](https://matplotlib.org) | Visualization (Bloch sphere, pulse plots) |

---

## References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Johansson et al. (2012). *QuTiP: An open-source Python framework for the dynamics of open quantum systems.* CPC.
4. Krantz et al. (2019). *A quantum engineer's guide to superconducting qubits.* Applied Physics Reviews, 6(2).
5. Niu et al. (2019). *Universal quantum control through deep reinforcement learning.* npj Quantum Information.
6. Fösel et al. (2018). *Reinforcement learning with neural networks for quantum feedback.* Physical Review X.

---

## License

MIT License. See `LICENSE` for details.

---

*Built as part of a Quantum Engineering portfolio project.*
