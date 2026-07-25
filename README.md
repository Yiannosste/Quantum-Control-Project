# RL-Driven Quantum Pulse Optimization

> **A reinforcement learning agent is benchmarked against gradient-based optimal control (GRAPE) for X-gate synthesis on a simulated 3-level transmon, under decoherence, leakage, and unknown per-episode miscalibration.** The honest result: **GRAPE (0.9738 fidelity) beats every physically deployable RL policy (0.9692)**. RL's apparent edge (0.9788) comes from exact mid-gate state access that no real qubit readout can provide inside a 10 ns gate — a designed mechanism test proves this by freezing the policy into a realizable pulse and watching its advantage invert into a loss.

---

## What This Project Does

Trains a **PPO** agent (Stable-Baselines3, custom **Gymnasium** env) to shape a two-quadrature (I/Q) microwave pulse that drives a **3-level transmon** — QuTiP, Lindblad master equation — from |0⟩ to |1⟩ (an X-gate), while minimizing **leakage** to the |2⟩ level (a real transmon failure mode, not a toy penalty), operating under **T1/T2 decoherence** and a **bandwidth-limited** control model, and staying robust to **hidden per-episode miscalibration** (random detuning + amplitude error the agent never observes).

The interesting question isn't "can RL reach high fidelity" — it's whether RL's advantage over classical optimal control survives the constraint that a real gate has to run open-loop. It doesn't, and this project measures that rather than assuming it.

---

## Results

Five controllers, identical physics and reward, evaluated on the same 150 paired miscalibration draws across 5 independent seeds. Full derivation, per-seed data, and statistics: [SCIENCE.md](SCIENCE.md).

| Controller | Nominal fidelity | Robust mean fidelity | Robust mean leakage | Deployable? |
|---|---|---|---|---|
| Naive constant pulse | 0.9619 | 0.9505 ± 0.0211 | 0.0200 | Yes |
| **GRAPE (nominal-only)** | 0.9867 | **0.9738 ± 0.0000** | 0.0049 | Yes |
| Ensemble-GRAPE (robust) | 0.9869 | 0.9741 ± 0.0000 | 0.0046 | Yes |
| RL (live/closed-loop) | 0.9803 ± 0.0022 | 0.9788 ± 0.0020 | 0.0024 | **No** |
| RL (frozen/open-loop) | — | 0.9692 ± 0.0021 | 0.0024 | Yes |

*"Deployable" = runnable without the mid-gate feedback a real qubit readout can't provide inside a 10 ns gate. Among deployable controllers, GRAPE wins.*

- **The mechanism test.** RL (live) beats both GRAPE variants — but it also sees the exact quantum state every step, which real hardware can't. Freezing that same trained policy into a fixed, open-loop pulse ("RL frozen") isolates whether the edge is a better waveform or exploited information. It's the latter: frozen RL scores *below* nominal GRAPE (p = 0.0117, 0 of 5 seeds ahead) — freezing doesn't just erase RL's advantage, it inverts it.
- **The exception: leakage.** RL's ~2× leakage advantage over both GRAPE variants survives freezing almost intact (0.0024 vs. 0.0024) — a genuinely learned, deployable benefit, unlike the fidelity edge.
- **Does training on the miscalibration distribution alone help GRAPE?** Barely — ensemble GRAPE beats nominal GRAPE by only +0.0002, an order of magnitude below RL's edge. Distributional knowledge isn't what RL is exploiting; live state access is.
- **Does the policy rediscover DRAG?** Mostly no — validated the detector against a synthetic ground truth first, then found 4 of 5 seeds show no meaningful match to the textbook correction. One seed converges close to it anyway (R² = +0.73). Full per-seed breakdown: [SCIENCE.md, §7.5](SCIENCE.md).

**Why this is trustworthy, not just measured:** two fair GRAPE baselines instead of a strawman, a designed mechanism test instead of a scoreboard, five seeds with paired statistics instead of one lucky run, a validated detector before trusting a negative result — and a real physics bug (T2 dephasing was silently running 2.29× too weak: 1142.9 ns simulated vs. a labeled 500 ns) caught by testing the actual production code path, fixed, and now guarded by a standing regression test (`tests/test_physics.py`). Details: [SCIENCE.md, §2](SCIENCE.md).

---

## Repository Structure

```
Quantum-Control-Project/
├── quantum_env.py       # Gymnasium environment (MDP, 3-level + miscalibration)
├── utils.py              # Hamiltonians, collapse ops, fidelity, plotting
├── train.py               # PPO training script
├── evaluate.py           # Evaluation & visualization
├── grape_baseline.py     # GRAPE + ensemble-GRAPE, distillation test, paired-eval harness
├── seed_study.py          # Full 5-seed, 5-controller robustness + mechanism study
├── drag_check.py          # DRAG-correlation detector validation + multi-seed check
├── tests/test_physics.py # Regression test: measured T1/T2 must match QUBIT_PARAMS
├── tutorial.ipynb         # Educational notebook (same physics, same experiments)
├── SCIENCE.md              # Full derivations, methodology, and results
├── models/, results/, logs/  # Generated at runtime
└── requirements.txt, LICENSE
```

## Installation

Python 3.9 – 3.11. No GPU needed — the policy is a small [64, 64] MLP over an 8-D state space.

```bash
git clone https://github.com/Yiannosste/Quantum-Control-Project.git && cd Quantum-Control-Project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/                                         # physics regression tests, ~3s
```

## Usage

| Command | What it does |
|---|---|
| `python train.py` | Train PPO (default 200k timesteps, ~25-30 min CPU) |
| `python evaluate.py` | Evaluate trained agent; plots pulse/Bloch trajectory |
| `python grape_baseline.py --model models/best_model.zip` | Fast single-seed 5-controller check |
| `python seed_study.py --n-seeds 5` | **The real study** — 5 seeds, paired stats. Hours on CPU |
| `python drag_check.py --model-root models_seedstudy --n-seeds 5` | Validated DRAG-correlation check |
| `jupyter notebook tutorial.ipynb` | Interactive walkthrough of everything above |

---

## Technical Stack

[QuTiP](https://qutip.org) (Lindblad simulation) · [Gymnasium](https://gymnasium.farama.org) (RL env) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io) (PPO) · [SciPy](https://scipy.org) (GRAPE optimization, statistics) · NumPy · Matplotlib

## References

1. Khaneja et al. (2005). *Optimal control of coupled spin dynamics.* J. Magnetic Resonance. (GRAPE)
2. Motzoi et al. (2009). *Simple pulses for elimination of leakage in weakly nonlinear qubits.* PRL. (DRAG)
3. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
4. Krantz et al. (2019). *A quantum engineer's guide to superconducting qubits.* Applied Physics Reviews.
5. Johansson et al. (2012). *QuTiP.* Computer Physics Communications.

Full reference list with RL-in-quantum-control citations: [SCIENCE.md](SCIENCE.md).

---

## License

MIT License. See `LICENSE`.

---

*Built as part of a Quantum Engineering portfolio project — demonstrating not just that a policy can be trained, but the full discipline of proving what it actually learned.*
