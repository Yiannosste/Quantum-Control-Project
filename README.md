# RL-Driven Quantum Pulse Optimization

> **A controlled comparison of reinforcement learning against gradient-based optimal control (GRAPE) for X-gate synthesis on a simulated 3-level transmon — and a mechanism test showing that every physically deployable version of the RL policy loses to GRAPE.** RL's apparent edge comes from mid-gate access to exact simulator state, which destructive, microsecond-scale qubit readout cannot provide inside a 10 ns gate; freezing the policy into a pulse a real controller could run erases that edge and drops it below the GRAPE baseline. Physics in an aggressive noise regime (T1 = 1 µs) chosen to make decoherence, leakage, and miscalibration all matter — not to represent state-of-the-art hardware. The leakage/miscalibration *structure* modeled here, not the specific T1/T2 numbers, is what's meant to generalize.

---

## Overview

This project trains a **Proximal Policy Optimization (PPO)** agent to shape a two-quadrature (I/Q) microwave control pulse that steers a **3-level transmon** (|0⟩, |1⟩, and the leakage level |2⟩) from the ground state |0⟩ to the excited state |1⟩ — a single-input-state transfer, not a full average-gate-fidelity characterization (see [SCIENCE.md, Section 3](SCIENCE.md)) — while:

- maximizing **gate fidelity**,
- minimizing **leakage** to |2⟩ (a real transmon failure mode that breaks error correction, not a toy penalty),
- minimizing **pulse energy**,
- operating under **open-system decoherence** (T1 relaxation + T2 dephasing),
- staying within a **bandwidth-limited** control model (no instantaneous pulse jumps), and
- remaining robust to **unknown per-episode miscalibration** (random detuning + amplitude error the agent never observes directly).

The central question this project answers is not "can RL reach high fidelity" (that's necessary but not interesting on its own) — it's **does RL's advantage over gradient-based optimal control survive the constraint that a real gate has to run open-loop?** Measured, not assumed, the answer is **no**: GRAPE (0.9765) beats every physically deployable version of the RL policy (0.9718); the closed-loop RL number that looks better (0.9818) depends on mid-gate state access no real qubit readout can provide inside a 10 ns gate. Three things make that a measured answer rather than a claim:

1. **Two GRAPE baselines, not one.** `grape_baseline.py` implements standard GRAPE (Khaneja et al. 2005), optimized against the known nominal qubit, *and* ensemble/robust GRAPE, optimized for average performance over the same miscalibration distribution RL trains on. Comparing RL only against a GRAPE pulse that never saw the miscalibration distribution would be comparing a distribution-trained method against a point-optimized one — not a fair test.
2. **A mechanism test, not just a scoreboard.** RL observes the simulator's exact quantum state every step — real qubit readout cannot do this mid-gate (destructive, hundreds of ns to µs, against a 10 ns gate here). So any RL-vs-GRAPE gap could mean "RL found a more robust waveform" *or* "RL is exploiting information no open-loop pulse could use." `seed_study.py` settles this directly: it freezes each trained policy into a fixed, open-loop pulse and re-evaluates it exactly like GRAPE — and the frozen pulse loses to GRAPE.
3. **Five seeds, paired evaluation, real statistics.** RL is trained across 5 independent seeds; GRAPE and ensemble-GRAPE are each optimized across 5 independent restart seeds; every controller is evaluated on an identical set of paired miscalibration draws; results are reported as mean ± std **across seeds**, with paired t-tests and sign tests, not just a single run's numbers.

The qubit dynamics are simulated with **QuTiP** using the Lindblad master equation. The RL agent is trained with **Stable-Baselines3** through a custom **Gymnasium** environment.

---

## Physics Background

> This section is a summary. For the full derivations, worked numerical
> checks, and explicitly-noted simplifications, see [SCIENCE.md](SCIENCE.md).

### 3-Level Transmon Hamiltonian

A transmon is a *weakly anharmonic* oscillator, not a true two-level system. Using the number operator `n = a†a` on a 3-level truncated Hilbert space:

```
H(t) = H₀ + Ωx(t)·Hx + Ωy(t)·Hy

H₀ = -Δ·n + (α/2)·n(n-1)     [drift — detuning + anharmonicity]
Hx = (a + a†) / 2             [control — in-phase ("I") coupling]
Hy = i(a† - a) / 2            [control — quadrature ("Q") coupling]
```

- **Δ** is the qubit-drive detuning (0 at resonance).
- **α** is the anharmonicity (typically ≈ -2π·250 MHz for a transmon) — 0 at levels 0,1; contributes exactly α at level 2, which is *why* a resonant 0-1 drive only weakly leaks into |2⟩ instead of driving it resonantly.
- **Ωx(t), Ωy(t)** are the two independent drive quadratures the RL agent controls. Ωy is the same real degree of freedom DRAG pulses use to cancel leakage — the agent isn't handed the DRAG formula, it has to find its own correction ([does it converge on DRAG? see below](#results-measured)).
- Truncating to 2 levels recovers the ordinary two-level qubit model exactly (`a→σ₋`, `Hx→σx/2`).

### Lindblad Master Equation (Open System)

```
dρ/dt = -i[H(t), ρ] + Σ_k ( L_k ρ L_k† - ½{L_k†L_k, ρ} )
```

| Channel | Collapse operator | Physical meaning |
|---------|-------------------|-----------------|
| T1 (energy relaxation) | `√(Γ₁) · a` | 1→0 decay at Γ₁, 2→1 decay at 2Γ₁ (harmonic-oscillator scaling, from `⟨n-1\|a\|n⟩=√n`) |
| T2* (pure dephasing) | `√(2Γ_φ) · n / 2` | Phase randomization |

Rates: `Γ₁ = 1/T1`, `Γ_φ = 1/T2 − 1/(2·T1)`. The dephasing operator's prefactor matters: a naively-transplanted two-level form (`√Γ_φ · σz/2`) gives *half* the intended decay rate — verified by direct simulation (measured 800 ns vs. a labeled 500 ns for the wrong form; the form above reproduces 500 ns exactly). See [SCIENCE.md, Section 2](SCIENCE.md) for the derivation and verification numbers.

Typical transmon values used: **T1 = 1 µs, T2 = 500 ns, α = -2π·250 MHz**.

### Gate Fidelity

```
F(ρ, |ψ_target⟩) = ⟨ψ_target| ρ |ψ_target⟩  ∈ [0, 1]
```

For the X-gate: `|ψ_target⟩ = |1⟩`. This is single-input-state transfer fidelity (|0⟩→|1⟩ only), not the full average gate fidelity used in randomized benchmarking — see [SCIENCE.md, Section 3](SCIENCE.md) for the distinction.

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | 8-D generalized Bloch vector `[P₁, P₂, Re ρ₀₁, Im ρ₀₁, Re ρ₁₂, Im ρ₁₂, Re ρ₀₂, Im ρ₀₂]` |
| **Action** | Normalized I/Q drive `(aₓ, a_y) ∈ [−1,1]²` → `Ωₓ(t)=aₓ·Ω_max`, `Ω_y(t)=a_y·Ω_max` |
| **Reward** | `r_t = −λ·(Ωₓ²+Ω_y²) − μ·P₂(t)` every step, `+ F(ρ, \|1⟩)` at the final step |
| **Episode** | 10 time steps × 1 ns = 10 ns gate |
| **Termination** | After `n_steps` steps |
| **Control realism** | Commands pass through a bandwidth-limiting RC filter (`bandwidth_tau`, default 0.5 ns) before reaching the qubit — no instantaneous pulse jumps |
| **Miscalibration** | Each episode samples a hidden detuning (`detuning_std`, default 0.03 rad/ns) and amplitude gain error (`amp_error_std`, default 5%), **not included in the observation** |

**On the state observation, precisely:** $s_t$ is read directly off the noiseless simulator density matrix — exact populations and coherences. The agent's observation never includes the hidden detuning/amplitude error, but it *does* include information no real qubit readout could provide mid-gate (destructive, hundreds of ns to µs, against a 10 ns gate). This is standard practice for simulation-based RL, not a claim about hardware feasibility — and it's exactly what [the mechanism test below](#results-measured) is designed to account for. The reward encourages the agent to reach |1⟩ with high fidelity **without wasting pulse energy or leaking to |2⟩**.

> **Why a 10 ns gate?** It's decoherence-limited *and* leakage-limited, and 10 ns is only a good choice because the agent has a real leakage-suppression channel to use — a naive pulse actually does *better* at 20 ns. Full derivation: [SCIENCE.md, Section 3](SCIENCE.md#3-gate-fidelity).

---

## Methodology: Five Controllers, One Physics, One Reward

| # | Controller | Sees miscalibration distribution? | Sees live mid-episode state? |
|---|---|---|---|
| 1 | Naive constant pulse | No | No |
| 2 | GRAPE (nominal-only) | No | No |
| 3 | Ensemble-GRAPE (robust) | Yes (optimized for it) | No |
| 4 | RL (live/closed-loop) | Yes (trained on it) | Yes |
| 5 | RL (frozen/open-loop) | Yes (inherited from training) | **No** — same waveform as #4, replayed blind |

Controllers 1–3 isolate whether distributional training alone (which GRAPE can also get, via #3) explains a robustness gap. Controller 5 isolates whether RL's *live* state access (which no open-loop method can have) explains the rest. All five are optimized/trained and evaluated against the identical reward and physics, across 5 seeds, on an identical set of paired evaluation draws. Full procedure: [SCIENCE.md, Section 5 & 7](SCIENCE.md).

---

## Repository Structure

```
Quantum-Control-Project/
│
├── quantum_env.py       # Custom Gymnasium environment (MDP, 3-level + miscalibration)
├── utils.py             # Hamiltonians, collapse ops, fidelity, plotting
├── train.py             # PPO training script (SB3)
├── evaluate.py          # Evaluation & visualization script
├── grape_baseline.py    # GRAPE + ensemble-GRAPE optimizers, distillation test, paired-eval harness
├── seed_study.py        # Full 5-seed, 5-controller robustness + mechanism study
├── drag_check.py        # DRAG-correlation detector validation + multi-seed policy check
│
├── tutorial.ipynb       # Educational Jupyter notebook
│
├── models/              # Saved model checkpoints (created at runtime)
│   ├── best_model.zip
│   ├── ppo_quantum_pulse.zip
│   └── checkpoints/
│
├── results/              # Output plots & statistics (created at runtime)
│   ├── training_rewards.png
│   ├── pulse_bloch_trajectory.png    # I/Q pulse + populations + Bloch sphere
│   ├── eval_summary.txt
│   ├── grape_comparison.txt          # single-seed 5-controller comparison
│   ├── grape_vs_rl_robustness.png    # single-seed fidelity distributions
│   ├── seed_study.txt / .json        # full 5-seed comparison, mean +- std across seeds, stats
│   ├── seed_study.png                # per-seed nominal/robust fidelity & leakage, all controllers
│   └── drag_check.txt                # DRAG-detector self-test + per-seed policy correlation/R^2
│
├── logs/                # TensorBoard logs (created at runtime)
├── requirements.txt
├── README.md
├── SCIENCE.md           # Detailed physics, methodology, and full results
└── LICENSE
```

---

## Installation

### Prerequisites
- Python 3.9 – 3.11
- No GPU required — the policy network is a small [64, 64] MLP over an 8-D
  state space, which trains fast on CPU. A GPU generally will not speed up
  training for a model this small (and may even be slower due to
  host↔device transfer overhead).

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
--timesteps        200000   Total training timesteps
--n-steps          10       Episode length (time steps)
--dt               1.0      Step duration [ns]
--omega-max        0.35     Max drive amplitude per quadrature [rad/ns]
--energy-pen       0.01     Pulse energy penalty weight
--leakage-pen      1.0      Per-step |2>-population penalty weight
--bandwidth-tau    0.5      AWG bandwidth RC time constant [ns] (0 disables)
--detuning-std     0.03     Per-episode hidden detuning stddev [rad/ns]
--amp-error-std    0.05     Per-episode hidden amplitude-gain error stddev
--lr               3e-4     Learning rate
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
- `results/pulse_bloch_trajectory.png` — I/Q pulse shape + level populations (leakage) + Bloch sphere
- `results/eval_summary.txt` — fidelity/leakage statistics over 10 episodes, each with a fresh random miscalibration draw

### 3. GRAPE / Ensemble-GRAPE / Distillation Baseline (single seed, quick check)

```bash
python grape_baseline.py --model models/best_model.zip
```

Optimizes a nominal GRAPE pulse and an ensemble (robust) GRAPE pulse, freezes the RL policy into an open-loop pulse (the mechanism test), and evaluates all five controllers on an identical set of paired miscalibration draws. Generates `results/grape_comparison.txt` and `results/grape_vs_rl_robustness.png`. Fast, single-seed — for the number to cite, use `seed_study.py`.

### 4. Full Multi-Seed Robustness + Mechanism Study

```bash
python seed_study.py --n-seeds 5 --timesteps 200000
```

Trains RL across 5 seeds, runs GRAPE/ensemble-GRAPE across 5 optimization seeds, freezes each trained policy for the distillation/mechanism test, evaluates every controller on the same paired draws, and reports **mean ± std across seeds** plus paired significance tests. This is the number to trust — a single-seed comparison cannot distinguish a real effect from training-seed noise. Takes a few hours on CPU (reusing already-trained models in `--model-root` skips the RL-training stage). Generates `results/seed_study.txt`, `.json`, and `.png`.

### 5. DRAG-Correlation Detector Check

```bash
python drag_check.py --model-root models_seedstudy --n-seeds 5
```

Validates the DRAG-correlation detector against a synthetic pulse with an exact analytic DRAG pair (should recover correlation = R² = 1.0000), then runs the same check on all 5 trained policies. Generates `results/drag_check.txt`. Fast — one nominal rollout per seed, no optimization.

### 6. Interactive Tutorial

```bash
jupyter notebook tutorial.ipynb
```

---

## Results (Measured)

All numbers below are read verbatim from `results/seed_study.txt` (5 seeds, 150 shared paired miscalibration draws per seed, RL = 200,000 PPO timesteps/seed). Full derivation and discussion: [SCIENCE.md, Section 7](SCIENCE.md).

**On this benchmark, gradient-based open-loop optimization (GRAPE, 0.9765) outperforms every physically deployable RL policy (0.9718).** RL's apparent advantage (0.9818) comes entirely from mid-gate access to exact simulator state, which destructive, microsecond-scale qubit readout cannot provide inside a 10 ns gate. Freezing the trained policy into a pulse a real controller could actually run doesn't just erase that advantage — it drops RL below the GRAPE baseline it appeared to beat.

| Controller | Nominal fidelity | Robust mean fidelity | Robust mean leakage |
|---|---|---|---|
| Naive constant pulse | 0.9649 | 0.9534 ± 0.0213 | 0.0199 |
| **GRAPE (nominal-only)** | 0.9895 | **0.9765 ± 0.0000** | 0.0050 |
| Ensemble-GRAPE (robust) | 0.9898 | 0.9769 ± 0.0000 | 0.0046 |
| RL (live/closed-loop) — *not deployable* | 0.9830 ± 0.0025 | 0.9818 ± 0.0025 | 0.0023 |
| RL (frozen/open-loop) — *deployable* | — | 0.9718 ± 0.0023 | 0.0022 |

"Deployable" means a real controller could run it without mid-gate feedback; "not deployable" means it needs information (exact, noiseless simulator state, every step) no real gate-time-scale readout can supply. Among deployable controllers, GRAPE wins.

(Both GRAPE variants' std is 0.0000 because all 5 optimization seeds converged to a bit-identical pulse — a well-behaved landscape, not a broken seed; see [SCIENCE.md, Section 7.2](SCIENCE.md).)

### The mechanism test: why RL's live number doesn't count

RL (live) beats both GRAPE baselines' robust-mean fidelity, but it also has something neither GRAPE variant has: exact, noiseless mid-episode state, which no real qubit readout could provide inside a 10 ns gate. **RL (frozen)** — the same trained policy, rolled out once on the nominal qubit and replayed as a fixed, open-loop pulse a real controller could actually run — settles whether RL (live)'s edge is a genuinely more robust waveform or exploitation of information that isn't physically available:

**RL (frozen) scores 0.9718 ± 0.0023 — significantly *worse* than nominal GRAPE (0.9765, paired t-test p=0.0167, 0 of 5 seeds ahead), let alone ensemble GRAPE (0.9769). Freezing the policy doesn't just erase its +0.0053 edge over GRAPE — it turns a win into a loss.**

This is not a null result about RL — it's a precise, signed answer: under the constraint that a real gate has to run open-loop, gradient-based optimal control wins this benchmark. What the result *does* support: fast, low-latency mid-gate state estimation (an active hardware research direction, not available for single-shot gate-time readout today) could in principle make closed-loop control competitive with, or better than, open-loop optimal control. It is not evidence that today's fixed-pulse calibration is already beatable by switching to RL.

### Leakage: the one advantage that survives freezing

Leakage escapes the computational subspace entirely, which is what actually breaks quantum error correction, in a way ordinary (recoverable) infidelity does not — and here the story is different from fidelity's. RL (frozen) leaks **0.0022** per gate, statistically indistinguishable from RL (live)'s 0.0023 (paired t-test p=0.35, only 3 of 5 seeds even favor live) — and both stay roughly **2x** lower than either GRAPE variant (0.0050 / 0.0046, paired t-test p=0.0003 for frozen vs. nominal GRAPE, consistent in all 5 seeds). Unlike the fidelity advantage, this one does not depend on live state access — it survives freezing into a deployable pulse. It's the one part of RL's apparent edge that isn't an artifact of unrealistic information: this training procedure produces a genuinely less-leaky waveform than either GRAPE baseline, even once it can no longer react.

### Does ensemble GRAPE close the gap to RL (live)?

Barely — and it doesn't need to, since nominal GRAPE already beats the only RL variant that's deployable. Ensemble GRAPE (0.9769) beats nominal GRAPE (0.9765) by only +0.0004 — a rounding error next to RL (live)'s +0.0049 to +0.0053 edge over both (paired t-test p<0.02, 5/5 seeds each, sign-test p=0.0312). Training on the right miscalibration distribution recovers almost none of RL (live)'s advantage over GRAPE; per the mechanism test above, the part of that advantage that matters — since RL (live) itself can't be deployed — comes overwhelmingly from live state access, not from distributional knowledge a fair open-loop baseline could also have used.

### Does the learned policy rediscover DRAG?

No — and the null result is itself validated, not asserted. The detection method (correlating learned Ωy(t) against the textbook DRAG correction −Ωx'(t)/α, `grape_baseline.py: drag_correlation_check`) was checked first against a synthetic pulse built from an *exact* analytic DRAG pair: it recovers a perfect correlation = 1.0000, R² = 1.0000 (`drag_check.py`, `results/drag_check.txt`), so a null on the real policy isn't a broken detector.

Run across all 5 `seed_study.py` policies: correlation is weak and inconsistent in sign (mean +0.11 ± 0.29, per-seed range −0.16 to +0.64), but R² is strongly negative in **every single seed** (mean −7.43, range −11.17 to −1.63) — the derivative-DRAG prediction is a far worse fit than just the mean, consistently. The policies instead learn a magnitude/phase-modulated drive: |Ωx+iΩy| decays from ~0.4 to ~0.15 rad/ns over the pulse in all 5 seeds, at a phase that's roughly constant *within* a seed but wildly different *across* seeds (spanning nearly the full circle) — a genuinely different leakage-suppression strategy from DRAG, not an imprecise version of it. See [SCIENCE.md, Section 7.5](SCIENCE.md) for the full per-seed breakdown.

---

## Technical Stack

| Library | Role |
|---------|------|
| [QuTiP](https://qutip.org) | Quantum simulation (Lindblad master equation) |
| [Gymnasium](https://gymnasium.farama.org) | RL environment interface |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io) | PPO implementation |
| [SciPy](https://scipy.org) | GRAPE baselines (L-BFGS-B) + paired statistics (t-test, sign test) |
| [NumPy](https://numpy.org) | Numerical computation |
| [Matplotlib](https://matplotlib.org) | Visualization (Bloch sphere, pulse, population plots) |

---

## References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Johansson et al. (2012). *QuTiP: An open-source Python framework for the dynamics of open quantum systems.* CPC.
4. Krantz et al. (2019). *A quantum engineer's guide to superconducting qubits.* Applied Physics Reviews, 6(2).
5. Niu et al. (2019). *Universal quantum control through deep reinforcement learning.* npj Quantum Information.
6. Fösel et al. (2018). *Reinforcement learning with neural networks for quantum feedback.* Physical Review X.
7. Khaneja et al. (2005). *Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms.* Journal of Magnetic Resonance. (GRAPE)
8. Motzoi et al. (2009). *Simple pulses for elimination of leakage in weakly nonlinear qubits.* Physical Review Letters. (DRAG)

---

## License

MIT License. See `LICENSE` for details.

---

*Built as part of a Quantum Engineering portfolio project.*
