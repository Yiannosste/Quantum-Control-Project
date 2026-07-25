# The Science Behind This Project

Full derivations and the complete experimental record behind [README.md](README.md)'s headline result: **on this benchmark, GRAPE beats every physically deployable RL policy** (§7). Every equation here is cross-checked against the actual code (`utils.py`, `quantum_env.py`); every number in §7 is read verbatim from `results/seed_study.txt` / `.json`, produced by `seed_study.py`.

---

## 1. Physics: 3-Level Transmon

A transmon is a weakly anharmonic oscillator, not a true two-level system — its third level $|2\rangle$ is reachable by the same drive that rotates $|0\rangle\leftrightarrow|1\rangle$. This project models all three levels as a $3\times3$ density matrix $\rho$ (`utils.py: build_hamiltonians`), using the ladder operator $a=$ `qutip.destroy(3)` and number operator $n=a^\dagger a$:

$$H(t) = \underbrace{-\Delta n + \frac{\alpha}{2}n(n-1)}_{H_0 \text{ (drift)}} + \underbrace{\Omega_x(t)\cdot\frac{a+a^\dagger}{2}}_{H_x \text{ (I)}} + \underbrace{\Omega_y(t)\cdot\frac{i(a^\dagger-a)}{2}}_{H_y \text{ (Q)}}$$

$\Delta$ is the drive detuning (0 nominal); $\alpha=-1.5708$ rad/ns ($-2\pi\times250$ MHz) is the anharmonicity — zero at $n=0,1$, exactly $\alpha$ at $n=2$, which is why a resonant 0-1 drive only *weakly* leaks into $|2\rangle$ instead of driving it resonantly. Truncating to 2 levels recovers the ordinary qubit model exactly ($a\to\sigma_-$, $H_x\to\sigma_x/2$).

**Bloch convention:** $|0\rangle,|1\rangle$ are the $+1,-1$ eigenstates of $\sigma_z$ (verified: `qt.expect(diag([1,-1,0]), ...)` gives exactly $\pm1$). $|2\rangle$ has no place on the sphere — tracked separately as leakage population $P_2$.

**Rabi check.** A resonant constant drive for duration $t$ rotates the Bloch vector by $\theta=\Omega_x t$; a $\pi$-pulse needs $\Omega_x=\pi/t_\pi$. For the default 10 ns gate this predicts $\Omega_x\approx0.3142$. Driving the environment at exactly that value with decoherence off and leakage disabled (2-level truncation) reproduces fidelity $=0.9999999$ (theory: 1.0) — confirms the rotation physics.

**Leakage.** $a$ connects every adjacent level pair, not just 0-1: $\langle1|a|2\rangle=\sqrt2$. Any pulse strong enough to rotate $|0\rangle\to|1\rangle$ quickly also drives $|1\rangle\to|2\rangle$, off-resonantly by $\alpha$ (small but nonzero, growing with drive strength — quantified in §3). **DRAG** (Motzoi et al. 2009) solves this in real hardware with a second quadrature drive $\approx-\dot\Omega_x(t)/\alpha$; this project gives the RL agent the same $\Omega_y$ channel but never hands it the formula — §7.5 checks what it discovers instead.

---

## 2. Decoherence: the Lindblad Equation, and a Real Bug

$$\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

| Channel | Operator (`utils.py: build_collapse_operators`) | Rate |
|---|---|---|
| $T_1$ relaxation | $L_1=\sqrt{\Gamma_1}\cdot a$ | $\Gamma_1=1/T_1$ |
| $T_2^*$ dephasing | $L_2=\sqrt{2\Gamma_\phi}\cdot n$ | $\Gamma_\phi=\frac{1}{T_2}-\frac{\Gamma_1}{2}$ |

Default `QUBIT_PARAMS`: $T_1=1000$ ns, $T_2=500$ ns, $\Omega_{\max}=0.35$ rad/ns — an aggressive noise regime chosen to make decoherence, leakage, and miscalibration all matter within a 10 ns gate, not to represent state-of-the-art hardware.

$L_1$'s $\sqrt{n}$ matrix elements give $\Gamma(2\to1)=2\Gamma_1$ automatically (harmonic-oscillator scaling). $\Gamma_\phi\ge0$ requires $T_2\le2T_1$ (satisfied: $500\le2000$).

**Deriving the $L_2$ prefactor.** For a diagonal Hermitian $L$ with eigenvalues $l_k$, the GKSL dissipator decays coherence $\rho_{jk}$ at rate $\frac{1}{2}(l_j-l_k)^2$. The naive qubit operator $\sqrt{\Gamma_\phi}\sigma_z/2$ gives eigenvalue splitting $\sqrt{\Gamma_\phi}$, hence rate $\Gamma_\phi/2$ — half the target. The correct 2-level form, $\sqrt{2\Gamma_\phi}\sigma_z/2$, gives exactly $\Gamma_\phi$. Generalizing to 3 levels: $n$ has eigenvalues $0,1,2,\dots$, spaced by exactly 1 — the *same* spacing as $\sigma_z/2$ (eigenvalues $\pm\frac{1}{2}$) — so $n$ is already the direct multi-level analog and needs **no** extra $\frac{1}{2}$ the way $\sigma_z$ itself (spacing 2) would.

**The bug this project shipped for a time:** $L_2=\sqrt{2\Gamma_\phi}\cdot n/2$ — carrying a $\sigma_z/2$-style `/2` onto an operator that didn't need one. Verified by direct simulation (equal superposition, no drive, coherence-decay fit):

```
Nominal T2 input                        : 500.0 ns
Measured T2 (sqrt(2*Gamma_phi)*n/2 form): 1142.9 ns   <- wrong by 2.29x (too little dephasing)
Measured T2 (sqrt(2*Gamma_phi)*n form)  :  500.0 ns   <- correct, matches to 4 sig. figs
```

Every fidelity number this project ever produced was generated under 2.29× too little dephasing until this was fixed. It was caught by testing the actual `build_collapse_operators()` output directly (`tutorial.ipynb`, §2) rather than a hand-verified stand-in, and is now guarded by `tests/test_physics.py` — a regression test asserting measured T1/T2 match `QUBIT_PARAMS` to 1%, so it cannot recur silently.

**T1 check:** starting in $|1\rangle$, $\langle\sigma_z\rangle(t)=1-2e^{-t/T_1}$ predicts $\langle\sigma_z\rangle(T_1)\approx+0.264$; simulation gives $\approx0.267$.

---

## 3. Gate Fidelity, and Why 10 ns

$$F(\rho,|\psi_{\text{target}}\rangle) = \langle\psi_{\text{target}}|\rho|\psi_{\text{target}}\rangle \in [0,1]$$

This is **single-state-transfer fidelity** ($|0\rangle\to|1\rangle$ only, one fixed input per episode) — not the standard **average gate fidelity** used in randomized benchmarking, which averages over a full/Haar-random input basis and would also verify $|1\rangle\to|0\rangle$ and superposition transfers. Treat it as a necessary check, not full gate-characterization.

**Decoherence vs. leakage trade off.** A slower gate accumulates more $T_1/T_2$ decay; a faster gate needs a stronger drive, which leaks more. Driving the naive, I-only, exactly-resonant $\pi$-pulse (leakage on, correct $T_1/T_2$, no bandwidth limit or miscalibration, to isolate this one effect) across gate durations:

| Gate time | $\Omega_x$ | Fidelity | Leakage $P_2$ |
|---|---|---|---|
| 50 ns | 0.0628 | 0.9622 | 0.0008 |
| 30 ns | 0.1047 | 0.9744 | 0.0023 |
| **20 ns** | **0.1571** | **0.9777** | **0.0049** |
| 15 ns | 0.2094 | 0.9746 | 0.0099 |
| 12 ns | 0.2618 | 0.9713 | 0.0130 |
| 10 ns | 0.3142 | 0.9617 | 0.0213 |
| 8 ns | 0.3927 | 0.9527 | 0.0267 |
| 5 ns | 0.6283 | 0.9038 | 0.0582 |

Fidelity **peaks at 20 ns and drops for shorter gates** — leakage overtakes decoherence once the drive is fast enough. 10 ns is only justified because the agent has a real leakage-suppression channel to use: a gradient-optimized 10 ns pulse using both quadratures (nominal GRAPE) reaches fidelity 0.9867, leakage 0.0049 — beating every entry above, including the 20 ns naive optimum (Krantz et al. 2019: 10-50 ns is typical for real single-qubit gates anyway).

---

## 4. MDP Formulation

| Component | Definition |
|---|---|
| State $s_t$ | 8-D generalized Bloch vector $\in[-1,1]^8$: $[P_1,P_2,\text{Re }\rho_{01},\text{Im }\rho_{01},\text{Re }\rho_{12},\text{Im }\rho_{12},\text{Re }\rho_{02},\text{Im }\rho_{02}]$ |
| Action $a_t$ | $(a_x,a_y)\in[-1,1]^2 \to \Omega_{x,y}=a_{x,y}\cdot\Omega_{\max}$ |
| Reward $r_t$ | $-\lambda(\Omega_x^2+\Omega_y^2) - \mu P_2$ every step, $+F(\rho_T,|1\rangle)$ at the end ($\lambda=0.01$, $\mu=1.0$) |
| Episode | $N=10$ steps $\times$ $dt=1$ ns $=$ 10 ns gate |

A $3\times3$ density matrix has $3^2-1=8$ real degrees of freedom, so $s_t$ is a *sufficient* Markovian state (the Lindblad equation is itself Markovian; phonon-bath correlation time ~ps ≪ 10 ns gate time). Commands pass through a first-order RC bandwidth filter ($\tau_{\text{AWG}}=0.5$ ns) before reaching the Hamiltonian — no instantaneous jumps.

**Per-episode miscalibration:** $\Delta\sim\mathcal N(0,\sigma_\Delta^2)$, $g\sim1+\mathcal N(0,\sigma_g^2)$ ($\sigma_\Delta=0.03$ rad/ns, $\sigma_g=0.05$) sampled once per episode, added to detuning / multiplied into amplitude — **not included in $s_t$**. A policy can only infer them from how $s_t$ evolves.

**The one deliberate unrealism, and why it matters:** $s_t$ is the *exact, noiseless* simulator density matrix — real dispersive readout is destructive and takes hundreds of ns to µs, far longer than this 10 ns gate. Standard practice for simulation-based RL, not a hardware claim — but it means the agent's state access is not something a real gate-time controller could have. §7.3 measures exactly how much of RL's performance depends on it.

---

## 5. Five Controllers, One Physics, One Reward

A single GRAPE baseline can't say anything reliable about *why* RL wins: (a) GRAPE optimized only for the nominal qubit isn't competing on RL's problem, and (b) RL's exact mid-episode state access (§4) is something no open-loop method could use even in principle — a win against *any* open-loop baseline conflates "better waveform" with "exploited unrealizable information." Five controllers separate those questions:

1. **Naive constant pulse** — analytic, leakage-blind, exactly-resonant I-only $\pi$-pulse. The floor.
2. **GRAPE (nominal-only)** — 20-parameter piecewise-constant $(\Omega_x,\Omega_y)$, L-BFGS-B, multi-start (random restarts over the full $[-1,1]$ range) against the *nominal* qubit only. Never sees the miscalibration distribution.
3. **Ensemble (robust) GRAPE** — same parameterization, optimized for *average* reward over a deterministic $3\times3$ quadrature grid over $(\Delta,g)$ at $\{-1,0,+1\}\sigma$ — a sample-average approximation of the expectation, still open-loop. The fair baseline for question (a).
4. **RL (live/closed-loop)** — PPO, fresh miscalibration every episode, observing $s_t$ every step.
5. **RL (frozen/open-loop)** — controller 4's trained policy, rolled out *once* on the nominal qubit, waveform frozen and replayed open-loop like GRAPE. If 4's edge is a better waveform, freezing barely changes it; if it's live reaction, freezing erases it. Answers question (b).

All five: identical reward, identical physics, identical 150 paired evaluation draws, across 5 independent seeds. RL uses **PPO** (Schulman et al. 2017) for its clipped-update training stability — useful here because the reward is sparse (fidelity bonus only at the last of 10 steps) and the physics is stiff (leakage and decoherence both punish aggressive exploration).

---

## 6. Known Simplifications

- **Fixed anharmonicity** — no charge dispersion beyond the single constant $\alpha$.
- **Truncated at 3 levels** — leakage to a hypothetical $|3\rangle$ isn't modeled; §3's naive-pulse fidelity *dropping* at fast gate times hints a real device could see effects beyond this truncation.
- **Static per-episode miscalibration** — resampled independently each episode, not correlated in time like real 1/f flux noise (a harder version of the same robustness question).
- **RWA** holds at these drive strengths but isn't checked automatically for arbitrary `omega_max`.
- **GRAPE's local optima** — a bounded number of L-BFGS-B restarts; neither GRAPE variant is necessarily the best pulse achievable by gradient methods.
- **200,000 PPO timesteps/seed** throughout — a different budget wasn't tested and could shift §7's numbers.

---

## 7. Results

### 7.1 Procedure

RL: 5 independent training seeds (200k timesteps each). GRAPE / ensemble-GRAPE: 5 independent random-restart seeds each. **Paired evaluation:** one fixed set of 150 $(\Delta,g)$ draws (`sample_draws`, seed 999), used for every controller and every seed, removing evaluation-noise confounds. **Statistic:** "robust mean/std" is hierarchical — each seed's mean over its 150 draws, then mean/std of those five per-seed means across seeds — the statistic that actually bounds whether a gap is real or seed noise. Paired t-tests and sign tests reported alongside.

### 7.2 Headline table

| Controller | Nominal fidelity | Robust mean fidelity | Robust mean leakage |
|---|---|---|---|
| Naive constant pulse | 0.9619 | 0.9505 ± 0.0211 | 0.0200 |
| GRAPE (nominal-only) | 0.9867 | 0.9738 ± 0.0000 | 0.0049 |
| Ensemble-GRAPE (robust) | 0.9869 | 0.9741 ± 0.0000 | 0.0046 |
| **RL (live/closed-loop)** | 0.9803 ± 0.0022 | **0.9788 ± 0.0020** | **0.0024** |
| RL (frozen/open-loop) | — | 0.9692 ± 0.0021 | 0.0024 |

Both GRAPE variants' std is 0.0000 because all 5 optimization seeds converged to a bit-identical pulse — `restart_seed` genuinely varies the random-restart inits, but the single physics-informed seeded init (the naive $\pi$-pulse) already sits at, or L-BFGS-B carries it to, the optimum every random restart failed to beat. Evidence the reported numbers are near the true optimum for this pulse class, not an artifact.

Paired significance (per-seed robust-mean fidelity, n=5):

| Comparison | Mean difference | Paired t-test | Sign test |
|---|---|---|---|
| RL (live) vs. GRAPE | +0.0050 | p=0.0071 | 5/5, p=0.0312 |
| RL (live) vs. Ensemble-GRAPE | +0.0047 | p=0.0084 | 5/5, p=0.0312 |
| RL (live) vs. RL (frozen) | +0.0096 | p=7.12e-07 | 5/5, p=0.0312 |
| RL (frozen) vs. GRAPE | -0.0047 | p=0.0117 | 0/5, p=1.0000 |

Paired significance (per-seed mean leakage, n=5):

| Comparison | Mean difference | Paired t-test | Sign test |
|---|---|---|---|
| RL (live) vs. RL (frozen) | +0.00002 | p=0.0231 | 4/5, p=0.1875 |
| RL (frozen) vs. GRAPE | -0.00249 | p=4.42e-06 | 0/5, p=1.0000 |

(Leakage "wins" counts *higher* leakage — read "0/5" for frozen-vs-GRAPE as frozen never leaking more, i.e. leaking less every seed. For live-vs-frozen, the ±0.00002 mean difference — not the small p-value — is what matters; the significance just reflects how tightly clustered that near-zero difference is.)

### 7.3 The mechanism test

RL (live) beats both GRAPE variants but also has exact mid-episode state access no real readout can provide — so the gap could be a genuinely more robust *waveform*, or *exploited live information*. RL (frozen) — each seed's policy, rolled out once on the nominal qubit, waveform frozen, replayed open-loop on the same 150 draws — settles it:

- Robust-mean fidelity drops from 0.9788 ± 0.0020 (live) to 0.9692 ± 0.0021 (frozen) — consistent across all 5 seeds (per-seed drop 0.0093-0.0103, p=7.12e-07).
- That drop *inverts* the GRAPE comparison: 0.9692 is significantly worse than nominal GRAPE's 0.9738 (p=0.0117, 0/5 seeds ahead) and worse than ensemble GRAPE's 0.9741 by nearly the same margin.

**Reading:** exploited information dominates. Restricted to what a real controller could run open-loop, GRAPE wins. Not a null result about RL — a precise, signed one: it supports that fast, low-latency mid-gate state estimation (an active hardware research direction, unavailable today) could make closed-loop control competitive with open-loop optimal control, not that today's fixed-pulse calibration is already beatable by switching to RL. The exception is leakage (§7.2): that advantage is overwhelmingly a property of the waveform itself and survives freezing nearly intact.

### 7.4 Does ensemble GRAPE close the gap to RL (live)?

Barely, and it doesn't need to — nominal GRAPE already beats the only deployable RL variant. Ensemble GRAPE (0.9741) beats nominal GRAPE (0.9738) by only +0.0002 — an order of magnitude below RL (live)'s edge over either (+0.0047 to +0.0050). Training on the correct miscalibration distribution recovers almost none of RL's advantage; per §7.3, nearly all of it depends on live state access, not distributional knowledge a fair open-loop baseline could also use. Ensemble GRAPE does modestly reduce leakage vs. nominal (0.0046 vs. 0.0049), but both remain ~2× RL (live)'s (0.0024).

### 7.5 Does the learned policy rediscover DRAG?

Checked whether $\Omega_y(t)\approx-\dot\Omega_x(t)/\alpha$ (correlation + best-fit-scale $R^2$) on each seed's frozen pulse (`grape_baseline.py: drag_correlation_check`). Validated the detector first against a synthetic Gaussian pulse paired with its *exact* analytic DRAG $\Omega_y(t)$ (`validate_drag_detector`, run via `drag_check.py`): recovers correlation = $R^2$ = 1.0000 — confirms a null result below isn't a broken detector (`results/drag_check.txt`).

| Seed | Correlation | $R^2$ |
|---|---|---|
| 0 | +0.02 | -9.47 |
| 1 | +0.83 | -0.89 |
| 2 | -0.32 | -12.18 |
| 3 | +0.86 | **+0.73** |
| 4 | +0.58 | -10.77 |
| **Mean ± std** | **+0.40 ± 0.47** | **-6.51 ± 5.35** |

**Mostly no, with one real exception.** 4 of 5 seeds show strongly negative $R^2$ — derivative-DRAG is a far worse predictor than the mean. Seed 3 breaks the pattern: correlation +0.86 *and* $R^2=+0.73$, a genuinely good fit. (Correlation alone is weaker evidence — seeds 1 and 4 also correlate well but still fit poorly, since correlation ignores a constant offset that the no-intercept $R^2$ fit does not; seed 3 is the only one where both agree.) The other four learn a magnitude/phase-modulated drive instead: $|\Omega_x+i\Omega_y|$ decays from $\approx0.39$-$0.49$ to $\approx0.15$-$0.17$ rad/ns over the pulse in all 5 seeds, at a phase roughly constant *within* a seed but spanning nearly the full circle *across* seeds ($\approx-70°,+8°,+100°,-158°,+112°$ for seeds 0-4) — consistent with an arbitrary, seed-dependent I/Q reference choice rather than a rediscovered physical relationship, except that one of five independent runs found something close to the closed-form correction anyway.

---

## References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Johansson et al. (2012). *QuTiP: An open-source Python framework for the dynamics of open quantum systems.* Computer Physics Communications.
4. Krantz et al. (2019). *A quantum engineer's guide to superconducting qubits.* Applied Physics Reviews, 6(2).
5. Niu et al. (2019). *Universal quantum control through deep reinforcement learning.* npj Quantum Information.
6. Fösel et al. (2018). *Reinforcement learning with neural networks for quantum feedback.* Physical Review X.
7. Khaneja et al. (2005). *Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms.* Journal of Magnetic Resonance. (GRAPE)
8. Motzoi et al. (2009). *Simple pulses for elimination of leakage in weakly nonlinear qubits.* Physical Review Letters. (DRAG)
