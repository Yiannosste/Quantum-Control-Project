# The Science Behind This Project

This document derives, in detail, the physics and reinforcement-learning
theory implemented in this repository, and reports the full experimental
comparison between reinforcement learning and gradient-based optimal
control. Every equation is cross-checked against the actual code
(`utils.py`, `quantum_env.py`) and, where practical, against a direct
numerical run of that code — not just against the textbook result in
isolation. Every number in Section 7 is read verbatim from
`results/seed_study.txt` / `.json`, produced by `seed_study.py`. File
references point to the functions that implement each piece of physics.

The result Section 7 arrives at — GRAPE beats every physically deployable
RL policy — is not the interesting part on its own; a single number either
way would prove little. What makes it trustworthy is everything built
around it: two genuinely fair GRAPE baselines instead of a strawman
(Section 5), a designed experiment that isolates *why* the gap exists
rather than just reporting that it does (Section 7.3), five independent
seeds with paired significance tests instead of a single lucky run
(Section 7.2), a detector validated against a known ground truth before
its negative result was trusted (Section 7.5), and a real
collapse-operator bug — simulating 2.29x too little dephasing — caught by
testing the actual production code path instead of a hand-verified
stand-in, and now guarded by a standing regression test (Section 2.1,
`tests/test_physics.py`). That methodology is the actual subject of this
document.

---

## 1. The Transmon as a Weakly-Anharmonic 3-Level System

A transmon superconducting qubit is not, strictly, a two-level system — it
is a weakly anharmonic oscillator, and its third level $|2\rangle$ is close
enough in energy to be reachable by the same drive that rotates
$|0\rangle\leftrightarrow|1\rangle$. This repository models all three levels
($|0\rangle$, $|1\rangle$, $|2\rangle$) as a $3\times3$ density matrix
$\rho$, truncating the transmon's full ladder at the first level that
matters for gate errors — Section 1.4 explains why a strict two-level model
would miss a real, physically important error channel, and Section 3
quantifies its size.

### 1.1 Bloch sphere convention (0-1 subspace)

The 0-1 subspace can still be visualized on a **Bloch sphere**, exactly as
in the two-level case, by restricting to the embedded Pauli operators (e.g.
$\sigma_z = \lvert0\rangle\langle0\rvert - \lvert1\rangle\langle1\rvert$,
padded with a zero in the $|2\rangle$ row/column):

$$\rho_{01\text{-subspace}} \propto I + x\sigma_x + y\sigma_y + z\sigma_z, \qquad (x,y,z) = (\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)$$

This repository follows the standard convention where $|0\rangle$ and
$|1\rangle$ are the $+1$ and $-1$ eigenstates of $\sigma_z$, respectively.
Concretely, in code (`quantum_env.py`, `reset()`) the qubit starts in
`qt.ket2dm(qt.basis(3, 0))`, and the target state (`target_state`) is
`qt.basis(3, 1)`. Verified directly:

```
sz3 = qt.Qobj(np.diag([1, -1, 0]))                    # sigma_z embedded in the 3-level space
qt.expect(sz3, qt.ket2dm(qt.basis(3, 0)))  ==  +1.0   # |0>, ground
qt.expect(sz3, qt.ket2dm(qt.basis(3, 1)))  ==  -1.0   # |1>, excited (target)
```

So on the Bloch sphere used throughout this project:

| Pole | $\langle\sigma_z\rangle$ | State |
|------|------|-------|
| North | $+1$ | $\lvert 0\rangle$ (ground, episode start) |
| South | $-1$ | $\lvert 1\rangle$ (excited, episode target) |

An **X-gate** is a $\pi$ rotation about the X-axis that swaps the poles:
$\lvert 0\rangle \to \lvert 1\rangle$. $\lvert2\rangle$ has no place on this
sphere at all — any population that reaches it is tracked separately as
**leakage** ($P_2$), not as a point on the Bloch sphere (Section 1.4).

### 1.2 System Hamiltonian

In the rotating-wave approximation (RWA), in a frame rotating at the drive
frequency, the Hamiltonian splits into a drift term and two control terms
(`utils.py: build_hamiltonians`), using the ladder operator $a$ on the
3-level truncated Hilbert space ($a=$ `qutip.destroy(3)`) and the number
operator $n=a^\dagger a$:

$$H(t) = \underbrace{-\Delta n + \frac{\alpha}{2}n(n-1)}_{H_0 \text{ (drift)}} + \underbrace{\Omega_x(t)\cdot\frac{a+a^\dagger}{2}}_{H_x \text{ (I)}} + \underbrace{\Omega_y(t)\cdot\frac{i(a^\dagger-a)}{2}}_{H_y \text{ (Q)}}$$

- $\Delta$: qubit-drive detuning. The code sets `omega_q = 0.0`
  (`QUBIT_PARAMS`) as the *nominal* value, i.e. the drive is nominally
  resonant — but Section 4 adds a random per-episode $\Delta$ on top of
  this nominal value, unknown to the agent.
- $\alpha$: the anharmonicity, `QUBIT_PARAMS['alpha'] = -1.5708` rad/ns
  ($-2\pi\times250$ MHz, a typical transmon value). At $n=0,1$ the
  anharmonic term $\frac{\alpha}{2}n(n-1)$ vanishes identically; at $n=2$ it
  contributes exactly $\alpha$. This is what keeps a resonant 0-1 drive from
  also resonantly driving 1-2.
- $\Omega_x(t), \Omega_y(t)$: the two drive quadratures, both RL-agent
  actions (denormalized: $\Omega_{x,y}(t) = a_{x,y}(t)\cdot\Omega_{\max}$).
  Truncating to 2 levels recovers the ordinary two-level qubit model
  exactly: $a\to\sigma_-$, $H_x\to\sigma_x/2$, and $H_y$ (and leakage)
  become moot.

### 1.3 Resonant (Rabi) rotation and the π-pulse condition

Restricted to the 0-1 subspace (ignoring leakage for a moment), a constant
drive $\Omega_x$ applied for a duration $t$ at resonance rotates the Bloch
vector by angle $\theta=\Omega_x t$ about the X-axis. A full population
transfer $|0\rangle\to|1\rangle$ (an ideal, noise-free, leakage-free X-gate)
therefore requires

$$\Omega_x \cdot t_\pi = \pi \quad\Longrightarrow\quad \Omega_x = \frac{\pi}{t_\pi}$$

**Verified against the code.** For the default 10 ns gate
(`n_steps=10`, `dt=1.0`), this predicts $\Omega_x=\pi/10\approx0.3142$.
Driving `QuantumPulseEnv` with a constant I-only amplitude of exactly that
value, with decoherence turned off ($T_1=T_2=10^9$ ns) *and* leakage
suppressed by disabling the anharmonic coupling, reproduces the ideal
result:

```
fidelity = 0.9999999   (theory: 1.0, exactly the π-pulse condition)
```

This confirms the Rabi/rotation physics is implemented correctly. The moment
leakage and finite $T_1/T_2$ are switched back on (i.e. the actual default
environment), this same "obvious" constant pulse stops being optimal —
that's the subject of Sections 1.4 and 3.

### 1.4 Leakage, and why a second control channel exists

$a$ (and therefore $H_x$) connects *every* adjacent pair of levels, not just
0 and 1: $\langle1\rvert a\lvert2\rangle=\sqrt2$, a matrix element directly
connecting $|1\rangle$ and $|2\rangle$ (verified via
`qt.destroy(3) * qt.basis(3,2) == sqrt(2)*qt.basis(3,1)`). So any pulse strong
enough to rotate $|0\rangle\to|1\rangle$ quickly also drives population
toward $|2\rangle$ — off-resonantly, by the anharmonicity $\alpha$, which is
why the leakage is small rather than catastrophic, but it is not zero, and
it grows with drive strength (Section 3 quantifies this directly).

This is a completely standard, real problem in transmon control, and it has
a completely standard, real solution: **DRAG** (Derivative Removal by
Adiabatic Gate, Motzoi et al. 2009) adds a second, quadrature ("Q") drive
component approximately proportional to $-\dot\Omega_x(t)/\alpha$, which
analytically cancels the leading-order leakage. This repository gives the
RL agent that same second channel ($\Omega_y$, i.e. $H_y$ above) — but never
computes or hands it the DRAG formula. Whatever leakage suppression the
agent uses, it has to discover from the reward signal (which includes an
explicit leakage penalty, Section 4) alone. Section 7.5 checks directly
whether it converges on the textbook DRAG correction or something else.

---

## 2. Open Quantum Systems: The Lindblad Master Equation

Real qubits are not isolated — they exchange energy and phase information
with their electromagnetic environment. This is modeled with the Lindblad
(GKSL) master equation for the density matrix $\rho$:

$$\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

The first term is unitary evolution under $H(t)$; the sum models
dissipation through collapse (Lindblad/jump) operators $L_k$.

### 2.1 Collapse operators (`utils.py: build_collapse_operators`)

Two decoherence channels are modeled, generalized to the 3-level truncated
Hilbert space:

| Channel | Operator | Rate | Physical meaning |
|---|---|---|---|
| $T_1$ relaxation | $L_1=\sqrt{\Gamma_1}\cdot a$ | $\Gamma_1 = 1/T_1$ | Spontaneous decay $\lvert n\rangle\to\lvert n{-}1\rangle$ |
| $T_2^{\ast}$ dephasing | $L_2=\sqrt{2\Gamma_\phi}\cdot n$ | $\Gamma_\phi = \frac{1}{T_2}-\frac{\Gamma_1}{2}$ | Random phase drift, no population change |

where $a=$ `qutip.destroy(3)` and $n=a^\dagger a$. Because
$\langle n{-}1\rvert a\lvert n\rangle=\sqrt{n}$, the single operator $L_1$
gives $\Gamma(1\to0)=\Gamma_1$ **and** $\Gamma(2\to1)=2\Gamma_1$
automatically — the standard "harmonic-oscillator" scaling assumed for a
weakly anharmonic transmon absent a separately-measured $T_1$ for
$\lvert2\rangle$. Truncating to 2 levels recovers $L_1=\sqrt{\Gamma_1}\sigma_-$
exactly.

The rate relation $\Gamma_\phi = 1/T_2 - 1/(2T_1)$ comes from decomposing the
standard total dephasing rate $1/T_2 = 1/(2T_1) + 1/T_2^{\ast}$ (energy decay
also destroys phase coherence at half its rate) and identifying
$\Gamma_\phi = 1/T_2^{\ast}$. **Physical constraint:** this requires
$T_2 \le 2T_1$; the code enforces $\Gamma_\phi\ge0$ via a `max(..., 0.0)`
clamp. The default parameters $T_1=1000$ ns, $T_2=500$ ns satisfy this with
margin ($500 \le 2000$).

**Deriving the $L_2$ prefactor, and why there is no extra $1/2$ on $n$.**
For a diagonal Hermitian Lindblad operator $L$ with eigenvalues $l_k$ on
level $k$, expanding the GKSL dissipator
$D[L]\rho = L\rho L^\dagger - \frac{1}{2}\{L^\dagger L,\rho\}$ on the
coherence $\rho_{jk}$ gives a decay rate of $\frac{1}{2}(l_j-l_k)^2$.

For the ordinary 2-level qubit, the *naively-transplanted* operator
$L=\sqrt{\Gamma_\phi}\cdot\sigma_z/2$ has eigenvalues $\pm\sqrt{\Gamma_\phi}/2$
on $\lvert0\rangle,\lvert1\rangle$, giving a decay rate of $\Gamma_\phi/2$ —
half of the target $\Gamma_\phi$. The *correct* 2-level operator,
$L=\sqrt{2\Gamma_\phi}\cdot\sigma_z/2$, has eigenvalues $\pm\sqrt{2\Gamma_\phi}/2$,
giving exactly $\Gamma_\phi$.

Generalizing to 3 levels: $n$ has eigenvalues $0,1,2,\dots$ — levels spaced
by exactly $1$, the *same* spacing as $\sigma_z/2$ (eigenvalues $\pm\frac{1}{2}$).
So $n$ is already the direct multi-level analog of $\sigma_z/2$ and needs no
extra $\frac{1}{2}$ the way $\sigma_z$ itself (spacing $2$) would — the correct
3-level operator is $L_2=\sqrt{2\Gamma_\phi}\cdot n$, obtained by directly
substituting $n$ for $\sigma_z/2$ in the correct 2-level form above.

We verified this by direct simulation: starting from an equal superposition
$(\lvert0\rangle+\lvert1\rangle)/\sqrt2$ with no drive, propagating under
`utils.py: build_collapse_operators`'s actual output, and fitting the
exponential decay of the 0-1 coherence:

```
Nominal T2 input                       : 500.0 ns
Measured T2 (sqrt(2*Gamma_phi)*n/2 form): 1142.9 ns   <- wrong by 2.29x (too little dephasing)
Measured T2 (sqrt(2*Gamma_phi)*n form)  : 500.0 ns    <- matches to 4 significant figures
```

The `n/2` form was, for a time, what this codebase actually shipped: every
fidelity number this project has ever reported was generated under 2.29x
too little dephasing before this fix. (`tutorial.ipynb`, Section 2,
reproduces this exact measurement against the real `build_collapse_operators()`
function, not a hand-rolled stand-in — testing the actual function is what
catches this class of bug; `tests/test_physics.py` guards against it
recurring via a regression test asserting measured T1/T2 match
`QUBIT_PARAMS` to 1%.) Every fidelity number in the current version of this
document and the README uses the corrected, verified operator.

### 2.2 Numerical check: T1 decay

Starting from the excited state $\lvert1\rangle$ ($\langle\sigma_z\rangle=-1$)
with no drive, pure $T_1$ relaxation drives the excited-state population as
$p_1(t)=e^{-t/T_1}$, so

$$\langle\sigma_z\rangle(t) = 1 - 2p_1(t) = 1 - 2e^{-t/T_1}$$

At $t=T_1$: $\langle\sigma_z\rangle = 1-2/e \approx +0.264$. Simulating this
exact scenario with `qt.mesolve` reproduces $\langle\sigma_z\rangle(T_1)
\approx 0.267$ (matches to simulation tolerance; see `tutorial.ipynb`,
section 2).

---

## 3. Gate Fidelity

State fidelity against a pure target $\lvert\psi_{\text{target}}\rangle$
(`utils.py: compute_fidelity`):

$$F(\rho, \lvert\psi_{\text{target}}\rangle) = \langle\psi_{\text{target}}\rvert \rho \lvert\psi_{\text{target}}\rangle \in [0, 1]$$

For the X-gate, $\lvert\psi_{\text{target}}\rangle = \lvert1\rangle$, so $F$
is exactly the excited-state population at the end of the gate.

**What this is not.** This is the fidelity of one specific state transfer,
$\lvert0\rangle\to\lvert1\rangle$, starting from a single fixed input state
every episode. It is *not* the standard **average gate fidelity** used in
randomized benchmarking and most gate-characterization literature, which
averages state fidelity over a full basis or Haar-random set of input
states (equivalently, relates to the process/Choi fidelity via
$F_{\text{avg}} = (d F_{\text{process}}+1)/(d+1)$) and would also verify
that, e.g., $\lvert1\rangle\to\lvert0\rangle$ and superposition states
transform correctly. An X-gate that only reliably does
$\lvert0\rangle\to\lvert1\rangle$ is not yet a verified gate in the
formal sense. Throughout this document and the README, "fidelity" means
this single-input-state transfer fidelity unless stated otherwise; treat it
as a necessary check on the pulse, not a substitute for full
gate-characterization.

### 3.1 Two competing error sources: decoherence vs. leakage

A slower gate spends longer exposed to $T_1$/$T_2$ decay; a faster gate
needs a stronger drive, which leaks more into $\lvert2\rangle$ (Section
1.4). These two error sources pull in opposite directions, so there is no
free lunch from "just going faster." We measured this directly by driving
the full 3-level environment (real leakage, correctly-calibrated $T_1/T_2$,
no bandwidth limiting or miscalibration, to isolate this one effect) with a
**naive, I-only, constant, exactly-resonant $\pi$-pulse** — the best pulse
a two-level analysis alone would recommend — across a range of gate
durations:

| Gate time | Required $\Omega_x$ | Fidelity | Leakage $P_2$ |
|---|---|---|---|
| 50 ns | 0.0628 | 0.9622 | 0.0008 |
| 30 ns | 0.1047 | 0.9744 | 0.0023 |
| **20 ns** | **0.1571** | **0.9777** | **0.0049** |
| 15 ns | 0.2094 | 0.9746 | 0.0099 |
| 12 ns | 0.2618 | 0.9713 | 0.0130 |
| 10 ns | 0.3142 | 0.9617 | 0.0213 |
| 8 ns | 0.3927 | 0.9527 | 0.0267 |
| 5 ns | 0.6283 | 0.9038 | 0.0582 |

The naive-pulse fidelity **peaks around 20 ns and gets *worse* for shorter
gates**, the opposite of "shorter is always better." Leakage overtakes
decoherence as the dominant error source once the drive is fast enough. A
20 ns gate with no leakage suppression at all already beats a naive 10 ns
gate — shortening the gate only pays off once something suppresses the
leakage that comes with the stronger drive it requires.

### 3.2 Why 10 ns anyway: leakage-suppressed control beats the naive optimum

The point of giving the agent a second (Q) control channel (Section 1.4) is
that it doesn't have to accept the naive-pulse tradeoff above. Nominal
GRAPE — a genuinely leakage-suppressing 10 ns pulse, gradient-optimized
against the full 3-level physics (`grape_baseline.py`, see Section 7 for
the full procedure and results) — reaches:

```
fidelity = 0.9867   leakage = 0.0049
```

(read verbatim from `results/seed_study.txt`, identical to the GRAPE row in
Section 7.2) — comfortably beating *every* entry in the naive-pulse table
above, including the 20 ns "sweet spot," despite running at the more
aggressive 10 ns gate time. This is the justification for the 10 ns default: it is
only a good choice *because* the agent has a real, functioning
leakage-suppression channel to use, not because shorter is automatically
better. Section 7 reports what the fully-trained RL policy achieves at this
same setting, plus how it and two GRAPE baselines hold up once per-episode
miscalibration (Section 4) is switched on. A 10 ns single-qubit gate is
itself unremarkable by real transmon standards (Krantz et al. 2019 report
typical single-qubit gate times in the 10-50 ns range).

---

## 4. MDP Formulation

| Component | Definition | Code |
|---|---|---|
| State $s_t$ | Generalized Bloch vector, 8-D, $\in[-1,1]^8$ | `_get_obs()` |
| Action $a_t$ | $(a_x,a_y)\in[-1,1]^2 \to \Omega_x,\Omega_y=a_x,a_y\cdot\Omega_{\max}$ | `step()` |
| Reward $r_t$ | $-\lambda(\Omega_x^2+\Omega_y^2) - \mu P_2$ every step, $+F(\rho_T,\lvert1\rangle)$ at the final step | `step()` |
| Episode | $N=10$ steps $\times$ $dt=1$ ns $=$ 10 ns gate | `n_steps`, `dt` |

### 4.1 State: an 8-D generalized Bloch vector

A $3\times3$ Hermitian, trace-1 density matrix has $3^2-1=8$ real degrees
of freedom (the qubit case's 3-component Bloch vector is the $n=2$ instance
of this same counting, $2^2-1=3$). `_get_obs()` returns exactly 8 numbers,
read directly off $\rho$: the two independent populations $P_1,P_2$
($P_0=1-P_1-P_2$ is implicit) and the real/imaginary parts of the three
coherences:

$$s_t = [P_1, P_2, \mathrm{Re}(\rho_{01}), \mathrm{Im}(\rho_{01}), \mathrm{Re}(\rho_{12}), \mathrm{Im}(\rho_{12}), \mathrm{Re}(\rho_{02}), \mathrm{Im}(\rho_{02})]$$

This is Markovian for the same reason the original 3-component Bloch state
was: the Lindblad equation is itself a Markovian (memoryless)
approximation, valid when the environment's correlation time is much
shorter than the gate time — an excellent approximation for millikelvin
transmon qubits (phonon-bath correlation time ~ps vs. gate time ~10 ns).

**A deliberate design choice, with a consequence explored in Section 7.**
$s_t$ is read directly off the noiseless simulator density matrix — exact
populations and coherences, no measurement noise or backaction. A real
qubit cannot produce this mid-gate: standard dispersive readout is
destructive to the state it measures and takes hundreds of ns to several
$\mu$s, against a 10 ns gate here. Giving the training environment this
information is standard practice for a simulation-only RL study (it is not
available on hardware, and this project does not claim otherwise) — but it
means the state observation the agent trains with is not something a real
gate-time-scale controller could have. Section 7.4 tests directly how much
of the resulting policy's performance depends on that access, by freezing
a trained policy into a fixed, open-loop pulse and re-evaluating it.

### 4.2 Action: two independent drive quadratures

$$\Omega_x(t) = a_x(t)\cdot\Omega_{\max}, \qquad \Omega_y(t) = a_y(t)\cdot\Omega_{\max}$$

$\Omega_y$ is not decorative — it is the same physical degree of freedom
DRAG uses (Section 1.4). PPO's Gaussian policy head extends from 1-D to 2-D
continuous actions with no algorithmic changes.

### 4.3 Bandwidth-limited control (`quantum_env.py: _apply_bandwidth_filter`)

The commanded $(\Omega_x,\Omega_y)$ never reaches the Hamiltonian directly.
It passes through a discrete first-order RC low-pass filter with time
constant $\tau_{\text{AWG}}$ (`bandwidth_tau`, default 0.5 ns):

$$\Omega_{\text{applied}}[i] = \Omega_{\text{applied}}[i-1] + \left(1-e^{-dt/\tau_{\text{AWG}}}\right)\left(\Omega_{\text{commanded}}[i]-\Omega_{\text{applied}}[i-1]\right)$$

the standard exponential-smoothing/RC discretization for a control signal
sampled every $dt$. This models a real arbitrary-waveform generator's
finite rise time — an instantaneous 1 ns piecewise-constant jump is not
something real control electronics can produce. The filter state persists
across an episode and resets at `reset()`.

### 4.4 Per-episode miscalibration (`quantum_env.py: reset()`)

At the start of every episode, before any action is taken, two values are
sampled once and held fixed for that episode:

$$\Delta \sim \mathcal{N}(0,\sigma_\Delta^2), \qquad g \sim 1+\mathcal{N}(0,\sigma_g^2)$$

($\sigma_\Delta=$ `detuning_std` $=0.03$ rad/ns, $\sigma_g=$
`amp_error_std` $=0.05$, by default). $\Delta$ is added to the nominal
detuning in $H_0$ (Section 1.2); $g$ multiplies the bandwidth-filtered
amplitude before it reaches the Hamiltonian. **Neither is included in
$s_t$** — a policy can only extract information about them from how the
observed state evolves within the episode. This is the mechanism the full
study (Section 7) is designed to isolate: does a policy trained this way
learn a waveform that is robust *by construction*, or does it learn to
infer $\Delta,g$ from $s_t$ and compensate live? Section 7.4 answers this
directly.

### 4.5 Reward

$$r_t = -\lambda\left(\Omega_x(t)^2+\Omega_y(t)^2\right) - \mu P_2(t) + \big[t=T\big]F(\rho_T,\lvert1\rangle)$$

with $\lambda=0.01$ (`energy_penalty`) and $\mu=1.0$ (`leakage_penalty`) by
default. The leakage term is evaluated **every step**, not only at the end
— $P_2$ read directly off the simulator's density matrix mid-episode (the
same simulation-only caveat as Section 4.1 applies) gives the agent a much
more immediate training signal than waiting for the one sparse fidelity
readout at $t=T$, where leakage's contribution to infidelity is
indistinguishable from ordinary incomplete rotation.

---

## 5. Five Controllers, One Physics, One Reward

Classical pulse optimization — **GRAPE** (Khaneja et al. 2005) — computes
the fidelity gradient with respect to a piecewise-constant pulse and
optimizes it directly against a known system model. Comparing RL against a
single GRAPE baseline is not enough to say anything reliable about *why*
one wins, for two separable reasons: (a) GRAPE optimized only for the
nominal, uncalibrated qubit is not competing on the same problem RL trains
on, and (b) RL's access to exact mid-episode state (Section 4.1) is not
something any open-loop method could use even in principle, so a win
against *any* open-loop baseline conflates "found a better waveform" with
"exploited information no open-loop method could have." This project runs
five controllers, specifically to separate those two questions:

1. **Naive constant pulse** (Section 3.1) — the analytic, leakage-blind,
   exactly-resonant I-only $\pi$-pulse. The floor: no optimization, no
   distributional robustness, no leakage suppression.
2. **GRAPE (nominal-only)** — piecewise-constant $(\Omega_x,\Omega_y)$
   (20 free parameters) optimized with L-BFGS-B against the *nominal*,
   zero-miscalibration qubit, with a genuine multi-start search (random
   restarts over the full $[-1,1]$ range, not a narrow band around zero).
   Gets to assume a perfectly-known system — the one real advantage a
   fixed offline optimizer has — but never sees the miscalibration
   distribution it is later evaluated against.
3. **Ensemble (robust) GRAPE** — the same parameterization, optimized to
   maximize *average* reward over a deterministic $3\times3$ quadrature
   grid over $(\Delta, g)$ at $\{-1,0,+1\}$ standard deviations (a
   sample-average approximation of the expectation over the miscalibration
   distribution — the standard way a competent GRAPE practitioner would
   handle a known-miscalibrated device). Still open-loop, still no
   mid-episode information, but now trained on the same distribution RL
   is — the fair baseline for question (a).
4. **RL (live/closed-loop)** — PPO, trained across freshly-sampled
   miscalibration every episode, observing $s_t$ every step.
5. **RL (frozen/open-loop)** — the exact same trained policy from (4),
   but run *once* on the nominal qubit and its output waveform frozen into
   a fixed, open-loop pulse, then evaluated exactly like GRAPE. If (4)'s
   advantage is a genuinely more robust waveform, freezing it should barely
   change its robustness. If (4)'s advantage instead comes from reacting to
   $\Delta,g$ live, freezing it should erase that advantage. This directly
   answers question (b).

All five are optimized/trained and evaluated against the *identical* reward
(energy penalty, leakage penalty, terminal fidelity bonus) and physics
(bandwidth filter, 3-level Lindblad dynamics), on an identical set of
paired evaluation draws (Section 7.1), across 5 independent seeds. **RL
(live)** uses **Proximal Policy Optimization (PPO)** (Schulman et al.,
2017) for its training stability: it clips the policy-gradient update to
prevent destructively large steps, which matters when the reward signal is
sparse (the fidelity bonus only arrives at the last step of each 10-step
episode) and the physics is stiff (leakage and decoherence both punish
overly aggressive exploration):

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\Big(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\Big)\right]$$

This is a training-stability statement about PPO in general, not a claim
re-derived from this project's specific results.

---

## 6. Known Simplifications / Limitations

Being transparent about what this model does *not* capture:

- **Fixed anharmonicity, no charge dispersion.** $\alpha$ is a single
  constant (`QUBIT_PARAMS['alpha']`); a real transmon's level spacing has
  additional structure (charge dispersion) this truncated-oscillator model
  doesn't include.
- **Truncated at 3 levels.** Population that would, on a real device, leak
  further to $\lvert3\rangle$ is not represented — this matters more as
  drive strength grows, and isn't bounded by anything in this model.
  Section 3.1's naive-pulse fidelity actually *dropping* at very fast gate
  times is a signal that a real device could see leakage effects beyond
  what this 3-level truncation can show.
- **Static, single-episode miscalibration.** Section 4.4's detuning and
  amplitude error are constant *within* an episode and independently
  resampled every episode — real qubit drift is correlated in time (e.g.
  1/f flux noise), which is a harder and more realistic version of the same
  robustness question.
- **RWA validity at high drive.** The rotating-wave approximation
  underlying $H(t)$ (Section 1.2) degrades as $\Omega$ becomes a
  non-negligible fraction of the qubit frequency; this project's drive
  strengths are well within where RWA holds, but this isn't checked
  automatically for arbitrary `omega_max` values a user might pass.
- **GRAPE's local optima.** `grape_baseline.py` runs a bounded number of
  L-BFGS-B restarts (Section 7); wider or more numerous restarts, or a
  different local optimizer, could in principle find a better pulse for
  either GRAPE variant. Neither GRAPE result here is necessarily *the* best
  pulse achievable by gradient methods.
- **Simulation-unit labels.** Code comments label `dt` as nanoseconds and
  `omega_max` as "rad/ns", and cross-checked against realistic transmon
  Rabi rates (Section 3) — but no unit-conversion layer enforces this
  anywhere in the code; a user changing `QUBIT_PARAMS` is responsible for
  keeping the units self-consistent.
- **Training budget not swept.** RL uses 200,000 PPO timesteps per seed
  throughout; a different budget was not tested and could shift the
  numbers in Section 7.

---

## 7. Results: GRAPE Beats Every Deployable RL Policy, and Why

**Headline: on this benchmark, gradient-based open-loop optimal control
(GRAPE, 0.9738 robust-mean fidelity) beats every physically deployable
version of the RL policy (RL frozen/open-loop, 0.9692).** The closed-loop RL
number that looks better (0.9788) depends on mid-episode access to exact,
noiseless simulator state — information no real qubit readout can provide
inside a 10 ns gate (Section 4.1). Freezing the trained policy into a pulse
a real controller could actually run does not merely erase its apparent
advantage over GRAPE; it inverts it. The exception is leakage: RL (frozen)
still leaks far less than either GRAPE variant, and by nearly the same
margin as RL (live) — the small residual gap between live and frozen
leakage is statistically detectable but practically negligible (Section
7.2 has the precise numbers).

### 7.1 Procedure

All five controllers (Section 5) are optimized/trained and evaluated under
identical conditions:

- **Seeds.** RL is trained across 5 independent seeds (200,000 timesteps
  each). GRAPE and ensemble GRAPE are each optimized across 5 independent
  random-restart seeds.
- **Paired evaluation.** A single fixed set of 150 paired $(\Delta, g)$
  miscalibration draws is sampled once (`grape_baseline.py: sample_draws`,
  master seed 999) and used to evaluate **every controller, every seed** —
  removing evaluation-noise confounds between controllers, so a reported
  difference reflects the controllers, not which random draws happened to
  favor one of them.
- **Statistic.** Reported "robust mean/std" is the hierarchical statistic
  that actually bounds whether a gap is real or seed noise: for each seed,
  compute that seed's mean fidelity over the 150 shared draws, then report
  the mean and std of those five per-seed means *across seeds*. Paired
  significance tests (paired t-test and sign test on the five per-seed
  means) are reported alongside.

### 7.2 Results

Verbatim from `results/seed_study.txt` (5 seeds, 150 shared paired
miscalibration draws, RL = 200,000 PPO timesteps/seed):

| Controller | Nominal fidelity | Robust mean fidelity | Robust mean leakage |
|---|---|---|---|
| Naive constant pulse | 0.9619 | 0.9505 ± 0.0211 | 0.0200 |
| GRAPE (nominal-only) | 0.9867 | 0.9738 ± 0.0000 | 0.0049 |
| Ensemble-GRAPE (robust) | 0.9869 | 0.9741 ± 0.0000 | 0.0046 |
| **RL (live/closed-loop)** | 0.9803 ± 0.0022 | **0.9788 ± 0.0020** | **0.0024** |
| RL (frozen/open-loop) | — | 0.9692 ± 0.0021 | 0.0024 |

**On the zero std for both GRAPE variants:** every one of the 5
optimization seeds converged to a bit-identical pulse. This is not a seeding
bug — `restart_seed` genuinely varies the random-restart initial guesses,
each drawn fresh from `np.random.default_rng(restart_seed)` — it means the
one seeded initial guess every restart set starts from (the naive I-only
$\pi$-pulse, Section 3.1) already sits at, or L-BFGS-B carries it to, the
same optimum every random restart failed to beat. A well-behaved
20-parameter landscape where multiple independent random starts can't beat
one physics-informed starting point is evidence the reported GRAPE numbers
are close to the true optimum for this pulse class, not an artifact.

**Leakage first, because it's the metric that matters most for real
hardware, and because it behaves differently from fidelity:** leakage
escapes the computational subspace entirely, which is what actually breaks
quantum error correction, in a way ordinary (recoverable) infidelity does
not. RL (live) leaks 0.0024 per gate on average, versus 0.0049 for nominal
GRAPE and 0.0046 for ensemble GRAPE — roughly 2x lower, in every one of the
5 seeds (RL's per-seed leakage ranges 0.0022-0.0026; both GRAPE variants'
constant values sit above that entire range). This advantage is close to
freezing-invariant: RL (frozen) leaks 0.0024, essentially the same as RL
(live) (mean difference +0.00002 — smaller than the table's rounding), and
still roughly 2x lower than either GRAPE variant (paired t-test p=4.4e-06
for RL (frozen) vs. nominal GRAPE, consistent in all 5 seeds). The
live-vs-frozen leakage difference is small enough to be practically
negligible, but with variance this tight across seeds it is technically
detectable (paired t-test p=0.023) even though the sign test — which only
asks about direction, not magnitude — is not (4 of 5 seeds favor live,
p=0.19). Leakage suppression, unlike the fidelity gain, does not depend
substantially on live state access — it is overwhelmingly a property of the
waveform itself.

Paired significance (per-seed robust-mean fidelity, n=5):

| Comparison | Mean difference | Paired t-test | Sign test (wins) |
|---|---|---|---|
| RL (live) vs. GRAPE | +0.0050 | p=0.0071 | 5/5, p=0.0312 |
| RL (live) vs. Ensemble-GRAPE | +0.0047 | p=0.0084 | 5/5, p=0.0312 |
| RL (live) vs. RL (frozen) | +0.0096 | p=7.12e-07 | 5/5, p=0.0312 |
| RL (frozen) vs. GRAPE | -0.0047 | p=0.0117 | 0/5, p=1.0000 |

Paired significance (per-seed mean leakage, n=5):

| Comparison | Mean difference | Paired t-test | Sign test (wins) |
|---|---|---|---|
| RL (live) vs. RL (frozen) | +0.00002 | p=0.0231 | 4/5, p=0.1875 |
| RL (frozen) vs. GRAPE | -0.00249 | p=4.42e-06 | 0/5, p=1.0000 |

(For the leakage rows, "wins" counts *higher* leakage per the shared
`paired_stats()` convention — read "0/5" for RL(frozen) vs. GRAPE as RL
(frozen) never leaking *more* than GRAPE, i.e. leaking less in all 5 seeds.
For RL(live) vs. RL(frozen), the mean difference — two-hundredths of a
percentage point of leakage — is the number that matters here; the paired
t-test's significance reflects how tightly clustered that tiny difference
is across seeds, not that it is large.)

### 7.3 The mechanism test: what happens when RL can't react

This is the result that determines what the rest of this section is allowed
to claim. RL (live) beats both GRAPE baselines' robust-mean fidelity — but
RL (live) also has something neither GRAPE variant has: exact, noiseless
mid-episode state (Section 4.1), which is not something a real qubit
readout could provide inside a 10 ns gate. That gap therefore has two
possible explanations that cannot be told apart just by comparing RL
(live) to GRAPE:

1. RL discovered a genuinely more robust *waveform* — one that would stay
   robust even played back open-loop, or
2. RL is exploiting live state information mid-episode, which a fixed pulse
   (of any kind) fundamentally cannot do.

**RL (frozen)** settles this. Taking each seed's trained policy, rolling it
out *once* on the nominal (zero-miscalibration) qubit, freezing the exact
waveform it produced, and replaying that fixed waveform open-loop across
the same 150 paired draws:

- RL (frozen)'s robust-mean fidelity is 0.9692 ± 0.0021, a drop of
  +0.0096 from RL (live)'s 0.9788 ± 0.0020 — consistent
  across all 5 seeds (per-seed drop ranges 0.0093-0.0103, paired t-test
  p=7.12e-07).
- That drop doesn't just erase RL's edge over GRAPE — it inverts it. RL
  (frozen)'s 0.9692 is significantly *worse* than nominal GRAPE's 0.9738
  (mean difference -0.0047, paired t-test p=0.0117, 0 of 5 seeds beat
  GRAPE, sign-test p=1.0000) and worse than ensemble GRAPE's 0.9741 by
  essentially the same margin. Once the policy can no longer react to what
  it observes, it doesn't merely lose its advantage over the two open-loop
  optimal-control baselines — it falls below both of them.

**Reading:** mechanism (2) is the dominant explanation, and it flips the
headline. RL (live)'s advantage over both GRAPE baselines depends
substantially on exploiting mid-episode state information no real
gate-time-scale experiment can provide — not primarily on having found an
inherently more robust pulse shape. Restricted to what a real controller
could actually run open-loop, GRAPE (0.9738) beats RL (0.9692): **on this
benchmark, gradient-based optimal control wins.** This is not a null result
about RL; it is a precise, signed statement of *what kind* of result this
is: strong evidence that a policy with access to fast, low-latency state
estimation (which does not exist for single-shot gate-time readout today,
but is an active hardware research direction) could meaningfully outperform
open-loop optimal control on this task — not a demonstration that today's
fixed-pulse calibration is already beatable by switching to RL. The one
exception is leakage (above), where RL's advantage is overwhelmingly a
property of the waveform itself and survives the same freezing test nearly
intact.

### 7.4 Does ensemble GRAPE close the gap to RL (live)?

This question is secondary to the headline result (nominal GRAPE already
beats the only deployable RL variant) but still matters for characterizing
*why* RL (live) — not deployable, but informative about what live access
buys — retains an edge over both open-loop baselines. Only marginally.
Ensemble GRAPE's robust-mean fidelity (0.9741) is barely
above nominal GRAPE's (0.9738) — a difference of +0.0002, an order of
magnitude smaller than RL (live)'s edge over either baseline (+0.0047 to
+0.0050). RL (live) still beats ensemble GRAPE by 0.0047 (paired t-test
p=0.0084, 5/5 seeds, sign-test p=0.0312) — a gap barely narrower than
against nominal GRAPE's +0.0050. Training on the correct miscalibration
distribution, in other words, recovers almost none of RL's advantage;
consistent with Section 7.3's frozen-policy result, nearly all of that
advantage depends on live mid-episode state access, not on knowing the
right distribution to optimize an open-loop pulse against. Ensemble GRAPE
does modestly reduce leakage relative to nominal GRAPE (0.0046 vs. 0.0049),
but both remain roughly 2x RL (live)'s leakage (0.0024).

### 7.5 Does the learned policy rediscover DRAG?

We checked whether each seed's trained policy's $\Omega_y(t)$ approximates
the standard first-order DRAG correction,
$\Omega_y(t)\approx-\dot\Omega_x(t)/\alpha$ (Section 1.4): correlation
coefficient, plus the $R^2$ of the best-fit single scale factor between the
two, over the frozen (Section 7.3) nominal-rollout pulse
(`grape_baseline.py: drag_correlation_check`). Before trusting a negative
result, we validated the detector itself
(`grape_baseline.py: validate_drag_detector`, run via `drag_check.py`)
against a synthetic Gaussian $\Omega_x(t)$ paired with its *exact* analytic
DRAG-derived $\Omega_y(t)$ — a pair the detector must recognize perfectly if
it works at all:

```
Detector self-test (synthetic, exact DRAG pair): correlation = 1.0000  R^2 = 1.0000
```

A perfect score, as it has to be for an exact pair by construction —
confirming the null result below isn't an artifact of a broken check.
(`results/drag_check.txt` has the full output.)

Run on all 5 `seed_study.py` policies (frozen, one nominal rollout each —
the identical waveform Section 7.3 evaluates):

| Seed | Correlation | $R^2$ |
|---|---|---|
| 0 | +0.02 | -9.47 |
| 1 | +0.83 | -0.89 |
| 2 | -0.32 | -12.18 |
| 3 | +0.86 | **+0.73** |
| 4 | +0.58 | -10.77 |
| **Mean ± std** | **+0.40 ± 0.47** | **-6.51 ± 5.35** |

**Mostly does not match — with one real exception.** In 4 of 5 seeds, $R^2$
is strongly negative: the simple derivative-DRAG prediction is a far worse
predictor of $\Omega_y(t)$ than just using its mean. Seed 3 breaks that
pattern: correlation +0.86 *and* $R^2=+0.73$ — a genuinely good fit, not
just a shape resemblance. (Correlation alone is a weaker signal here:
seeds 1 and 4 also show moderately-high correlation, +0.83 and +0.58, but
still strongly negative $R^2$ — correlation is invariant to a constant
offset, $R^2$'s no-intercept scale fit is not, so a curve that tracks
DRAG's *shape* but sits at the wrong DC level correlates well while still
fitting poorly. Seed 3 is the only one where both agree.) Read plainly:
across 5 independently-trained policies, one converged on something close
to the textbook DRAG relation and four did not — this is not the strong,
uniform null result a single-seed check would have reported, and multi-seed
evaluation is what surfaces that nuance instead of hiding it in a single
data point.

What the other four policies do instead, viewed as a complex drive
$\varepsilon(t)=\Omega_x(t)+i\Omega_y(t)$: the **magnitude** $|\varepsilon(t)|$
decays from $\approx0.39$-$0.49$ rad/ns at the start of the pulse to
$\approx0.15$-$0.17$ rad/ns at the end — consistently, in all 5 seeds,
including seed 3. The **phase** is roughly constant *within* a given seed's
own pulse (drifting by only a few tens of degrees across the 10 steps) but
the specific value differs sharply *across* seeds — observed values span
nearly the full circle (seed 0: $\approx-70°$, seed 1: $\approx+8°$, seed 2:
$\approx+100°$, seed 3: $\approx-158°$, seed 4: $\approx+112°$). Consistent
magnitude decay, inconsistent absolute phase, is what a policy with no
reward term favoring any particular I/Q reference axis (and none favoring
the DRAG relation specifically) would be expected to produce most of the
time: some working leakage-suppression strategy (Section 3.2), usually
genuinely different from textbook DRAG rather than a noisy version of it —
except that, in at least one of five independent training runs, gradient
descent on the same reward found something close to the closed-form
correction anyway.

### 7.6 What this study demonstrates, independent of the scoreboard

Five things had to go right for the headline result to be trustworthy
rather than merely measured: genuinely fair baselines instead of a
strawman comparison (Section 5); a designed test that isolates *mechanism*
rather than just reporting an outcome (Section 7.3); statistics that can
tell a real effect apart from seed noise (Section 7.2); a validated null
result instead of an asserted one (Section 7.5); and a real physics bug —
2.29x too little dephasing, silently affecting every fidelity number this
project had ever produced — caught by testing the actual production code
path rather than a hand-verified stand-in, root-caused to a one-character
fix, and now guarded by a standing regression test (Section 2.1,
`tests/test_physics.py`) so it cannot recur unnoticed. Any one of these
missing would have made the headline result unreliable regardless of which
controller it favored. Producing a result that survives all five is the
actual subject of this document, and it is a harder, more transferable
skill than getting RL to win.

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
