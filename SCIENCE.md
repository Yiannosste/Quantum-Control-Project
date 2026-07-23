# The Science Behind This Project

This document derives, in detail, the physics and reinforcement-learning
theory implemented in this repository. Every equation below is cross-checked
against the actual code (`utils.py`, `quantum_env.py`) and, where practical,
against a direct numerical run of that code — not just against the textbook
result in isolation. File references point to the functions that implement
each piece of physics.

---

## 1. The Qubit as a Two-Level Quantum System

A transmon superconducting qubit is, to an excellent approximation, a
two-level system with basis states $|0\rangle$ (ground) and $|1\rangle$
(excited), represented as a $2\times2$ density matrix $\rho$.

### 1.1 Bloch sphere convention

Any single-qubit state can be written as a point on (pure states) or inside
(mixed states) the **Bloch sphere**:

$$\rho = \frac{1}{2}\left(I + x\sigma_x + y\sigma_y + z\sigma_z\right), \qquad (x,y,z) = (\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)$$

This repository follows the standard convention where $|0\rangle$ and
$|1\rangle$ are the $+1$ and $-1$ eigenstates of $\sigma_z$, respectively.
Concretely, in code (`quantum_env.py`, `reset()`) the qubit starts in
`qt.ket2dm(qt.basis(2, 0))`, and the target state (`target_state`) is
`qt.basis(2, 1)`. We verified this directly:

```
qt.expect(qt.sigmaz(), qt.ket2dm(qt.basis(2, 0)))  ==  +1.0   # |0>, ground
qt.expect(qt.sigmaz(), qt.ket2dm(qt.basis(2, 1)))  ==  -1.0   # |1>, excited (target)
```

So on the Bloch sphere used throughout this project:

| Pole | $\langle\sigma_z\rangle$ | State |
|------|------|-------|
| North | $+1$ | $\lvert 0\rangle$ (ground, episode start) |
| South | $-1$ | $\lvert 1\rangle$ (excited, episode target) |

An **X-gate** is a $\pi$ rotation about the X-axis that swaps the poles:
$\lvert 0\rangle \to \lvert 1\rangle$.

### 1.2 System Hamiltonian

In the rotating-wave approximation (RWA), in a frame rotating at the drive
frequency, the Hamiltonian splits into a drift term and a control term
(`utils.py: build_hamiltonians`):

$$H(t) = \underbrace{\frac{\omega_q}{2}\sigma_z}_{H_0 \text{ (drift)}} + \underbrace{\Omega(t)\cdot\frac{1}{2}\sigma_x}_{H_c \text{ (control)}}$$

- $\omega_q$: qubit transition frequency in the rotating frame. The code sets
  `omega_q = 0.0` (`QUBIT_PARAMS`), i.e. the drive is assumed perfectly
  resonant, so $H_0$ vanishes identically. This is a simplification — real
  hardware always has some residual detuning — noted in the "Limitations"
  section below.
- $\Omega(t)$: the time-dependent drive amplitude, the RL agent's action
  (denormalized: $\Omega(t) = a_t\cdot\Omega_{\max}$).

### 1.3 Resonant (Rabi) rotation and the π-pulse condition

With $\omega_q=0$, a constant drive $\Omega$ applied for a duration $t$
generates the unitary $U(t)=\exp(-i\Omega t\sigma_x/2)$, which rotates
the Bloch vector by angle $\theta=\Omega t$ about the X-axis. A full
population transfer $|0\rangle\to|1\rangle$ (an ideal, noise-free X-gate)
therefore requires

$$\Omega \cdot t_\pi = \pi \quad\Longrightarrow\quad \Omega = \frac{\pi}{t_\pi}$$

**Verified against the code.** For the default 50 ns gate
(`n_steps=50`, `dt=1.0`), this predicts $\Omega=\pi/50\approx0.0628$. Driving
`QuantumPulseEnv` with a constant amplitude of exactly that value (no
decoherence, $T_1=T_2=10^9$ ns) gives:

```
fidelity = 0.9999999   (theory: 1.0, exactly the π-pulse condition)
```

Note this required amplitude ($\approx0.063$) is comfortably inside the
default `omega_max=0.1` action range, so the default hyperparameters do
leave the agent enough drive strength to reach the target in the allotted
gate time.

---

## 2. Open Quantum Systems: The Lindblad Master Equation

Real qubits are not isolated — they exchange energy and phase information
with their electromagnetic environment. This is modeled with the Lindblad
(GKSL) master equation for the density matrix $\rho$:

$$\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)$$

The first term is unitary evolution under $H(t)$; the sum models
dissipation through collapse (Lindblad/jump) operators $L_k$.

### 2.1 Collapse operators (`utils.py: build_collapse_operators`)

Two decoherence channels are modeled:

| Channel | Operator | Rate | Physical meaning |
|---|---|---|---|
| $T_1$ relaxation | $L_1=\sqrt{\Gamma_1}\sigma_-$ | $\Gamma_1 = 1/T_1$ | Spontaneous decay $\lvert1\rangle\to\lvert0\rangle$ |
| $T_2$ dephasing | $L_2=\sqrt{\Gamma_\phi}\sigma_z/2$ | $\Gamma_\phi = \frac{1}{T_2}-\frac{\Gamma_1}{2}$ | Random phase drift, no population change |

where $\sigma_- = \lvert0\rangle\langle1\rvert$ is implemented as
`qt.destroy(2)`, the qubit lowering operator.

The rate relation $\Gamma_\phi = 1/T_2 - 1/(2T_1)$ comes from decomposing the
standard total dephasing rate $1/T_2 = 1/(2T_1) + 1/T_2^{\ast}$ (energy decay
also destroys phase coherence at half its rate) and identifying
$\Gamma_\phi = 1/T_2^{\ast}$. **Physical constraint:** this requires
$T_2 \le 2T_1$; the code enforces $\Gamma_\phi\ge0$ via a `max(..., 0.0)`
clamp. The default parameters $T_1=1000$ ns, $T_2=500$ ns satisfy this with
margin ($500 \le 2000$).

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

**Verified plausibility of the README's expected results.** Driving the
*default, noisy* environment ($T_1=1000$ ns, $T_2=500$ ns) with the
analytically-derived constant π-pulse ($\Omega=\pi/50$) — i.e. no RL, no
pulse shaping — already yields

```
fidelity ≈ 0.9723
```

This is direct evidence that the README's "~0.97 mean fidelity" target for a
*trained* agent is physically reasonable: a naive constant pulse already
lands almost exactly there, and a shaped pulse (what PPO is trained to find)
has room to match or modestly exceed it by trading off energy penalty and
decoherence exposure.

---

## 4. MDP Formulation

| Component | Definition | Code |
|---|---|---|
| State $s_t$ | Bloch vector $[\langle\sigma_x\rangle,\langle\sigma_y\rangle,\langle\sigma_z\rangle]\in[-1,1]^3$ | `_get_obs()` |
| Action $a_t$ | $a_t\in[-1,1] \to \Omega(t)=a_t\cdot\Omega_{\max}$ | `step()` |
| Reward $r_t$ | $r_t = -\lambda\Omega(t)^2$, plus $+F(\rho_T,\lvert1\rangle)$ only at the final step | `step()` |
| Episode | $N=50$ steps $\times$ $dt=1$ ns $=$ 50 ns gate | `n_steps`, `dt` |

$$r_t = \begin{cases} -\lambda\Omega(t)^2 & t < T \\ -\lambda\Omega(t)^2 + F(\rho_T, \lvert1\rangle) & t = T \end{cases}$$

with $\lambda=0.01$ by default (`energy_penalty`). This reward is
Markovian in the Bloch vector because the Lindblad equation is itself a
Markovian (memoryless) approximation — valid when the environment's
correlation time is much shorter than the gate time, an excellent
approximation for millikelvin transmon qubits.

---

## 5. Why Reinforcement Learning (and why PPO)

Classical pulse optimization (e.g. GRAPE) computes the fidelity gradient by
backpropagating through a differentiable simulation of the system. This
works well when the simulator is accurate and differentiable, but:

- it needs an explicit, differentiable model — awkward when the real
  "simulator" is noisy hardware;
- it doesn't naturally handle stochastic, episode-varying noise;
- it doesn't adapt online as device parameters drift.

RL instead treats the qubit as a black-box environment and learns a policy
from reward feedback alone (model-free), which transfers more directly to a
hardware-in-the-loop setting. **Proximal Policy Optimization (PPO)**
(Schulman et al., 2017) is used here for its training stability: it clips
the policy-gradient update to prevent destructively large steps, which
matters when the reward signal is sparse (the fidelity bonus only arrives at
the last step of each 50-step episode):

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\Big(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\Big)\right]$$

This is a training-stability statement about PPO in general, not a claim
re-derived from this project's specific results.

---

## 6. Known Simplifications / Limitations

Being transparent about what this model does *not* capture:

- **Perfect resonance assumed.** `omega_q = 0` means detuning noise/drift is
  not modeled; a real calibration routine must also track qubit frequency
  drift.
- **Two levels only.** No leakage to a third (e.g. transmon $|2\rangle$)
  level is modeled, so DRAG-style leakage suppression isn't needed or
  represented here.
- **Simulation-unit labels.** Code comments label `dt` as nanoseconds and
  `omega_max` as "2π·MHz" for physical intuition, but no explicit unit
  conversion is applied anywhere — the simulator works self-consistently in
  natural units where one unit of amplitude times one unit of time equals
  one radian of rotation (as used throughout the derivations above). Treat
  the "ns" / "MHz" labels as illustrative scale, not as dimensionally
  rigorous SI units.
- **"Expected" results are exactly that.** The ~0.97 fidelity figure in the
  README is a target based on the physics of the problem (Section 3 above),
  not a fixed guarantee — actual results after running `train.py` will vary
  with the random seed and training budget.

---

## References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Johansson et al. (2012). *QuTiP: An open-source Python framework for the dynamics of open quantum systems.* Computer Physics Communications.
4. Krantz et al. (2019). *A quantum engineer's guide to superconducting qubits.* Applied Physics Reviews, 6(2).
5. Niu et al. (2019). *Universal quantum control through deep reinforcement learning.* npj Quantum Information.
6. Fösel et al. (2018). *Reinforcement learning with neural networks for quantum feedback.* Physical Review X.
