"""
utils.py
--------
Physics helpers: Hamiltonians, collapse operators, fidelity, and plotting.

Physical model (weakly-anharmonic transmon, truncated to `num_levels`
levels, 3 by default: |0>, |1>, |2>)
--------------------------------------------------------------------
Drift Hamiltonian, in the frame rotating at the drive frequency:
    H0 = -Delta * n  +  (alpha/2) * n * (n - 1)
where n = a^dag a is the number operator, Delta is the qubit-drive
detuning, and alpha < 0 is the anharmonicity (the |1>-|2> transition sits
alpha below the |0>-|1> transition, as in a real transmon). At n=0,1 the
anharmonic term vanishes identically; at n=2 it contributes exactly alpha,
so |2> is detuned by alpha from where a perfectly harmonic ladder would
put it. This is what makes a resonant |0>-|1> drive only weakly leak
population into |2> rather than resonantly driving it.

Control Hamiltonian (I/Q microwave drive, resonant with the 0-1 transition):
    Hx = (a + a^dag) / 2      (in-phase / "I" quadrature)
    Hy = i(a^dag - a) / 2     (quadrature / "Q" quadrature)
    H(t) = H0 + Omega_x(t) Hx + Omega_y(t) Hy

Driving only Hx (Omega_y=0) is what a naive two-level pulse does; because
`a` couples every adjacent pair of levels, this also drives 1<->2 (leakage).
The Q quadrature is exactly the extra control real hardware uses for DRAG
(Derivative Removal by Adiabatic Gate, Motzoi et al. 2009) to cancel that
leakage to leading order. This project's RL agent is given that same
control and has to learn to use it, rather than being handed the DRAG
formula directly.

Truncating to num_levels=2 recovers the standard two-level qubit model exactly:
a -> sigma_minus, Hx -> sigma_x/2, and Hy/leakage become moot.

Open-system Lindblad terms:
    T1  (energy relaxation) : c1 = sqrt(1/T1) * a
        Because <n|a|n> has matrix element sqrt(n) between adjacent levels,
        this gives Gamma(1->0) = 1/T1 and Gamma(2->1) = 2/T1 automatically
        (the standard "harmonic-oscillator" scaling assumed for a weakly
        anharmonic transmon absent a separately-measured T1 for |2>).
    T2* (pure dephasing)    : c2 = sqrt(2*Gamma_phi) * n
        where  1/T2 = 1/(2*T1) + Gamma_phi
        (see the note on the prefactor below for the derivation).

All frequencies in units of rad/ns (equivalently 2*pi*GHz); times in ns.
This is dimensionally literal, not a "natural units" rescaling: e.g.
omega_max=0.35 rad/ns is exactly a 2*pi*55.7 MHz Rabi rate, consistent with
real transmon single-qubit drive strengths.

Note on the T2 collapse-operator prefactor
-------------------------------------------
For a diagonal Hermitian Lindblad operator L with eigenvalues l_k on level
k, the coherence rho_jk decays at rate (1/2)*(l_j - l_k)^2 (from the GKSL
dissipator D[L]rho = L rho L^dag - (1/2){L^dag L, rho}). The number operator
n has eigenvalues 0, 1, 2, ..., so consecutive levels are spaced by exactly 1,
the same spacing as the qubit operator sigma_z/2 (eigenvalues +-1/2). That
means n is already the direct multi-level analog of sigma_z/2; it does not
need an extra 1/2 the way sigma_z itself (spacing 2) would. With
L = sqrt(2*Gamma_phi)*n, the 0-1 coherence decay rate works out to exactly
Gamma_phi, as required by 1/T2 = 1/(2*T1) + Gamma_phi.

A tempting but wrong alternative is sqrt(2*Gamma_phi)*n/2 (carrying the
sigma_z/2-style "/2" over to n despite n not needing one), which gives a
decay rate of only Gamma_phi/4, not Gamma_phi. Verified numerically (equal
superposition, no drive, coherence-decay fit): that form simulates T2 as
1142.9 ns against a QUBIT_PARAMS["T2"] label of 500 ns (2.29x too slow,
i.e. 2.29x too little dephasing). The form used here (no extra /2)
reproduces the labeled T2 to 4 significant figures. See SCIENCE.md, Section
2, for the full derivation and verification numbers, and
tests/test_physics.py for a standing regression test against exactly this
failure mode.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Default physical parameters ───────────────────────────────────────────────
QUBIT_PARAMS = {
    "omega_q": 0.0,     # qubit-drive detuning [rad/ns] (0 -> perfect resonance)
    "T1": 1000.0,       # energy relaxation time  [ns]  (~1 us typical transmon)
    "T2": 500.0,        # total dephasing time    [ns]  (~500 ns typical)
    "alpha": -1.5708,   # anharmonicity [rad/ns] = -2*pi*250 MHz, typical transmon
    "num_levels": 3,    # |0>, |1>, |2> (leakage-capable transmon model)
}


# ── Hamiltonian construction ──────────────────────────────────────────────────

def build_hamiltonians(params: dict = None, num_levels: int = None) -> tuple:
    """
    Returns (H0, Hx, Hy) as QuTiP Qobj operators on a `num_levels`-dimensional
    truncated Hilbert space.

    H0 : drift (detuning + anharmonicity)
    Hx : in-phase ("I") control coupling
    Hy : quadrature ("Q") control coupling
    """
    if params is None:
        params = QUBIT_PARAMS
    if num_levels is None:
        num_levels = params.get("num_levels", 3)

    detuning = params.get("omega_q", 0.0)
    alpha = params.get("alpha", 0.0)

    a = qt.destroy(num_levels)
    n = a.dag() * a

    H0 = -detuning * n + (alpha / 2.0) * n * (n - 1)
    Hx = 0.5 * (a + a.dag())
    Hy = 0.5 * 1j * (a.dag() - a)
    return H0, Hx, Hy


# ── Lindblad collapse operators ───────────────────────────────────────────────

def build_collapse_operators(params: dict = None, num_levels: int = None) -> list:
    """
    Returns a list of QuTiP collapse operators for T1 and T2 processes on a
    `num_levels`-dimensional truncated Hilbert space.

    c1 = sqrt(Gamma1) * a        (amplitude damping, n-scaled: 2->1 decays
                                   twice as fast as 1->0, per <n|a|n-1>=sqrt(n))
    c2 = sqrt(2*Gamma_phi) * n   (pure dephasing; see module docstring for
                                   the derivation: n already has the same
                                   consecutive-level spacing as sigma_z/2, so
                                   no extra 1/2 belongs here)

    where:
        Gamma1    = 1 / T1
        Gamma2    = 1 / T2
        Gamma_phi = Gamma2 - Gamma1/2   (pure dephasing rate)
    """
    if params is None:
        params = QUBIT_PARAMS
    if num_levels is None:
        num_levels = params.get("num_levels", 3)

    T1 = params.get("T1", 1e6)   # very long if not specified
    T2 = params.get("T2", 1e6)

    Gamma1 = 1.0 / T1
    Gamma2 = 1.0 / T2
    Gamma_phi = max(Gamma2 - Gamma1 / 2.0, 0.0)  # must be non-negative

    a = qt.destroy(num_levels)
    n = a.dag() * a

    c_ops = []
    if Gamma1 > 0:
        c_ops.append(np.sqrt(Gamma1) * a)
    if Gamma_phi > 0:
        c_ops.append(np.sqrt(2.0 * Gamma_phi) * n)

    return c_ops


# ── Fidelity ──────────────────────────────────────────────────────────────────

def compute_fidelity(rho: qt.Qobj, target_ket: qt.Qobj) -> float:
    """
    State fidelity F = <psi_target| rho |psi_target>.

    Works for both pure-state kets and density matrices, and for any
    Hilbert space dimension (rho and target_ket must match). For a target
    that is |1> in a 3-level space, population that has leaked to |2> (or
    stayed in |0>) is automatically counted as infidelity here, with no
    separate leakage term needed.

    Parameters
    ----------
    rho        : final simulated state (density matrix or ket)
    target_ket : desired target state as a ket vector

    Returns
    -------
    float in [0, 1]
    """
    if rho.type == "ket":
        rho = qt.ket2dm(rho)
    target_dm = qt.ket2dm(target_ket)
    fidelity = float((target_dm * rho).tr().real)
    return np.clip(fidelity, 0.0, 1.0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pulse_and_bloch(
    amplitudes_x: np.ndarray,
    amplitudes_y: np.ndarray,
    populations: np.ndarray,
    bloch01: np.ndarray,
    dt: float = 1.0,
    fidelity: float = None,
    save_path: str = None,
):
    """
    Three-panel figure for the 3-level (leakage-capable) environment:
      Left   : I/Q pulse shapes Omega_x(t), Omega_y(t)
      Middle : level populations P0(t), P1(t), P2(t), makes leakage visible
      Right  : Bloch-sphere trajectory of the 0-1 subspace (projected;
               ignores leakage population, shown separately in the middle
               panel)

    Parameters
    ----------
    amplitudes_x, amplitudes_y : 1-D arrays of pulse amplitudes [rad/ns]
    populations                : (n_steps+1, 3) array of [P0, P1, P2]
    bloch01                    : (n_steps+1, 3) array of 0-1 subspace
                                  Bloch coordinates [<sx01>, <sy01>, <sz01>]
    dt                         : timestep duration [ns]
    fidelity                   : final fidelity (displayed in title if given)
    save_path                  : if given, save the figure to this path
    """
    times = np.arange(len(amplitudes_x)) * dt

    fig = plt.figure(figsize=(19, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── Left: I/Q pulse shapes ──────────────────────────────────────────
    ax_pulse = fig.add_subplot(gs[0])
    ax_pulse.step(times, amplitudes_x, where="post",
                  color="#2196F3", linewidth=2, label=r"$\Omega_x$ (I)")
    ax_pulse.step(times, amplitudes_y, where="post",
                  color="#E53935", linewidth=2, label=r"$\Omega_y$ (Q, DRAG-like)")
    ax_pulse.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_pulse.set_xlabel("Time (ns)", fontsize=12)
    ax_pulse.set_ylabel(r"Drive amplitude $\Omega$ (rad/ns)", fontsize=12)
    ax_pulse.set_title("Optimized I/Q Pulse Shape", fontsize=13, fontweight="bold")
    ax_pulse.legend(fontsize=10)
    ax_pulse.grid(True, alpha=0.3)

    # ── Middle: level populations (leakage) ─────────────────────────────
    ax_pop = fig.add_subplot(gs[1])
    pop_times = np.arange(len(populations)) * dt
    ax_pop.plot(pop_times, populations[:, 0], color="#43A047", linewidth=2, label=r"$P_0$")
    ax_pop.plot(pop_times, populations[:, 1], color="#1E88E5", linewidth=2, label=r"$P_1$")
    ax_pop.plot(pop_times, populations[:, 2], color="#FB8C00", linewidth=2, label=r"$P_2$ (leakage)")
    ax_pop.set_xlabel("Time (ns)", fontsize=12)
    ax_pop.set_ylabel("Population", fontsize=12)
    ax_pop.set_ylim([-0.05, 1.05])
    ax_pop.set_title("Level Populations", fontsize=13, fontweight="bold")
    ax_pop.legend(fontsize=10)
    ax_pop.grid(True, alpha=0.3)

    # ── Right: Bloch sphere (0-1 subspace, leakage ignored here) ────────
    ax_bloch = fig.add_subplot(gs[2], projection="3d")
    _draw_bloch_sphere(ax_bloch)

    xs, ys, zs = bloch01[:, 0], bloch01[:, 1], bloch01[:, 2]
    n = len(xs) - 1
    for i in range(n):
        color = plt.cm.plasma(i / max(n - 1, 1))
        ax_bloch.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                      color=color, linewidth=2)

    ax_bloch.scatter(*bloch01[0],  color="green",  s=80, zorder=5, label="|0⟩ start")
    ax_bloch.scatter(*bloch01[-1], color="red",    s=80, zorder=5, label="|1⟩ end")
    ax_bloch.scatter(0, 0, -1,     color="gold",   s=60, zorder=5,
                     marker="*", label="|1⟩ target")

    ax_bloch.set_xlabel("X", fontsize=10)
    ax_bloch.set_ylabel("Y", fontsize=10)
    ax_bloch.set_zlabel("Z", fontsize=10)
    title = "0-1 Subspace Bloch Trajectory"
    if fidelity is not None:
        title += f"\nFinal Fidelity = {fidelity:.4f}"
    ax_bloch.set_title(title, fontsize=12, fontweight="bold")
    ax_bloch.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Figure saved -> {save_path}")
    plt.show()
    return fig


def _draw_bloch_sphere(ax):
    """Draw a wireframe Bloch sphere on 3-D axes."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.4, alpha=0.5)

    # Axes lines
    # Note: with sigma_z eigenstates basis(2,0)=|0> and basis(2,1)=|1>,
    # <sigma_z> = +1 at |0> (north pole) and -1 at |1> (south pole).
    for vec, label in [([1,0,0],"|+X⟩"), ([-1,0,0],"|−X⟩"),
                        ([0,1,0],"|+Y⟩"), ([0,-1,0],"|−Y⟩"),
                        ([0,0,1],"|0⟩"),  ([0,0,-1],"|1⟩")]:
        ax.plot([0, vec[0]*1.15], [0, vec[1]*1.15], [0, vec[2]*1.15],
                color="gray", linewidth=0.6, linestyle="--")
        ax.text(vec[0]*1.2, vec[1]*1.2, vec[2]*1.2, label, fontsize=7,
                ha="center", color="dimgray")

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])


def plot_training_rewards(rewards: list, window: int = 50, save_path: str = None):
    """
    Plot episodic rewards and a rolling average during training.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.4, color="#90CAF9", label="Episode reward")
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(rewards)), rolling,
                color="#1565C0", linewidth=2, label=f"{window}-ep rolling avg")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("PPO Training Curve", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Training curve saved -> {save_path}")
    plt.show()
    return fig
