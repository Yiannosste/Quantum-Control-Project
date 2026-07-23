"""
utils.py
--------
Physics helpers: Hamiltonians, collapse operators, fidelity, and plotting.

Physical model
--------------
Drift Hamiltonian  (lab-frame, rotating-wave approximation):
    H0 = (omega_q / 2) * sigma_z

Control Hamiltonian (resonant microwave drive):
    Hc = (1/2) * sigma_x

Full Hamiltonian:
    H(t) = H0 + Omega(t) * Hc

Open-system Lindblad terms:
    T1  (energy relaxation) : c1 = sqrt(1/T1) * sigma_minus
    T2* (pure dephasing)    : c2 = sqrt(1/T_phi) * sigma_z / 2
    where  1/T2 = 1/(2*T1) + 1/T_phi

All frequencies in units of 2pi * MHz, times in ns.
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── Default physical parameters ───────────────────────────────────────────────
QUBIT_PARAMS = {
    "omega_q": 0.0,    # qubit frequency in rotating frame [2pi * MHz]
                       # (set to 0 → perfect resonance; add detuning to explore)
    "T1": 1000.0,      # energy relaxation time  [ns]  (~1 µs typical transmon)
    "T2": 500.0,       # total dephasing time    [ns]  (~500 ns typical)
}


# ── Hamiltonian construction ──────────────────────────────────────────────────

def build_hamiltonians(params: dict = None) -> tuple:
    """
    Returns (H0, Hc) as QuTiP Qobj operators.

    H0 : drift (qubit frequency term)
    Hc : control (microwave drive coupling)
    """
    if params is None:
        params = QUBIT_PARAMS

    omega_q = params.get("omega_q", 0.0)

    H0 = (omega_q / 2.0) * qt.sigmaz()   # drift
    Hc = 0.5 * qt.sigmax()               # control coupling
    return H0, Hc


# ── Lindblad collapse operators ───────────────────────────────────────────────

def build_collapse_operators(params: dict = None) -> list:
    """
    Returns a list of QuTiP collapse operators for T1 and T2 processes.

    c1 = sqrt(Gamma1) * sigma_minus   (amplitude damping / energy relaxation)
    c2 = sqrt(Gamma_phi) * sigma_z/2  (pure dephasing)

    where:
        Gamma1   = 1 / T1
        Gamma2   = 1 / T2
        Gamma_phi = Gamma2 - Gamma1/2   (pure dephasing rate)
    """
    if params is None:
        params = QUBIT_PARAMS

    T1  = params.get("T1", 1e6)   # very long if not specified
    T2  = params.get("T2", 1e6)

    Gamma1   = 1.0 / T1
    Gamma2   = 1.0 / T2
    Gamma_phi = max(Gamma2 - Gamma1 / 2.0, 0.0)  # must be non-negative

    c_ops = []
    if Gamma1 > 0:
        c_ops.append(np.sqrt(Gamma1) * qt.destroy(2))          # sigma_minus
    if Gamma_phi > 0:
        c_ops.append(np.sqrt(Gamma_phi) * qt.sigmaz() / 2.0)   # dephasing

    return c_ops


# ── Fidelity ──────────────────────────────────────────────────────────────────

def compute_fidelity(rho: qt.Qobj, target_ket: qt.Qobj) -> float:
    """
    State fidelity F = <psi_target| rho |psi_target>.

    Works for both pure-state kets and density matrices.

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
    amplitudes: np.ndarray,
    trajectory: np.ndarray,
    dt: float = 1.0,
    fidelity: float = None,
    save_path: str = None,
):
    """
    Two-panel figure:
      Left  : optimized pulse shape Omega(t) vs time
      Right : Bloch sphere trajectory

    Parameters
    ----------
    amplitudes : 1-D array of pulse amplitudes [2pi MHz]
    trajectory : (n_steps+1, 3) array of Bloch coordinates
    dt         : timestep duration [ns]
    fidelity   : final fidelity (displayed in title if provided)
    save_path  : if given, save the figure to this path
    """
    times = np.arange(len(amplitudes)) * dt

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Left: Pulse shape ────────────────────────────────────────────────
    ax_pulse = fig.add_subplot(gs[0])
    ax_pulse.step(times, amplitudes / (2 * np.pi), where="post",
                  color="#2196F3", linewidth=2)
    ax_pulse.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_pulse.set_xlabel("Time (ns)", fontsize=12)
    ax_pulse.set_ylabel(r"Drive amplitude $\Omega(t)$ / $2\pi$ (MHz)", fontsize=12)
    ax_pulse.set_title("Optimized Pulse Shape", fontsize=13, fontweight="bold")
    ax_pulse.grid(True, alpha=0.3)

    # ── Right: Bloch sphere ──────────────────────────────────────────────
    ax_bloch = fig.add_subplot(gs[1], projection="3d")
    _draw_bloch_sphere(ax_bloch)

    xs, ys, zs = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    # Colour-code trajectory by time (blue → red)
    n = len(xs) - 1
    for i in range(n):
        color = plt.cm.plasma(i / max(n - 1, 1))
        ax_bloch.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                      color=color, linewidth=2)

    # Start / end markers
    ax_bloch.scatter(*trajectory[0],  color="green",  s=80, zorder=5, label="|0⟩ start")
    ax_bloch.scatter(*trajectory[-1], color="red",    s=80, zorder=5, label="|1⟩ end")
    ax_bloch.scatter(0, 0, -1,        color="gold",   s=60, zorder=5,
                     marker="*", label="|1⟩ target")

    ax_bloch.set_xlabel("X", fontsize=10)
    ax_bloch.set_ylabel("Y", fontsize=10)
    ax_bloch.set_zlabel("Z", fontsize=10)
    title = "Bloch Sphere Trajectory"
    if fidelity is not None:
        title += f"\nFinal Fidelity = {fidelity:.4f}"
    ax_bloch.set_title(title, fontsize=12, fontweight="bold")
    ax_bloch.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Figure saved → {save_path}")
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
        print(f"[utils] Training curve saved → {save_path}")
    plt.show()
    return fig
