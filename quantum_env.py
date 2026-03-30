"""
quantum_env.py
--------------
Custom Gymnasium environment for RL-driven quantum pulse optimization.

The agent controls the amplitude of a microwave drive pulse at each timestep
to steer a superconducting qubit from |0> to |1> (X-gate), maximizing fidelity
while minimizing pulse energy.

MDP Formulation:
  State  : Bloch sphere coordinates [<X>, <Y>, <Z>]  (3-vector in [-1, 1]^3)
  Action : Normalized drive amplitude Omega(t) in [-1, 1]
  Reward : +fidelity bonus at final step, -energy penalty each step
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import qutip as qt
from utils import (
    build_hamiltonians,
    build_collapse_operators,
    compute_fidelity,
    QUBIT_PARAMS,
)


class QuantumPulseEnv(gym.Env):
    """
    Single-qubit X-gate environment with Lindblad open-system dynamics.

    Parameters
    ----------
    n_steps        : number of discrete time steps per episode
    dt             : duration of each time step [ns]
    omega_max      : maximum drive amplitude [MHz * 2pi]
    energy_penalty : weight for the pulse-energy penalty term
    params         : dict of qubit physical parameters (see utils.QUBIT_PARAMS)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_steps: int = 50,
        dt: float = 1.0,          # ns
        omega_max: float = 0.1,   # 2pi * MHz
        energy_penalty: float = 0.01,
        params: dict = None,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.dt = dt
        self.omega_max = omega_max
        self.energy_penalty = energy_penalty
        self.params = params if params is not None else QUBIT_PARAMS

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Action: single continuous value normalised to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Observation: [<X>, <Y>, <Z>] on the Bloch sphere
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ── Physics objects (built once) ──────────────────────────────────
        self.H0, self.Hc = build_hamiltonians(self.params)
        self.c_ops = build_collapse_operators(self.params)

        # Target state: |1> = excited state
        self.target_state = qt.basis(2, 1)

        # Pauli operators for expectation values
        self.sx = qt.sigmax()
        self.sy = qt.sigmay()
        self.sz = qt.sigmaz()

        # Episode state
        self._state_dm: qt.Qobj = None
        self._step_count: int = 0
        self._pulse_amplitudes: list = []

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start in ground state |0><0|
        self._state_dm = qt.ket2dm(qt.basis(2, 0))
        self._step_count = 0
        self._pulse_amplitudes = []
        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        # Denormalize action to physical units
        amplitude = float(action[0]) * self.omega_max  # [2pi MHz]
        self._pulse_amplitudes.append(amplitude)

        # Build time-dependent Hamiltonian: H = H0 + amplitude * Hc
        H = [self.H0, [self.Hc, self._constant_coeff(amplitude)]]

        # Integrate Lindblad master equation for one timestep
        tlist = np.linspace(0, self.dt, 10)
        result = qt.mesolve(
            H,
            self._state_dm,
            tlist,
            self.c_ops,
            options=qt.Options(nsteps=5000, rtol=1e-8, atol=1e-10),
        )
        self._state_dm = result.states[-1]
        self._step_count += 1

        terminated = self._step_count >= self.n_steps

        # Reward
        reward = -self.energy_penalty * amplitude ** 2  # energy penalty every step
        if terminated:
            fidelity = compute_fidelity(self._state_dm, self.target_state)
            reward += fidelity  # large positive reward for high fidelity

        obs = self._get_obs()
        info = {}
        if terminated:
            fidelity = compute_fidelity(self._state_dm, self.target_state)
            info["fidelity"] = float(fidelity)
            info["pulse_amplitudes"] = list(self._pulse_amplitudes)

        return obs.astype(np.float32), float(reward), terminated, False, info

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return Bloch-sphere coordinates as the observation."""
        ex = float(qt.expect(self.sx, self._state_dm))
        ey = float(qt.expect(self.sy, self._state_dm))
        ez = float(qt.expect(self.sz, self._state_dm))
        return np.array([ex, ey, ez], dtype=np.float32)

    @staticmethod
    def _constant_coeff(amplitude: float):
        """Return a QuTiP-compatible coefficient function for a constant amplitude."""
        def coeff(t, args=None):
            return amplitude
        return coeff

    def get_bloch_trajectory(self, amplitudes: list) -> np.ndarray:
        """
        Re-simulate an episode with the given pulse amplitudes and return
        the full Bloch-sphere trajectory as an (n_steps+1, 3) array.
        Used for evaluation / plotting.
        """
        dm = qt.ket2dm(qt.basis(2, 0))
        trajectory = [self._dm_to_bloch(dm)]
        for amp in amplitudes:
            H = [self.H0, [self.Hc, self._constant_coeff(amp)]]
            tlist = np.linspace(0, self.dt, 10)
            result = qt.mesolve(
                H, dm, tlist, self.c_ops,
                options=qt.Options(nsteps=5000, rtol=1e-8, atol=1e-10),
            )
            dm = result.states[-1]
            trajectory.append(self._dm_to_bloch(dm))
        return np.array(trajectory)

    def _dm_to_bloch(self, dm: qt.Qobj) -> np.ndarray:
        ex = float(qt.expect(self.sx, dm))
        ey = float(qt.expect(self.sy, dm))
        ez = float(qt.expect(self.sz, dm))
        return np.array([ex, ey, ez])
