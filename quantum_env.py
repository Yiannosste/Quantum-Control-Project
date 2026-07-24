"""
quantum_env.py
--------------
Custom Gymnasium environment for RL-driven quantum pulse optimization on a
3-level (leakage-capable) transmon qubit.

The agent controls two quadratures (I and Q) of a microwave drive pulse at
each timestep to steer the qubit from |0> to |1> (X-gate), maximizing
fidelity while minimizing pulse energy and leakage to |2>. Three features
distinguish this from a toy two-level model:

1. Leakage. Driving the 0-1 transition also weakly drives 1-2 (Section 1.2,
   SCIENCE.md); the Q quadrature is the same real degree of freedom DRAG
   pulses use to suppress it (Motzoi et al. 2009) -- the agent has to find
   that suppression itself, it is not handed the DRAG formula.
2. Bandwidth-limited control. The Hamiltonian never sees the agent's raw
   command; it sees that command passed through a first-order RC filter
   (`bandwidth_tau`), modeling a real AWG's finite rise time. Instant
   piecewise-constant jumps are not physically realizable.
3. Per-episode miscalibration. At reset(), a random qubit-drive detuning and
   a random multiplicative amplitude-calibration error are sampled and held
   fixed for that episode -- and are NOT part of the observation. A policy
   can only compensate for them by inferring their effect from the observed
   state trajectory so far. Note that the observation IS the exact, noiseless
   simulator density matrix (see _get_obs below) -- real qubit readout is
   destructive and far slower than this 10 ns gate, so this is a simulation-
   training convenience, not a physically realizable feedback scheme. See
   grape_baseline.py's distill_policy() for the direct test of how much of
   any RL-vs-GRAPE gap depends on that (unrealistic) access, by freezing a
   trained policy into a fixed, open-loop pulse and re-evaluating it.

MDP Formulation:
  State  : 8-D generalized Bloch vector (populations + 01/12/02 coherences)
  Action : Normalized (Omega_x, Omega_y) in [-1, 1]^2
  Reward : -energy penalty -leakage penalty every step, +fidelity at the end
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
    3-level X-gate environment with Lindblad open-system dynamics, I/Q
    control, a bandwidth-limited AWG model, and per-episode miscalibration.

    Parameters
    ----------
    n_steps         : number of discrete time steps per episode
    dt              : duration of each time step [ns]
    omega_max       : maximum commanded drive amplitude per quadrature [rad/ns]
    energy_penalty  : weight for the pulse-energy penalty term
    leakage_penalty : weight for the per-step |2>-population penalty
    bandwidth_tau   : RC low-pass time constant modeling finite AWG bandwidth
                      [ns]. The qubit sees a filtered ("lagged") version of
                      the commanded amplitude, not the commanded value
                      itself. 0 disables filtering (ideal, instantaneous AWG).
    detuning_std    : stddev of the per-episode random qubit-drive detuning
                      [rad/ns], sampled at reset() and NOT observable.
    amp_error_std   : stddev of the per-episode random multiplicative
                      amplitude-calibration error (e.g. 0.05 = 5%), sampled
                      at reset() and NOT observable.
    params          : dict of qubit physical parameters (see utils.QUBIT_PARAMS)

    Note on defaults
    ----------------
    Two error sources pull in opposite directions: decoherence (worse for
    slower gates) and leakage to |2> (worse for faster gates, since a
    stronger drive is needed). A naive, leakage-blind pulse's fidelity
    actually peaks around a 20 ns gate and gets *worse* below that -- see
    SCIENCE.md, Section 3, for the measured curve. The 10 ns default used
    here is only a good choice because omega_y gives the policy a real
    leakage-suppression channel to use; without it, a slower gate would be
    the better naive strategy. Bandwidth limiting and miscalibration are
    further, independent error sources layered on top of both.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_steps: int = 10,
        dt: float = 1.0,
        omega_max: float = 0.35,
        energy_penalty: float = 0.01,
        leakage_penalty: float = 1.0,
        bandwidth_tau: float = 0.5,
        detuning_std: float = 0.03,
        amp_error_std: float = 0.05,
        params: dict = None,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.dt = dt
        self.omega_max = omega_max
        self.energy_penalty = energy_penalty
        self.leakage_penalty = leakage_penalty
        self.bandwidth_tau = bandwidth_tau
        self.detuning_std = detuning_std
        self.amp_error_std = amp_error_std
        self.params = dict(params if params is not None else QUBIT_PARAMS)
        self.num_levels = self.params.get("num_levels", 3)

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Action: (Omega_x, Omega_y) normalised to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation: [P1, P2, Re01, Im01, Re12, Im12, Re02, Im02]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # c_ops don't depend on detuning, so build once; H0/Hx/Hy are
        # rebuilt each reset() since the detuning is randomized per episode.
        self.c_ops = build_collapse_operators(self.params, self.num_levels)
        self.H0 = self.Hx = self.Hy = None

        self.target_state = qt.basis(self.num_levels, 1)

        # Episode state
        self._state_dm: qt.Qobj = None
        self._step_count: int = 0
        self._amp_x_applied: float = 0.0   # filtered ("post-AWG") amplitude
        self._amp_y_applied: float = 0.0
        self._pulse_x: list = []
        self._pulse_y: list = []
        self._episode_detuning: float = 0.0
        self._episode_amp_gain: float = 1.0

        self._np_random_local = np.random.default_rng()

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random_local = np.random.default_rng(seed)

        # Sample this episode's hidden miscalibration -- unless the caller
        # supplies explicit values via options={'detuning':.., 'amp_gain':..},
        # which forces a specific (detuning, amp_gain) pair instead of
        # sampling one. Used for paired/ensemble evaluation, where every
        # controller must see the identical set of miscalibration draws.
        if options and "detuning" in options and "amp_gain" in options:
            detuning = options["detuning"]
            amp_gain = options["amp_gain"]
        else:
            detuning = (self._np_random_local.normal(0.0, self.detuning_std)
                        if self.detuning_std > 0 else 0.0)
            amp_gain = (1.0 + self._np_random_local.normal(0.0, self.amp_error_std)
                        if self.amp_error_std > 0 else 1.0)
        self._episode_detuning = float(detuning)
        self._episode_amp_gain = float(amp_gain)

        episode_params = dict(self.params)
        episode_params["omega_q"] = self.params.get("omega_q", 0.0) + detuning
        self.H0, self.Hx, self.Hy = build_hamiltonians(episode_params, self.num_levels)

        self._state_dm = qt.ket2dm(qt.basis(self.num_levels, 0))
        self._step_count = 0
        self._amp_x_applied = 0.0
        self._amp_y_applied = 0.0
        self._pulse_x = []
        self._pulse_y = []

        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        ax = float(np.clip(action[0], -1.0, 1.0))
        ay = float(np.clip(action[1], -1.0, 1.0))
        cmd_x = ax * self.omega_max
        cmd_y = ay * self.omega_max

        applied_x, applied_y = self._apply_bandwidth_filter(cmd_x, cmd_y)
        self._pulse_x.append(applied_x)
        self._pulse_y.append(applied_y)

        # Amplitude-calibration error: the qubit responds to a scaled
        # version of what the AWG actually outputs; the agent never sees
        # the scale factor directly.
        true_x = applied_x * self._episode_amp_gain
        true_y = applied_y * self._episode_amp_gain

        self._state_dm = self._propagate(self._state_dm, true_x, true_y, self.H0, self.Hx, self.Hy)
        self._step_count += 1
        terminated = self._step_count >= self.n_steps

        p2 = self._population(self._state_dm, 2)
        reward = -self.energy_penalty * (cmd_x ** 2 + cmd_y ** 2)
        reward -= self.leakage_penalty * p2

        info = {}
        if terminated:
            fidelity = compute_fidelity(self._state_dm, self.target_state)
            reward += fidelity
            info["fidelity"] = float(fidelity)
            info["leakage"] = float(p2)
            info["pulse_x"] = list(self._pulse_x)
            info["pulse_y"] = list(self._pulse_y)
            info["detuning"] = self._episode_detuning
            info["amp_gain"] = self._episode_amp_gain

        obs = self._get_obs()
        return obs.astype(np.float32), float(reward), terminated, False, info

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_bandwidth_filter(self, cmd_x: float, cmd_y: float):
        """First-order RC low-pass: the AWG output lags the command by
        `bandwidth_tau`, so a step change in `action` is not reproduced
        instantly in the physical drive seen by the qubit."""
        if self.bandwidth_tau > 0:
            alpha_f = 1.0 - np.exp(-self.dt / self.bandwidth_tau)
        else:
            alpha_f = 1.0
        self._amp_x_applied += alpha_f * (cmd_x - self._amp_x_applied)
        self._amp_y_applied += alpha_f * (cmd_y - self._amp_y_applied)
        return self._amp_x_applied, self._amp_y_applied

    def _propagate(self, dm, amp_x, amp_y, H0, Hx, Hy):
        H = [H0, [Hx, self._const(amp_x)], [Hy, self._const(amp_y)]]
        tlist = np.linspace(0, self.dt, 10)
        result = qt.mesolve(
            H, dm, tlist, self.c_ops,
            options=qt.Options(nsteps=5000, rtol=1e-8, atol=1e-10),
        )
        return result.states[-1]

    def _get_obs(self) -> np.ndarray:
        """Generalized Bloch vector: populations P1, P2 (P0 = 1-P1-P2 is
        implicit) plus the real/imaginary parts of the 01, 12, 02
        coherences -- the minimal sufficient statistic for a 3-level
        density matrix (8 real numbers for 8 = 3^2-1 degrees of freedom)."""
        rho = self._state_dm.full()
        p1 = rho[1, 1].real
        p2 = rho[2, 2].real if self.num_levels > 2 else 0.0
        re01, im01 = rho[0, 1].real, rho[0, 1].imag
        if self.num_levels > 2:
            re12, im12 = rho[1, 2].real, rho[1, 2].imag
            re02, im02 = rho[0, 2].real, rho[0, 2].imag
        else:
            re12 = im12 = re02 = im02 = 0.0
        return np.array([p1, p2, re01, im01, re12, im12, re02, im02], dtype=np.float64)

    def _population(self, dm, level: int) -> float:
        if level >= self.num_levels:
            return 0.0
        return float(dm.full()[level, level].real)

    @staticmethod
    def _const(value: float):
        def coeff(t, args=None):
            return value
        return coeff

    def get_full_trajectory(self, pulse_x, pulse_y, detuning: float = 0.0, amp_gain: float = 1.0):
        """
        Re-simulate an episode with given (already-bandwidth-filtered) I/Q
        amplitudes -- e.g. straight from info["pulse_x"]/info["pulse_y"] --
        and return (populations, bloch01), each an (n_steps+1, 3) array, for
        plotting. `detuning`/`amp_gain` let you replay under a specific
        (possibly non-nominal) episode condition.
        """
        episode_params = dict(self.params)
        episode_params["omega_q"] = self.params.get("omega_q", 0.0) + detuning
        H0, Hx, Hy = build_hamiltonians(episode_params, self.num_levels)

        dm = qt.ket2dm(qt.basis(self.num_levels, 0))
        pops = [self._dm_populations(dm)]
        bloch = [self._dm_bloch01(dm)]
        for ax, ay in zip(pulse_x, pulse_y):
            dm = self._propagate(dm, ax * amp_gain, ay * amp_gain, H0, Hx, Hy)
            pops.append(self._dm_populations(dm))
            bloch.append(self._dm_bloch01(dm))
        return np.array(pops), np.array(bloch)

    def _dm_populations(self, dm: qt.Qobj) -> np.ndarray:
        d = dm.full()
        return np.array([d[i, i].real for i in range(self.num_levels)])

    def _dm_bloch01(self, dm: qt.Qobj) -> np.ndarray:
        """0-1 subspace Bloch vector, matching <sigma_x>,<sigma_y>,<sigma_z>
        convention of the standard 2-level qubit model (verified sign-by-sign
        against qt.expect(sigmax/y/z(), .) in SCIENCE.md)."""
        d = dm.full()
        x = 2.0 * d[0, 1].real
        y = -2.0 * d[0, 1].imag
        z = (d[0, 0] - d[1, 1]).real
        return np.array([x, y, z])
