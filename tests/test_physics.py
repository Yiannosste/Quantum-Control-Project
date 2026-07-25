"""
tests/test_physics.py
----------------------
Regression tests against a real, previously-shipped bug: the T2 collapse
operator carried a spurious /2 (sqrt(2*Gamma_phi)*n/2 instead of
sqrt(2*Gamma_phi)*n), which simulated T2 as 2.29x longer than the value in
QUBIT_PARAMS (1142.9 ns measured vs. a labeled 500 ns) -- i.e. every result
in this repository was generated under ~2.3x too little dephasing. See
utils.py's module docstring ("Note on the T2 collapse-operator prefactor")
for the derivation.

These tests measure T1 and T2 directly from build_collapse_operators()'s
actual output via free-decay simulation, and assert they match
QUBIT_PARAMS to a tight tolerance -- so a regression like this fails CI
instead of silently shipping in every downstream fidelity number.
"""

import numpy as np
import pytest
import qutip as qt

from utils import build_collapse_operators, QUBIT_PARAMS


def _measure_t1(params, tlist_max=4000.0, n_points=4000):
    c_ops = build_collapse_operators(params)
    num_levels = params.get("num_levels", 3)
    n = qt.destroy(num_levels).dag() * qt.destroy(num_levels)
    psi1 = qt.ket2dm(qt.basis(num_levels, 1))
    tlist = np.linspace(0, tlist_max, n_points)
    result = qt.mesolve(qt.Qobj(np.zeros((num_levels, num_levels))), psi1, tlist, c_ops, [n])
    pop1 = result.expect[0]
    mask = pop1 > 1e-3
    slope, _ = np.polyfit(tlist[mask], np.log(pop1[mask]), 1)
    return -1.0 / slope


def _measure_t2(params, tlist_max=800.0, n_points=4000):
    c_ops = build_collapse_operators(params)
    num_levels = params.get("num_levels", 3)
    psi_plus = (qt.basis(num_levels, 0) + qt.basis(num_levels, 1)).unit()
    sx01 = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])[:num_levels, :num_levels])
    sy01 = qt.Qobj(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])[:num_levels, :num_levels])
    tlist = np.linspace(0, tlist_max, n_points)
    result = qt.mesolve(qt.Qobj(np.zeros((num_levels, num_levels))), psi_plus, tlist, c_ops, [sx01, sy01])
    coherence = np.abs(result.expect[0] + 1j * result.expect[1])
    mask = coherence > 1e-6
    slope, _ = np.polyfit(tlist[mask], np.log(coherence[mask]), 1)
    return -1.0 / slope


def test_t1_matches_qubit_params():
    measured = _measure_t1(QUBIT_PARAMS)
    assert measured == pytest.approx(QUBIT_PARAMS["T1"], rel=0.01)


def test_t2_matches_qubit_params():
    measured = _measure_t2(QUBIT_PARAMS)
    assert measured == pytest.approx(QUBIT_PARAMS["T2"], rel=0.01)


def test_t2_not_the_known_bad_prefactor():
    """The specific regression this file exists to catch: sqrt(2*Gamma_phi)*n/2
    (spurious /2) simulates T2 ~2.29x too long. Fail loudly, not just
    approximately, if that factor of ~2.29 reappears."""
    measured = _measure_t2(QUBIT_PARAMS)
    wrong_factor = measured / QUBIT_PARAMS["T2"]
    assert wrong_factor < 1.5, (
        f"Measured T2 ({measured:.1f} ns) is {wrong_factor:.2f}x the labeled "
        f"T2 ({QUBIT_PARAMS['T2']} ns) -- this is the signature of the spurious "
        f"/2 in the T2 collapse operator (see utils.py's module docstring)."
    )
