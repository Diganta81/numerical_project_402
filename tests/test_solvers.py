"""
Solver-level tests.

The central property is that **every method must land on the same solution**:
they differ in the equations they iterate on, not in the answer.  That is what
makes the timing and iteration-count comparisons meaningful, so it is checked
for all seven solvers on every test system.
"""
from __future__ import annotations

import numpy as np
import pytest

from powerflow import build_ybus, load_case
from powerflow.solvers import SOLVERS, SolverOptions, standard_nr

TOL = 1e-10
NEWTON_KEYS = ["SNR", "PNR", "PNR+", "RCI", "FDLF-XB", "FDLF-BX"]


def residual_norm(case, v) -> float:
    """Largest power mismatch at the buses where it must vanish."""
    ybus = build_ybus(case)
    mismatch = case.s_sch - v * np.conj(ybus @ v)
    return max(
        np.max(np.abs(mismatch.real[case.pvpq])) if len(case.pvpq) else 0.0,
        np.max(np.abs(mismatch.imag[case.pq])) if len(case.pq) else 0.0,
    )


@pytest.mark.parametrize("key", NEWTON_KEYS)
def test_solver_satisfies_the_power_flow_equations(any_case, key):
    """Whatever it iterates on, a converged solver must zero the power mismatch."""
    res = SOLVERS[key](any_case, SolverOptions(tol=TOL, max_iter=80))
    assert res.converged, f"{key} failed on {any_case.name}: {res.message}"
    assert residual_norm(any_case, res.v) < 1e-7


@pytest.mark.parametrize("key", NEWTON_KEYS)
def test_solver_respects_voltage_setpoints(any_case, key):
    res = SOLVERS[key](any_case, SolverOptions(tol=TOL, max_iter=80))
    fixed = np.concatenate([any_case.slack, any_case.pv])
    assert np.allclose(np.abs(res.v)[fixed], any_case.vm_set[fixed], atol=1e-7)
    assert np.allclose(np.angle(res.v)[any_case.slack], any_case.va_set[any_case.slack])


def test_all_methods_agree(any_case):
    reference = standard_nr.solve(any_case, SolverOptions(tol=1e-12, max_iter=80))
    assert reference.converged
    for key in NEWTON_KEYS:
        res = SOLVERS[key](any_case, SolverOptions(tol=TOL, max_iter=80))
        assert np.max(np.abs(res.v - reference.v)) < 1e-6, f"{key} disagrees on {any_case.name}"


def test_gauss_seidel_agrees_on_small_systems(small_case):
    reference = standard_nr.solve(small_case, SolverOptions(tol=1e-12))
    res = SOLVERS["GS"](small_case, SolverOptions(tol=1e-8, max_iter=4000))
    assert res.converged
    assert np.max(np.abs(res.v - reference.v)) < 1e-5


def test_sparse_and_dense_solvers_agree(any_case):
    for key in ("SNR", "PNR", "PNR+", "RCI"):
        dense = SOLVERS[key](any_case, SolverOptions(tol=TOL, max_iter=80))
        sparse = SOLVERS[key](any_case, SolverOptions(tol=TOL, max_iter=80, sparse=True))
        assert sparse.converged
        assert np.max(np.abs(dense.v - sparse.v)) < 1e-8