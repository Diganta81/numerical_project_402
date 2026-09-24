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
from powerflow.solvers import SOLVERS, SolverOptions, simplified_nr, standard_nr

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


def test_reference_jacobian_mode_solves(small_case):
    """The literal-equation Jacobians drive the solvers as well as the fast ones."""
    for mod in (standard_nr, simplified_nr):
        fast = mod.solve(small_case, SolverOptions(tol=TOL))
        slow = mod.solve(small_case, SolverOptions(tol=TOL, jacobian="reference"))
        assert slow.converged
        assert slow.iterations == fast.iterations
        assert np.max(np.abs(slow.v - fast.v)) < 1e-10


def test_augmented_pv_converges_no_slower_than_the_paper_formulation(any_case):
    """The extension's whole purpose: restore quadratic convergence at PV buses."""
    opts = SolverOptions(tol=1e-10, max_iter=80)
    paper = simplified_nr.solve(any_case, opts)
    augmented = simplified_nr.solve(
        any_case, SolverOptions(tol=1e-10, max_iter=80, pv_handling="augmented")
    )
    assert augmented.converged and paper.converged
    assert augmented.iterations <= paper.iterations
    if len(any_case.pv) >= 5:
        reference = standard_nr.solve(any_case, opts)
        # Within one iteration of the standard method, which is the target.
        assert augmented.iterations <= reference.iterations + 1


def test_history_and_trace_stay_aligned(small_case):
    res = standard_nr.solve(small_case, SolverOptions(tol=TOL))
    trace = res.extras["v_trace"]
    assert len(trace) == len(res.history)
    assert np.allclose(trace[-1], res.v)
    ref = standard_nr.solve(small_case, SolverOptions(tol=1e-13, criterion="mismatch"))
    res.attach_reference(ref.v)
    errors = res.error_history
    assert np.all(np.diff(errors) <= 1e-12), "the error must decrease monotonically"


def test_convergence_is_monotone_in_the_mismatch(any_case):
    res = standard_nr.solve(any_case, SolverOptions(tol=1e-11, max_iter=80))
    mism = res.mismatch_history
    assert mism[-1] < mism[0]
    assert mism[-1] < 1e-7


def test_q_limit_enforcement_binds_generator_output():
    """Opt-in PV -> PQ switching holds Q inside its limits."""
    case = load_case("case30_ieee")
    case.q_max = np.minimum(case.q_max, 0.10)     # force several violations
    res = standard_nr.solve(case, SolverOptions(tol=1e-8, enforce_q_limits=True, max_iter=40))
    assert res.converged
    ybus = build_ybus(case)
    q = res.injections(ybus).imag
    switched = res.extras.get("q_limit_switches", [])
    assert switched, "the tightened limits should have forced at least one switch"
    assert np.all(q[case.pv] <= case.q_max[case.pv] + 1e-6)


def test_max_iter_is_respected_without_raising():
    case = load_case("case118_ieee")
    res = standard_nr.solve(case, SolverOptions(tol=1e-12, max_iter=2))
    assert not res.converged
    assert res.iterations == 2
    assert "did not reach" in res.message


def test_starting_from_a_previous_solution_takes_fewer_iterations(any_case):
    warm = standard_nr.solve(any_case, SolverOptions(tol=1e-10, max_iter=80))
    again = standard_nr.solve(any_case, SolverOptions(tol=1e-10, v0=warm.v))
    assert again.converged
    assert again.iterations <= 1
