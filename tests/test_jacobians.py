from __future__ import annotations

import numpy as np

from powerflow import build_ybus
from powerflow.solvers import simplified_nr, standard_nr


def perturbed_states(case, v, eps: float):
    pvpq, pq = case.pvpq, case.pq
    vm, va = np.abs(v), np.angle(v)
    for j, k in enumerate(pvpq):                 # angle unknowns
        for sign in (+1, -1):
            a = va.copy()
            a[k] += sign * eps
            yield j, sign, vm * np.exp(1j * a)
    for j, k in enumerate(pq):                   # magnitude unknowns
        for sign in (+1, -1):
            m = vm.copy()
            m[k] += sign * eps
            yield len(pvpq) + j, sign, m * np.exp(1j * va)


def numeric_jacobian(case, ybus, v, residual, eps: float = 1e-6) -> np.ndarray:
    n_x = len(case.pvpq) + len(case.pq)
    jac = np.zeros((len(residual(v)), n_x))
    for j, sign, v_pert in perturbed_states(case, v, eps):
        jac[:, j] += sign * residual(v_pert) / (2 * eps)
    return jac


def test_standard_reference_matches_vectorized(small_case):
    ybus = build_ybus(small_case)
    v = small_case.flat_start() * np.exp(1j * 0.05)      
    fast = standard_nr.jacobian_vectorized(small_case, ybus, v)
    ref = standard_nr.jacobian_reference(small_case, ybus, v)
    assert np.allclose(fast, ref, rtol=1e-10, atol=1e-9 * max(1.0, np.max(np.abs(ref))))


def test_simplified_reference_matches_vectorized(small_case):
    ybus = build_ybus(small_case)
    v = small_case.flat_start() * np.exp(1j * 0.05)
    s_eff = simplified_nr.effective_schedule(small_case, ybus, v)
    fast = simplified_nr.jacobian_vectorized(small_case, ybus, v, s_eff)
    ref = simplified_nr.jacobian_reference(small_case, ybus, v, s_eff)
    assert np.allclose(fast, ref, rtol=1e-10, atol=1e-9 * max(1.0, np.max(np.abs(ref))))


def test_standard_jacobian_matches_finite_differences(small_case):
    case, ybus = small_case, build_ybus(small_case)
    v = case.flat_start() * np.exp(1j * 0.03)

    def calculated_power(v_now):
        s = v_now * np.conj(ybus @ v_now)
        return np.concatenate([s.real[case.pvpq], s.imag[case.pq]])

    analytic = standard_nr.jacobian(case, ybus, v)
    numeric = numeric_jacobian(case, ybus, v, calculated_power)
    scale = max(1.0, np.max(np.abs(analytic)))
    assert np.allclose(analytic, numeric, atol=1e-4 * scale)


def test_simplified_jacobian_matches_finite_differences(small_case):
    case, ybus = small_case, build_ybus(small_case)
    v = case.flat_start() * np.exp(1j * 0.03)
    s_eff = simplified_nr.effective_schedule(case, ybus, v)

    def residual(v_now):
        return -simplified_nr.mismatch_vector(case, ybus, v_now, s_eff)

    analytic = simplified_nr.jacobian(case, ybus, v, s_eff)
    numeric = numeric_jacobian(case, ybus, v, residual)
    scale = max(1.0, np.max(np.abs(analytic)))
    assert np.allclose(analytic, numeric, atol=1e-4 * scale)


def test_sparse_matches_dense(small_case):
    case = small_case
    v = case.flat_start() * np.exp(1j * 0.02)
    dense_y = build_ybus(case, sparse=False)
    sparse_y = build_ybus(case, sparse=True)

    for mod, args in ((standard_nr, ()), (simplified_nr, (None,))):
        dense = mod.jacobian(case, dense_y, v, *args)
        sparse = mod.jacobian(case, sparse_y, v, *args)
        assert np.allclose(dense, sparse.toarray(), atol=1e-9 * max(1.0, np.max(np.abs(dense))))
