"""Y-bus construction, case loading and branch-flow bookkeeping."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from powerflow import PQ, PV, REF, available_cases, build_ybus, load_case
from powerflow.ybus import branch_flows, sparsity


def test_every_shipped_case_loads():
    names = available_cases()
    assert len(names) >= 9
    for name in names:
        case = load_case(name)
        assert case.n_bus > 0
        assert len(case.slack) == 1, f"{name} must have exactly one slack bus"
        assert set(case.bus_type) <= {PQ, PV, REF}


def test_index_partitions_are_disjoint_and_complete(any_case):
    case = any_case
    parts = [case.slack, case.pv, case.pq]
    combined = np.concatenate(parts)
    assert len(combined) == case.n_bus
    assert len(set(combined.tolist())) == case.n_bus
    assert np.array_equal(case.pvpq, np.sort(np.concatenate([case.pv, case.pq])))


def test_ybus_is_symmetric_without_phase_shifters(any_case):
    case = any_case
    if np.any(case.shift != 0):
        pytest.skip("phase shifters make Y-bus asymmetric by design")
    y = build_ybus(case)
    assert np.allclose(y, y.T, atol=1e-12)


def test_ybus_row_sums_equal_shunts_for_untapped_lines():
    """With no taps, shifts or charging, every row of Y sums to its bus shunt."""
    case = load_case("case3_saadat")
    y = build_ybus(case)
    assert np.allclose(y.sum(axis=1), case.y_shunt, atol=1e-10)


def test_sparse_and_dense_ybus_agree(any_case):
    dense = build_ybus(any_case, sparse=False)
    sparse = build_ybus(any_case, sparse=True)
    assert sp.issparse(sparse)
    assert np.allclose(dense, sparse.toarray(), atol=1e-12)


def test_sparsity_statistics(any_case):
    stats = sparsity(build_ybus(any_case, sparse=True))
    assert stats["n"] == any_case.n_bus
    assert 0 < stats["density"] <= 1.0
    assert stats["nnz"] >= any_case.n_bus


def test_branch_flows_conserve_power(any_case):
    """Sum of branch flows into a bus equals its net injection."""
    case = any_case
    y = build_ybus(case)
    rng = np.random.default_rng(0)
    v = case.flat_start() * np.exp(1j * rng.normal(0, 0.02, case.n_bus))

    s_from, s_to = branch_flows(case, v)
    injected = np.zeros(case.n_bus, dtype=complex)
    np.add.at(injected, case.f_bus, s_from)
    np.add.at(injected, case.t_bus, s_to)
    injected += v * np.conj(case.y_shunt * v)

    expected = v * np.conj(y @ v)
    assert np.allclose(injected, expected, atol=1e-10)


def test_flat_start_respects_setpoints(any_case):
    case = any_case
    v = case.flat_start()
    fixed = np.concatenate([case.slack, case.pv])
    assert np.allclose(np.abs(v)[fixed], case.vm_set[fixed])
    assert np.allclose(np.abs(v)[case.pq], 1.0)
    assert np.allclose(np.angle(v), np.angle(v)[case.slack[0]])


def test_scaled_case_scales_injections():
    case = load_case("case30_ieee")
    doubled = case.scaled(2.0)
    load = case.p_sch < 0
    assert np.allclose(doubled.p_sch[load], 2 * case.p_sch[load])
    assert np.allclose(doubled.q_sch, 2 * case.q_sch)
    assert case.p_sch is not doubled.p_sch


def test_tap_and_shift_are_honoured():
    """A 1:t transformer must scale the off-diagonal admittance by 1/t."""
    case = load_case("case57_ieee")
    tapped = np.flatnonzero(case.tap != 1.0)
    assert len(tapped) > 0, "case57 is expected to contain off-nominal taps"
    y = build_ybus(case)
    br = tapped[0]
    f, t = case.f_bus[br], case.t_bus[br]
    ys = 1.0 / (case.br_r[br] + 1j * case.br_x[br])
    parallel = np.sum((case.f_bus == f) & (case.t_bus == t)) + np.sum(
        (case.f_bus == t) & (case.t_bus == f)
    )
    if parallel == 1:
        assert y[f, t] == pytest.approx(-ys / case.tap[br], rel=1e-12)
