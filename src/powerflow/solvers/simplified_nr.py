from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..case import PowerCase
from ..results import PowerFlowResult
from ..ybus import build_ybus, ybus_polar
from .base import (
    IterationTracker,
    assemble_reduced,
    SolverOptions,
    initial_voltage,
    linear_solve,
    max_abs,
    s_injected,
    solve_with_q_limits,
)

METHOD = "Simplified NR (current mismatch)"
SHORT = "PNR"

def effective_schedule(case: PowerCase, ybus, v: np.ndarray) -> np.ndarray:
    s = case.s_sch.copy()
    pv = case.pv
    if len(pv):
        q_cal = s_injected(ybus, v).imag
        s[pv] = case.p_sch[pv] + 1j * q_cal[pv]
    return s


def current_mismatch(case: PowerCase, ybus, v: np.ndarray, s_eff=None) -> Tuple[np.ndarray, np.ndarray]:
    if s_eff is None:
        s_eff = effective_schedule(case, ybus, v)
    f = np.conj(s_eff / v) - (ybus @ v)
    return f.real, f.imag


def mismatch_vector(case: PowerCase, ybus, v: np.ndarray, s_eff=None) -> np.ndarray:
    g, h = current_mismatch(case, ybus, v, s_eff)
    return np.concatenate([g[case.pvpq], h[case.pq]])


def jacobian(case: PowerCase, ybus, v: np.ndarray, s_eff=None, mode: str = "vectorized"):
    if mode == "reference":
        return jacobian_reference(case, ybus, v, s_eff)
    return jacobian_vectorized(case, ybus, v, s_eff)


def derivative_matrices(case: PowerCase, ybus, v: np.ndarray, s_eff=None):
    if s_eff is None:
        s_eff = effective_schedule(case, ybus, v)
    vm = np.abs(v)
    a = np.conj(s_eff / v)
    e_jd = v / vm
    if sp.issparse(ybus):
        j_delta = 1j * (ybus @ sp.diags(v)) - sp.diags(1j * a)
        j_vm = ybus @ sp.diags(e_jd) + sp.diags(a / vm)
    else:
        j_delta = 1j * ybus * v[None, :] - np.diag(1j * a)
        j_vm = ybus * e_jd[None, :] + np.diag(a / vm)
    return j_delta, j_vm


def jacobian_vectorized(case: PowerCase, ybus, v: np.ndarray, s_eff=None):
    j_delta, j_vm = derivative_matrices(case, ybus, v, s_eff)
    return assemble_reduced(j_delta, j_vm, case.pvpq, case.pq)


def jacobian_blocks_reference(case: PowerCase, ybus, v: np.ndarray, s_eff=None):
    if s_eff is None:
        s_eff = effective_schedule(case, ybus, v)
    y_mag, theta = ybus_polar(ybus)
    vm, va = np.abs(v), np.angle(v)
    n = case.n_bus
    diag = np.arange(n)

    beta = theta + va[None, :]              # theta_ki + delta_i  (no -delta_k!)
    sin_b, cos_b = np.sin(beta), np.cos(beta)

    a = np.abs(s_eff) / vm                  # |S_sch,k / V_k|
    gamma = va - np.angle(s_eff)            # delta_k - phi_k
    sin_g, cos_g = np.sin(gamma), np.cos(gamma)

    j1 = -vm[None, :] * y_mag * sin_b
    j2 = y_mag * cos_b
    j3 = vm[None, :] * y_mag * cos_b
    j4 = y_mag * sin_b

    j1[diag, diag] += a * sin_g
    j2[diag, diag] += (a / vm) * cos_g
    j3[diag, diag] -= a * cos_g
    j4[diag, diag] += (a / vm) * sin_g
    return j1, j2, j3, j4


def jacobian_reference(case: PowerCase, ybus, v: np.ndarray, s_eff=None) -> np.ndarray:
    j1, j2, j3, j4 = jacobian_blocks_reference(case, ybus, v, s_eff)
    pvpq, pq = case.pvpq, case.pq
    return np.block(
        [
            [j1[np.ix_(pvpq, pvpq)], j2[np.ix_(pvpq, pq)]],
            [j3[np.ix_(pq, pvpq)], j4[np.ix_(pq, pq)]],
        ]
    )


def jacobian_augmented(case: PowerCase, ybus, v: np.ndarray, s_eff):
    pvpq, pq, pv = case.pvpq, case.pq, case.pv
    j_delta, j_vm = derivative_matrices(case, ybus, v, s_eff)

    if sp.issparse(ybus):
        j_q = sp.csr_matrix(
            (1j / np.conj(v[pv]), (pv, pv)), shape=(case.n_bus, case.n_bus), dtype=complex
        )
        rows_g = [j_delta[pvpq, :][:, pvpq].real, j_vm[pvpq, :][:, pq].real, j_q[pvpq, :][:, pv].real]
        rows_h = [j_delta[pvpq, :][:, pvpq].imag, j_vm[pvpq, :][:, pq].imag, j_q[pvpq, :][:, pv].imag]
        return sp.bmat([rows_g, rows_h], format="csr")

    j_q = np.zeros((case.n_bus, case.n_bus), dtype=complex)
    j_q[pv, pv] = 1j / np.conj(v[pv])
    return np.block(
        [
            [
                j_delta[np.ix_(pvpq, pvpq)].real,
                j_vm[np.ix_(pvpq, pq)].real,
                j_q[np.ix_(pvpq, pv)].real,
            ],
            [
                j_delta[np.ix_(pvpq, pvpq)].imag,
                j_vm[np.ix_(pvpq, pq)].imag,
                j_q[np.ix_(pvpq, pv)].imag,
            ],
        ]
    )


def solve(case: PowerCase, options: Optional[SolverOptions] = None, ybus=None) -> PowerFlowResult:
    options = (options or SolverOptions()).validate()
    if options.enforce_q_limits:
        from dataclasses import replace

        return solve_with_q_limits(
            lambda c, o: solve(c, o), case, replace(options, enforce_q_limits=True)
        )
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)
    if options.pv_handling == "augmented":
        return _solve_augmented(case, options, ybus)
    return _solve_paper(case, options, ybus)


def _solve_paper(case: PowerCase, options: SolverOptions, ybus) -> PowerFlowResult:
    pvpq, pq = case.pvpq, case.pq
    n_ang = len(pvpq)

    v = initial_voltage(case, options)
    vm, va = np.abs(v), np.angle(v)
    tracker = IterationTracker(options)

    while True:
        s_eff = effective_schedule(case, ybus, v)          # Q at PV buses, eq. (22)
        f = mismatch_vector(case, ybus, v, s_eff)          # eqs. (4), (5)
        if tracker.begin_step(v, max_abs(f)):
            break

        jac = jacobian(case, ybus, v, s_eff, options.jacobian)   # eqs. (9)-(16)
        dx = linear_solve(jac, f)                                # eq. (8)

        # Equation (17): x^{h+1} = x^h + dx
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang:]
        v = vm * np.exp(1j * va)
        tracker.end_step(max_abs(dx))

    return tracker.finish(METHOD, case, v)


def _solve_augmented(case: PowerCase, options: SolverOptions, ybus) -> PowerFlowResult:
    pvpq, pq, pv = case.pvpq, case.pq, case.pv
    n_ang, n_vm = len(pvpq), len(pq)

    v = initial_voltage(case, options)
    vm, va = np.abs(v), np.angle(v)
    q = case.q_sch.copy()
    if len(pv):                                   # first estimate of the PV Q's
        q[pv] = s_injected(ybus, v).imag[pv]
    tracker = IterationTracker(options)

    while True:
        s_eff = case.p_sch + 1j * q
        g, h = current_mismatch(case, ybus, v, s_eff)
        f = np.concatenate([g[pvpq], h[pvpq]])
        if tracker.begin_step(v, max_abs(f)):
            break

        dx = linear_solve(jacobian_augmented(case, ybus, v, s_eff), f)
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang : n_ang + n_vm]
        q[pv] += dx[n_ang + n_vm :]
        v = vm * np.exp(1j * va)
        tracker.end_step(max_abs(dx[: n_ang + n_vm]))

    return tracker.finish(
        METHOD + " [augmented PV]", case, v, q_pv=q[pv], pv_handling="augmented"
    )
