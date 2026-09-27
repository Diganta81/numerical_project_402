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

METHOD = "Standard NR (power mismatch)"
SHORT = "SNR"


def power_mismatch(case: PowerCase, ybus, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_cal = s_injected(ybus, v)
    return case.p_sch - s_cal.real, case.q_sch - s_cal.imag


def mismatch_vector(case: PowerCase, ybus, v: np.ndarray) -> np.ndarray:
    dp, dq = power_mismatch(case, ybus, v)
    return np.concatenate([dp[case.pvpq], dq[case.pq]])


def derivative_matrices(case: PowerCase, ybus, v: np.ndarray, s_eff=None):
    ibus = ybus @ v
    vnorm = v / np.abs(v)
    if sp.issparse(ybus):
        d_v, d_i, d_vn = sp.diags(v), sp.diags(ibus), sp.diags(vnorm)
        ds_dvm = d_v @ (ybus @ d_vn).conjugate() + d_i.conjugate() @ d_vn
        ds_dva = 1j * (d_v @ (d_i - ybus @ d_v).conjugate())
    else:
        ds_dvm = v[:, None] * np.conj(ybus * vnorm[None, :]) + np.diag(np.conj(ibus) * vnorm)
        ds_dva = 1j * v[:, None] * np.conj(np.diag(ibus) - ybus * v[None, :])
    return ds_dva, ds_dvm


def jacobian(case: PowerCase, ybus, v: np.ndarray, mode: str = "vectorized"):
    if mode == "reference":
        return jacobian_reference(case, ybus, v)
    return jacobian_vectorized(case, ybus, v)


def jacobian_vectorized(case: PowerCase, ybus, v: np.ndarray):
    ds_dva, ds_dvm = derivative_matrices(case, ybus, v)
    return assemble_reduced(ds_dva, ds_dvm, case.pvpq, case.pq)


def jacobian_reference(case: PowerCase, ybus, v: np.ndarray) -> np.ndarray:
    y_mag, theta = ybus_polar(ybus)
    vm, va = np.abs(v), np.angle(v)
    n = case.n_bus
    diag = np.arange(n)

    alpha = theta + va[None, :] - va[:, None]
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    vv = vm[:, None] * vm[None, :] * y_mag    
    vk_y = vm[:, None] * y_mag                
    vi_y = vm[None, :] * y_mag                
    off = ~np.eye(n, dtype=bool)

    j1 = -vv * sin_a
    j2 = vk_y * cos_a
    j3 = -vv * cos_a
    j4 = -vk_y * sin_a

    j1[diag, diag] = np.sum(np.where(off, vv * sin_a, 0.0), axis=1)
    j2[diag, diag] = 2 * vm * y_mag[diag, diag] * np.cos(theta[diag, diag]) + np.sum(
        np.where(off, vi_y * cos_a, 0.0), axis=1
    )
    j3[diag, diag] = np.sum(np.where(off, vv * cos_a, 0.0), axis=1)
    j4[diag, diag] = -2 * vm * y_mag[diag, diag] * np.sin(theta[diag, diag]) - np.sum(
        np.where(off, vi_y * sin_a, 0.0), axis=1
    )

    pvpq, pq = case.pvpq, case.pq
    return np.block(
        [
            [j1[np.ix_(pvpq, pvpq)], j2[np.ix_(pvpq, pq)]],
            [j3[np.ix_(pq, pvpq)], j4[np.ix_(pq, pq)]],
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
    pvpq, pq = case.pvpq, case.pq
    n_ang = len(pvpq)

    v = initial_voltage(case, options)
    vm, va = np.abs(v), np.angle(v)
    tracker = IterationTracker(options)

    while True:
        f = mismatch_vector(case, ybus, v)
        if tracker.begin_step(v, max_abs(f)):
            break

        jac = jacobian(case, ybus, v, options.jacobian)
        dx = linear_solve(jac, f)

        # Equation (17): x^{h+1} = x^h + dx
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang:]
        v = vm * np.exp(1j * va)
        tracker.end_step(max_abs(dx))

    return tracker.finish(METHOD, case, v)
