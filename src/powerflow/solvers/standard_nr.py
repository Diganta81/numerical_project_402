r"""
Standard Newton-Raphson power flow -- the *power mismatch* formulation (SNR).

This is the benchmark the base paper measures against.  The unknowns are the
voltage angles of every non-slack bus and the voltage magnitudes of every PQ
bus; the residuals are the real and reactive power mismatches

.. math::

    \Delta P_k = P_{\mathrm{sch},k} - P_{\mathrm{cal},k}, \qquad
    \Delta Q_k = Q_{\mathrm{sch},k} - Q_{\mathrm{cal},k}

with (base paper, equations 21 and 22)

.. math::

    P_{\mathrm{cal},k} = \sum_i |V_k V_i Y_{ki}| \cos(\theta_{ki}+\delta_i-\delta_k)

    Q_{\mathrm{cal},k} = -\sum_i |V_k V_i Y_{ki}| \sin(\theta_{ki}+\delta_i-\delta_k)

The update equation is the paper's equation (18),

.. math::

    \begin{bmatrix}\Delta P\\ \Delta Q\end{bmatrix} =
    \begin{bmatrix}J_1 & J_2\\ J_3 & J_4\end{bmatrix}
    \begin{bmatrix}\Delta\delta\\ \Delta|V|\end{bmatrix}

followed by ``x <- x + dx`` (equation 17).  Note that ``J_2`` and ``J_4`` hold
the *unnormalised* derivatives with respect to the magnitude (not
``|V| d/d|V|``), matching the paper's equations and therefore its
floating-point operation counts in Table 1.

Two Jacobian builders are provided and are numerically identical:

``reference``
    A literal transcription of the textbook element formulas -- equations
    (19) and (20) of the paper and their J2/J3/J4 counterparts.
``vectorized``
    The equivalent complex-analytic form dS/ddelta, dS/d|V|.  This is what the
    benchmarks run, and it is the only one with a sparse path.

``tests/test_jacobians.py`` asserts that the two agree to machine precision.
"""
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


# --------------------------------------------------------------- residuals
def power_mismatch(case: PowerCase, ybus, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Full-length real and reactive power mismatch vectors, p.u.

    Implements equations (21) and (22) through ``S = V . conj(Y V)``.
    """
    s_cal = s_injected(ybus, v)
    return case.p_sch - s_cal.real, case.q_sch - s_cal.imag


def mismatch_vector(case: PowerCase, ybus, v: np.ndarray) -> np.ndarray:
    """Right-hand side of equation (18): ``[dP over pv+pq; dQ over pq]``."""
    dp, dq = power_mismatch(case, ybus, v)
    return np.concatenate([dp[case.pvpq], dq[case.pq]])


# --------------------------------------------------------------- Jacobians
def derivative_matrices(case: PowerCase, ybus, v: np.ndarray, s_eff=None):
    """Return ``(dS/ddelta, dS/d|V|)`` as full complex matrices.

    Obtained by differentiating ``S = V . conj(Y V)``; the real and imaginary
    parts supply the four sub-matrices of equation (18).  ``case`` and ``s_eff``
    are unused and present only so that this has the same signature as
    :func:`powerflow.solvers.simplified_nr.derivative_matrices`, which lets the
    benchmark time the two head to head.

    This is the step the paper's FLOP analysis is about; the subsequent slicing
    (:func:`powerflow.solvers.base.assemble_reduced`) is identical for both
    methods.
    """
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
    """Assemble the Jacobian of equation (18)."""
    if mode == "reference":
        return jacobian_reference(case, ybus, v)
    return jacobian_vectorized(case, ybus, v)


def jacobian_vectorized(case: PowerCase, ybus, v: np.ndarray):
    ds_dva, ds_dvm = derivative_matrices(case, ybus, v)
    return assemble_reduced(ds_dva, ds_dvm, case.pvpq, case.pq)


def jacobian_reference(case: PowerCase, ybus, v: np.ndarray) -> np.ndarray:
    r"""Literal element-by-element Jacobian (dense only).

    With ``alpha_ki = theta_ki + delta_i - delta_k``, the off-diagonal entries
    (paper equation 19 and its siblings), for ``k != i``, are

    .. math::

        \partial P_k/\partial\delta_i = -|V_kV_iY_{ki}|\sin\alpha_{ki}

        \partial P_k/\partial|V_i|    =  |V_kY_{ki}|\cos\alpha_{ki}

        \partial Q_k/\partial\delta_i = -|V_kV_iY_{ki}|\cos\alpha_{ki}

        \partial Q_k/\partial|V_i|    = -|V_kY_{ki}|\sin\alpha_{ki}

    and the diagonal entries (paper equation 20 and its siblings) are

    .. math::

        \partial P_k/\partial\delta_k = \sum_{i\ne k}|V_kV_iY_{ki}|\sin\alpha_{ki}

        \partial P_k/\partial|V_k| = 2|V_kY_{kk}|\cos\theta_{kk}
                                     + \sum_{i\ne k}|V_iY_{ki}|\cos\alpha_{ki}

        \partial Q_k/\partial\delta_k = \sum_{i\ne k}|V_kV_iY_{ki}|\cos\alpha_{ki}

        \partial Q_k/\partial|V_k| = -2|V_kY_{kk}|\sin\theta_{kk}
                                     - \sum_{i\ne k}|V_iY_{ki}|\sin\alpha_{ki}
    """
    y_mag, theta = ybus_polar(ybus)
    vm, va = np.abs(v), np.angle(v)
    n = case.n_bus
    diag = np.arange(n)

    alpha = theta + va[None, :] - va[:, None]
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    vv = vm[:, None] * vm[None, :] * y_mag    # |V_k V_i Y_ki|
    vk_y = vm[:, None] * y_mag                # |V_k Y_ki|
    vi_y = vm[None, :] * y_mag                # |V_i Y_ki|
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


# ------------------------------------------------------------------- solver
def solve(case: PowerCase, options: Optional[SolverOptions] = None, ybus=None) -> PowerFlowResult:
    """Solve the power flow of ``case`` with the standard NR method."""
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
