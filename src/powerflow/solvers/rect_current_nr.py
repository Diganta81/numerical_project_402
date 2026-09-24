r"""
Rectangular current-injection Newton-Raphson (RCI).

Project extension.  Slide 4 of the project proposal writes the nodal current
injection in rectangular components,

.. math::

    I_i = I_i^r + j I_i^m = \sum_{k=1}^{N} (G_{ik} + j B_{ik})(e_k + j f_k),

and forms the mismatch from ``(dI^r, dI^m)``.  That is *not* what the base paper
does -- the paper keeps the polar state ``(|V|, delta)`` and only splits the
*mismatch* into real and imaginary parts -- so this module implements the
proposal's formulation as a separate solver, and the comparison between the two
is one of the results of the project.  The formulation is the classical current
injection method of da Costa, Pereira & Martins, which is reference [26] of the
base paper.

**State and equations.**  The unknowns are ``(e_k, f_k)`` at every non-slack
bus, plus ``Q_k`` at every PV bus.  The residuals are

.. math::

    R^r_k = \frac{P_k e_k + Q_k f_k}{e_k^2+f_k^2} - \sum_i (G_{ki}e_i - B_{ki}f_i)

    R^m_k = \frac{P_k f_k - Q_k e_k}{e_k^2+f_k^2} - \sum_i (B_{ki}e_i + G_{ki}f_i)

for every non-slack bus, closed at PV buses by the magnitude constraint
``R^V_k = |V_k^{sch}|^2 - (e_k^2 + f_k^2)``.

**Why it is attractive.**  The network part of the Jacobian is literally
``+/- G`` and ``+/- B`` taken straight from ``Y_bus``: it is *constant*.  Only
the ``2x2`` diagonal blocks (and the PV columns) change from iteration to
iteration, so an iteration updates ``O(n)`` numbers instead of rebuilding the
whole matrix -- the same structural win the base paper is after, obtained a
different way.  The trade-off is a system of size ``2(n-1) + n_{pv}`` rather
than the paper's ``(n-1) + n_{pq}``.

Unlike the other Newton solvers here this one uses the ordinary sign
convention: ``J = dR/dx`` and ``x <- x - J^{-1} R``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..case import PowerCase
from ..results import PowerFlowResult
from ..ybus import build_ybus
from .base import IterationTracker, SolverOptions, initial_voltage, linear_solve, max_abs, s_injected

METHOD = "Rectangular current injection"
SHORT = "RCI"


def residuals(case: PowerCase, ybus, v: np.ndarray, q: np.ndarray):
    """Current-injection residuals ``(R^r, R^m)`` (full length) and ``R^V`` (PV only)."""
    s = case.p_sch + 1j * q
    i_sch = np.conj(s / v)                 # (P e + Q f)/|V|^2 + j (P f - Q e)/|V|^2
    i_cal = ybus @ v
    r = i_sch - i_cal
    r_v = case.vm_set[case.pv] ** 2 - np.abs(v[case.pv]) ** 2
    return r.real, r.imag, r_v


def jacobian(case: PowerCase, ybus, v: np.ndarray, q: np.ndarray):
    """Assemble ``dR/dx`` with rows ``[R^r; R^m; R^V]`` and columns ``[de; df; dQ]``."""
    ns, pv = case.pvpq, case.pv
    n_ns, n_pv = len(ns), len(pv)

    g = ybus.real
    b = ybus.imag
    dense = not sp.issparse(ybus)

    e, f = v.real, v.imag
    p = case.p_sch
    v2 = e * e + f * f
    v4 = v2 * v2
    e2mf2 = e * e - f * f

    # Diagonal contributions of the scheduled-current term.
    dIr_de = (-p * e2mf2 - 2.0 * e * q * f) / v4
    dIr_df = (q * e2mf2 - 2.0 * e * p * f) / v4
    dIm_de = dIr_df                      # equal by the Cauchy-Riemann structure
    dIm_df = (p * e2mf2 + 2.0 * q * e * f) / v4

    def sub(mat, rows, cols):
        return mat[rows, :][:, cols] if sp.issparse(mat) else mat[np.ix_(rows, cols)]

    def diag_block(base, add):
        """Network block ``base`` plus a diagonal correction, restricted to ns."""
        blk = sub(base, ns, ns)
        d = sp.diags(add[ns]) if sp.issparse(base) else np.diag(add[ns])
        return blk + d

    j_re = diag_block(-g, dIr_de)        # dR^r/de
    j_rf = diag_block(b, dIr_df)         # dR^r/df
    j_me = diag_block(-b, dIm_de)        # dR^m/de
    j_mf = diag_block(-g, dIm_df)        # dR^m/df

    # --- PV columns: dR/dQ_k = ( f_k/|V_k|^2 , -e_k/|V_k|^2 ) ---
    pos = {int(k): i for i, k in enumerate(ns)}
    rows_pv = np.array([pos[int(k)] for k in pv], dtype=int)
    j_rq = np.zeros((n_ns, n_pv))
    j_mq = np.zeros((n_ns, n_pv))
    if n_pv:
        j_rq[rows_pv, np.arange(n_pv)] = f[pv] / v2[pv]
        j_mq[rows_pv, np.arange(n_pv)] = -e[pv] / v2[pv]

    # --- PV rows: dR^V/de = -2e, dR^V/df = -2f ---
    j_ve = np.zeros((n_pv, n_ns))
    j_vf = np.zeros((n_pv, n_ns))
    if n_pv:
        j_ve[np.arange(n_pv), rows_pv] = -2.0 * e[pv]
        j_vf[np.arange(n_pv), rows_pv] = -2.0 * f[pv]
    j_vq = np.zeros((n_pv, n_pv))

    if dense:
        return np.block(
            [
                [np.asarray(j_re), np.asarray(j_rf), j_rq],
                [np.asarray(j_me), np.asarray(j_mf), j_mq],
                [j_ve, j_vf, j_vq],
            ]
        )
    return sp.bmat(
        [
            [j_re, j_rf, sp.csr_matrix(j_rq)],
            [j_me, j_mf, sp.csr_matrix(j_mq)],
            [sp.csr_matrix(j_ve), sp.csr_matrix(j_vf), sp.csr_matrix(j_vq)],
        ],
        format="csr",
    )


def solve(case: PowerCase, options: Optional[SolverOptions] = None, ybus=None) -> PowerFlowResult:
    """Solve the power flow of ``case`` by the rectangular current-injection method."""
    options = (options or SolverOptions()).validate()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)

    ns, pv = case.pvpq, case.pv
    n_ns = len(ns)

    v = initial_voltage(case, options)
    q = case.q_sch.copy()
    if len(pv):
        q[pv] = s_injected(ybus, v).imag[pv]
    tracker = IterationTracker(options)

    while True:
        r_re, r_im, r_v = residuals(case, ybus, v, q)
        f = np.concatenate([r_re[ns], r_im[ns], r_v])
        if tracker.begin_step(v, max_abs(f)):
            break

        dx = linear_solve(jacobian(case, ybus, v, q), -f)     # J dx = -R
        e_new = v.real.copy()
        f_new = v.imag.copy()
        e_new[ns] += dx[:n_ns]
        f_new[ns] += dx[n_ns : 2 * n_ns]
        q[pv] += dx[2 * n_ns :]
        v_prev = v
        v = e_new + 1j * f_new
        tracker.end_step(max_abs(v - v_prev))

    return tracker.finish(METHOD, case, v, q_pv=q[pv])
