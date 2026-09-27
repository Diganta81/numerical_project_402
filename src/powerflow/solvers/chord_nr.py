from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..case import PowerCase
from ..results import PowerFlowResult
from ..ybus import build_ybus
from . import simplified_nr, standard_nr
from .base import IterationTracker, SolverOptions, initial_voltage, max_abs

METHODS = {
    "PNR": "Chord NR (current mismatch, paper formulation)",
    "SNR": "Chord NR (power mismatch)",
}


# ------------------------------------------------------------ LU helpers
def factorize(jac):
    """LU-factorise ``jac`` once; the result is reused by :func:`substitute`."""
    if sp.issparse(jac):
        return ("sparse", spla.splu(sp.csc_matrix(jac)))
    return ("dense", sla.lu_factor(np.asarray(jac), check_finite=False))


def substitute(factors, rhs: np.ndarray) -> np.ndarray:
    """Forward and back substitution with stored LU factors (no re-factorisation)."""
    kind, lu = factors
    if kind == "sparse":
        return lu.solve(rhs)
    return sla.lu_solve(lu, rhs, check_finite=False)


# ------------------------------------------------------------ residual / J
def _residual(case: PowerCase, ybus, v: np.ndarray, base: str):
    """Mismatch vector for the chosen formulation, plus PNR's effective schedule."""
    if base == "PNR":
        s_eff = simplified_nr.effective_schedule(case, ybus, v)   # lagged Q, eq. (22)
        return simplified_nr.mismatch_vector(case, ybus, v, s_eff), s_eff
    return standard_nr.mismatch_vector(case, ybus, v), None


def _jacobian(case: PowerCase, ybus, v: np.ndarray, base: str, s_eff, mode: str):
    if base == "PNR":
        return simplified_nr.jacobian(case, ybus, v, s_eff, mode)
    return standard_nr.jacobian(case, ybus, v, mode)


# ------------------------------------------------------------------ solver
def solve(
    case: PowerCase,
    options: Optional[SolverOptions] = None,
    ybus=None,
    base: str = "PNR",
    refresh_every: Optional[int] = None,
) -> PowerFlowResult:
    """Solve the power flow of ``case`` with the chord (fixed-Jacobian) method.

    Parameters
    ----------
    base
        ``"PNR"`` (the paper's current-mismatch equations) or ``"SNR"``
        (standard power-mismatch equations).
    refresh_every
        ``None`` for the pure chord method (one factorisation per solve), or an
        integer ``m`` to re-factorise every ``m`` iterations.
    """
    if base not in METHODS:
        raise ValueError(f"base must be one of {sorted(METHODS)}, got {base!r}")
    if refresh_every is not None and refresh_every < 1:
        raise ValueError("refresh_every must be a positive integer or None")
    options = (options or SolverOptions()).validate()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)

    pvpq, pq = case.pvpq, case.pq
    n_ang = len(pvpq)

    v = initial_voltage(case, options)
    vm, va = np.abs(v), np.angle(v)
    tracker = IterationTracker(options)
    factors = None
    n_factorizations = 0
    step = 0

    while True:
        f, s_eff = _residual(case, ybus, v, base)
        if tracker.begin_step(v, max_abs(f)):
            break

        if factors is None or (refresh_every is not None and step % refresh_every == 0):
            jac = _jacobian(case, ybus, v, base, s_eff, options.jacobian)
            factors = factorize(jac)
            n_factorizations += 1

        dx = substitute(factors, f)
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang:]
        v = vm * np.exp(1j * va)
        step += 1
        tracker.end_step(max_abs(dx))

    return tracker.finish(
        METHODS[base], case, v, n_factorizations=n_factorizations, base=base,
        refresh_every=refresh_every,
    )


def solve_pnr(case: PowerCase, options: Optional[SolverOptions] = None, ybus=None):
    """Chord method on the base paper's equations (the extension's main subject)."""
    return solve(case, options, ybus, base="PNR")


def solve_snr(case: PowerCase, options: Optional[SolverOptions] = None, ybus=None):
    """Chord method on the standard power-mismatch equations (for a fair comparison)."""
    return solve(case, options, ybus, base="SNR")
