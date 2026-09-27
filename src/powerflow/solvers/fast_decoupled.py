from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from ..case import PowerCase
from ..results import PowerFlowResult
from ..ybus import build_ybus
from .base import IterationTracker, SolverOptions, initial_voltage, max_abs, s_injected

METHOD = "Fast Decoupled (XB)"
SHORT = "FDLF"


def make_b_matrices(case: PowerCase, variant: str = "XB", sparse: bool = False) -> Tuple:
    variant = variant.upper()
    if variant not in ("XB", "BX"):
        raise ValueError(f"unknown FDLF variant {variant!r}")

    # ---- B' ----
    cp = copy.deepcopy(case)
    cp.br_b = np.zeros_like(cp.br_b)
    cp.tap = np.ones_like(cp.tap)
    cp.y_shunt = np.zeros_like(cp.y_shunt)
    if variant == "XB":
        cp.br_r = np.zeros_like(cp.br_r)
    b_prime = -build_ybus(cp, sparse=sparse).imag

    # ---- B'' ----
    cpp = copy.deepcopy(case)
    cpp.shift = np.zeros_like(cpp.shift)
    if variant == "BX":
        cpp.br_r = np.zeros_like(cpp.br_r)
    b_dprime = -build_ybus(cpp, sparse=sparse).imag

    pvpq, pq = case.pvpq, case.pq
    if sparse:
        return b_prime[pvpq, :][:, pvpq].tocsc(), b_dprime[pq, :][:, pq].tocsc()
    return b_prime[np.ix_(pvpq, pvpq)], b_dprime[np.ix_(pq, pq)]


def _factorise(mat, sparse: bool):
    """Pre-factorise a constant matrix; FDLF's whole point is doing this once."""
    if sparse:
        lu = spla.splu(mat.tocsc())
        return lu.solve
    lu_and_piv = __import__("scipy.linalg", fromlist=["lu_factor"]).lu_factor(mat)
    from scipy.linalg import lu_solve

    return lambda rhs: lu_solve(lu_and_piv, rhs)


def solve(
    case: PowerCase,
    options: Optional[SolverOptions] = None,
    ybus=None,
    variant: str = "XB",
) -> PowerFlowResult:
    """Solve the power flow of ``case`` with the fast decoupled method."""
    options = (options or SolverOptions()).validate()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)

    pvpq, pq = case.pvpq, case.pq
    b_prime, b_dprime = make_b_matrices(case, variant, options.sparse)
    solve_p = _factorise(b_prime, options.sparse)
    solve_q = _factorise(b_dprime, options.sparse) if len(pq) else None

    v = initial_voltage(case, options)
    vm, va = np.abs(v), np.angle(v)
    tracker = IterationTracker(options)

    def scaled_mismatch(v_now):
        """``(dP/|V|, dQ/|V|)`` -- the right-hand sides of the two half-steps."""
        mis = (case.s_sch - s_injected(ybus, v_now)) / np.abs(v_now)
        return mis.real[pvpq], mis.imag[pq]

    while True:
        dp, dq = scaled_mismatch(v)
        if tracker.begin_step(v, max(max_abs(dp), max_abs(dq))):
            break

        # --- P-delta half step ---
        d_va = solve_p(dp)
        va[pvpq] += d_va
        v = vm * np.exp(1j * va)

        # --- Q-|V| half step, using the freshly updated angles ---
        d_vm = np.zeros(0)
        if solve_q is not None:
            _, dq = scaled_mismatch(v)
            d_vm = solve_q(dq)
            vm[pq] += d_vm
            v = vm * np.exp(1j * va)

        tracker.end_step(max(max_abs(d_va), max_abs(d_vm)))

    return tracker.finish(f"Fast Decoupled ({variant})", case, v, variant=variant)
