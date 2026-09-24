"""
Power-flow solvers.

Every solver exposes the same entry point::

    solve(case, options=None, ybus=None) -> PowerFlowResult

so they can be swapped freely in the benchmarks.  They differ only in the set of
nonlinear equations they iterate on:

=========  ==================================  ===========================================
Key        Module                              Residual
=========  ==================================  ===========================================
``SNR``    :mod:`.standard_nr`                 power mismatch, polar state (the benchmark)
``PNR``    :mod:`.simplified_nr`               current mismatch, polar state (**base paper**)
``PNR+``   :mod:`.simplified_nr` (augmented)   as above, PV reactive power as an unknown
``RCI``    :mod:`.rect_current_nr`             current mismatch, rectangular state
``FDLF``   :mod:`.fast_decoupled`              decoupled power mismatch, constant B', B''
``GS``     :mod:`.gauss_seidel`                nodal voltage fixed point
=========  ==================================  ===========================================
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

from ..results import PowerFlowResult
from . import fast_decoupled, gauss_seidel, rect_current_nr, simplified_nr, standard_nr
from .base import SolverOptions, solve_with_q_limits

__all__ = [
    "SolverOptions",
    "solve_with_q_limits",
    "standard_nr",
    "simplified_nr",
    "rect_current_nr",
    "fast_decoupled",
    "gauss_seidel",
    "SOLVERS",
    "PAPER_SOLVERS",
    "get_solver",
]


def _pnr_augmented(case, options=None, ybus=None) -> PowerFlowResult:
    options = replace(options or SolverOptions(), pv_handling="augmented")
    return simplified_nr.solve(case, options, ybus)


def _fdlf_bx(case, options=None, ybus=None) -> PowerFlowResult:
    return fast_decoupled.solve(case, options, ybus, variant="BX")


#: Every solver, keyed by the short label used in tables and figures.
SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
    "PNR+": _pnr_augmented,
    "RCI": rect_current_nr.solve,
    "FDLF-XB": fast_decoupled.solve,
    "FDLF-BX": _fdlf_bx,
    "GS": gauss_seidel.solve,
}

#: The two methods the base paper compares (its Table 5 and Figures 4-10).
PAPER_SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
}

#: Human-readable names, for figure legends and report tables.
SOLVER_LABELS = {
    "SNR": "Standard NR (power mismatch)",
    "PNR": "Simplified NR (current mismatch)",
    "PNR+": "Simplified NR + augmented PV",
    "RCI": "Rectangular current injection",
    "FDLF-XB": "Fast decoupled (XB)",
    "FDLF-BX": "Fast decoupled (BX)",
    "GS": "Gauss-Seidel",
}


def get_solver(key: str) -> Callable[..., PowerFlowResult]:
    try:
        return SOLVERS[key]
    except KeyError:
        raise KeyError(f"unknown solver {key!r}; choose from {sorted(SOLVERS)}") from None
