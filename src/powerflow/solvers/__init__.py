from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

from ..results import PowerFlowResult
from . import fast_decoupled, rect_current_nr, simplified_nr, standard_nr
from .base import SolverOptions, solve_with_q_limits

__all__ = [
    "SolverOptions",
    "solve_with_q_limits",
    "standard_nr",
    "simplified_nr",
    "rect_current_nr",
    "SOLVERS",
    "PAPER_SOLVERS",
    "get_solver",
    "fast_decoupled",
]


def _pnr_augmented(case, options=None, ybus=None) -> PowerFlowResult:
    options = replace(options or SolverOptions(), pv_handling="augmented")
    return simplified_nr.solve(case, options, ybus)


SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
    "PNR+": _pnr_augmented,
    "RCI": rect_current_nr.solve,
    "FDLF-XB": fast_decoupled.solve,
    "FDLF-BX": _fdlf_bx,
}

PAPER_SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
}

SOLVER_LABELS = {
    "SNR": "Standard NR (power mismatch)",
    "PNR": "Simplified NR (current mismatch)",
    "PNR+": "Simplified NR + augmented PV",
    "RCI": "Rectangular current injection",
    "FDLF-XB": "Fast decoupled (XB)",
    "FDLF-BX": "Fast decoupled (BX)",
}


def get_solver(key: str) -> Callable[..., PowerFlowResult]:
    try:
        return SOLVERS[key]
    except KeyError:
        raise KeyError(f"unknown solver {key!r}; choose from {sorted(SOLVERS)}") from None
