from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

from ..results import PowerFlowResult
from . import simplified_nr, standard_nr
from .base import SolverOptions, solve_with_q_limits

__all__ = [
    "SolverOptions",
    "solve_with_q_limits",
    "standard_nr",
    "simplified_nr",
    "SOLVERS",
    "PAPER_SOLVERS",
    "get_solver",
]


def _pnr_augmented(case, options=None, ybus=None) -> PowerFlowResult:
    options = replace(options or SolverOptions(), pv_handling="augmented")
    return simplified_nr.solve(case, options, ybus)


SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
    "PNR+": _pnr_augmented,
}

PAPER_SOLVERS: Dict[str, Callable[..., PowerFlowResult]] = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
}

SOLVER_LABELS = {
    "SNR": "Standard NR (power mismatch)",
    "PNR": "Simplified NR (current mismatch)",
    "PNR+": "Simplified NR + augmented PV",
}


def get_solver(key: str) -> Callable[..., PowerFlowResult]:
    try:
        return SOLVERS[key]
    except KeyError:
        raise KeyError(f"unknown solver {key!r}; choose from {sorted(SOLVERS)}") from None
