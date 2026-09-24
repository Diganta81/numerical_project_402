"""Result objects shared by every solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class IterationRecord:
    """State of one Newton (or Gauss-Seidel) iteration.

    max_mismatch is the solver's native residual norm; max_dv is the largest
    correction applied to the voltage vector (the termination criterion);
    max_v_error is the distance to the converged solution, filled in later by
    PowerFlowResult.attach_reference.
    """

    iteration: int
    max_mismatch: float
    max_dv: float = np.nan
    max_v_error: float = np.nan


@dataclass
class PowerFlowResult:
    """Outcome of a power-flow solve."""

    method: str
    case_name: str
    v: np.ndarray                      # complex bus voltages, p.u.
    converged: bool
    iterations: int
    history: List[IterationRecord] = field(default_factory=list)
    elapsed: float = np.nan            # seconds, single solve
    message: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def vm(self) -> np.ndarray:
        return np.abs(self.v)

    @property
    def va_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.v))

    @property
    def mismatch_history(self) -> np.ndarray:
        return np.array([r.max_mismatch for r in self.history])

    @property
    def dv_history(self) -> np.ndarray:
        return np.array([r.max_dv for r in self.history])

    @property
    def error_history(self) -> np.ndarray:
        return np.array([r.max_v_error for r in self.history])

    def attach_reference(self, v_ref: np.ndarray) -> "PowerFlowResult":
        """Fill in max_v_error against a converged reference solution."""
        for rec, v in zip(self.history, self.extras.get("v_trace", [])):
            rec.max_v_error = float(np.max(np.abs(v - v_ref)))
        return self

    def injections(self, ybus) -> np.ndarray:
        """Complex power injection implied by the solved voltages, in p.u."""
        import scipy.sparse as sp

        i = ybus @ self.v if sp.issparse(ybus) else ybus.dot(self.v)
        return self.v * np.conj(i)

    def summary(self) -> str:
        status = "converged" if self.converged else "DIVERGED"
        return (
            f"{self.method:<28s} {self.case_name:<18s} {status:>10s} in "
            f"{self.iterations:2d} iterations, {self.elapsed * 1e3:8.3f} ms"
        )

    def __repr__(self) -> str:
        return f"<PowerFlowResult {self.summary()}>"


def max_voltage_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest complex-plane distance between two voltage vectors, in p.u."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
