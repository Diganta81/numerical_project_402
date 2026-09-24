from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..case import PQ, PowerCase
from ..results import IterationRecord, PowerFlowResult


@dataclass
class SolverOptions:

    tol: float = 1e-6
    criterion: str = "voltage"        
    max_iter: int = 50
    sparse: bool = False              
    v0: Optional[np.ndarray] = None   
    jacobian: str = "vectorized"      
    enforce_q_limits: bool = False
    record_trace: bool = True

    pv_handling: str = "paper"

    def validate(self) -> "SolverOptions":
        if self.criterion not in ("voltage", "mismatch"):
            raise ValueError(f"unknown criterion {self.criterion!r}")
        if self.jacobian not in ("vectorized", "reference"):
            raise ValueError(f"unknown jacobian mode {self.jacobian!r}")
        if self.pv_handling not in ("paper", "augmented"):
            raise ValueError(f"unknown pv_handling {self.pv_handling!r}")
        return self


def matvec(ybus, v: np.ndarray) -> np.ndarray:
    return ybus @ v


def s_injected(ybus, v: np.ndarray) -> np.ndarray:
    return v * np.conj(matvec(ybus, v))


def linear_solve(jac, rhs: np.ndarray) -> np.ndarray:
    if sp.issparse(jac):
        return spla.spsolve(jac.tocsc(), rhs)
    return np.linalg.solve(jac, rhs)


def assemble_reduced(m_angle, m_magnitude, pvpq: np.ndarray, pq: np.ndarray):
    if sp.issparse(m_angle):
        return sp.bmat(
            [
                [m_angle[pvpq, :][:, pvpq].real, m_magnitude[pvpq, :][:, pq].real],
                [m_angle[pq, :][:, pvpq].imag, m_magnitude[pq, :][:, pq].imag],
            ],
            format="csr",
        )
    return np.block(
        [
            [m_angle[np.ix_(pvpq, pvpq)].real, m_magnitude[np.ix_(pvpq, pq)].real],
            [m_angle[np.ix_(pq, pvpq)].imag, m_magnitude[np.ix_(pq, pq)].imag],
        ]
    )


def max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if len(x) else 0.0



class IterationTracker:
    def __init__(self, options: SolverOptions):
        self.options = options
        self.history: List[IterationRecord] = []
        self.trace: List[np.ndarray] = []
        self.n_updates = 0
        self.converged = False
        self._stop_next = False

    def begin_step(self, v: np.ndarray, mismatch: float) -> bool:
        rec = IterationRecord(iteration=len(self.history), max_mismatch=mismatch)
        self.history.append(rec)
        if self.options.record_trace:
            self.trace.append(np.array(v, dtype=complex))

        if self._stop_next:      
            self.converged = True
            return True
        if self.options.criterion == "mismatch" and mismatch < self.options.tol:
            self.converged = True
            return True
        if not np.isfinite(mismatch) or mismatch > 1e12:
            return True                           # blown up: stop, not converged
        return self.n_updates >= self.options.max_iter

    def end_step(self, dv: float) -> None:
        self.history[-1].max_dv = dv
        self.n_updates += 1
        if self.options.criterion == "voltage" and dv < self.options.tol:
            self._stop_next = True

    def finish(self, method: str, case: PowerCase, v: np.ndarray, **extras) -> PowerFlowResult:
        msg = "" if self.converged else (
            f"did not reach tol={self.options.tol:g} within {self.options.max_iter} iterations"
        )
        return PowerFlowResult(
            method=method,
            case_name=case.name,
            v=v,
            converged=self.converged,
            iterations=self.n_updates,
            history=self.history,
            message=msg,
            extras={"v_trace": self.trace, **extras},
        )


# -------------------------------------------------------------- start values
def initial_voltage(case: PowerCase, options: SolverOptions) -> np.ndarray:
    if options.v0 is not None:
        return np.array(options.v0, dtype=complex)
    return case.flat_start()


def solve_with_q_limits(
    solver: Callable[..., PowerFlowResult],
    case: PowerCase,
    options: SolverOptions,
    max_switches: int = 10,
) -> PowerFlowResult:
    import copy

    from ..ybus import build_ybus

    work = copy.deepcopy(case)
    work.name = case.name
    opts = replace(options, enforce_q_limits=False)
    total_iter = 0
    switched: List[int] = []

    for _ in range(max_switches + 1):
        res = solver(work, opts)
        total_iter += res.iterations
        if not res.converged:
            break

        ybus = build_ybus(work, sparse=options.sparse)
        q = np.imag(s_injected(ybus, res.v))
        violated = []
        for k in work.pv:
            if q[k] > case.q_max[k] + 1e-8:
                work.q_sch[k] = case.q_max[k]
                violated.append(k)
            elif q[k] < case.q_min[k] - 1e-8:
                work.q_sch[k] = case.q_min[k]
                violated.append(k)
        if not violated:
            res.iterations = total_iter
            res.extras["q_limit_switches"] = switched
            return res
        work.bus_type = work.bus_type.copy()
        work.bus_type[violated] = PQ
        switched.extend(int(work.bus_ids[k]) for k in violated)
        opts = replace(opts, v0=res.v)

    res.iterations = total_iter
    res.extras["q_limit_switches"] = switched
    return res
