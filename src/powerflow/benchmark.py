"""
Timing and memory measurement.

The base paper reports wall-clock times on 2003-era hardware (Table 4), which
cannot be reproduced.  What *can* be reproduced is the shape of the result: the
ratio of standard to simplified NR time, how the ratio moves with system size,
and how the two compare against FDLF and Gauss-Seidel.  This module makes those
measurements defensible:

* every solve is repeated and the **minimum** is reported (the least
  noise-contaminated sample), with the median and spread kept alongside;
* ``Y_bus`` is built once, outside the timed region, so the measurement isolates
  the iteration loop -- which is the only thing the two methods differ in;
* a warm-up solve is discarded so that NumPy/SciPy first-call overheads and
  CPU frequency ramp-up do not land in the first sample;
* **time per iteration** is reported next to total time, because the two methods
  do not always take the same number of iterations and the paper's claim is
  specifically about the cost *of one iteration*.

Memory is measured with :mod:`tracemalloc`, which counts Python-level
allocations (including NumPy array headers and SciPy factorisation buffers) but
not the interpreter baseline.  It is reported as the peak increment above the
pre-solve level.
"""
from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:            # pandas is imported lazily inside the functions that need it
    import pandas

from .case import PowerCase
from .results import PowerFlowResult
from .solvers import SOLVERS, SolverOptions
from .solvers.base import assemble_reduced, linear_solve
from .ybus import build_ybus


@dataclass
class BenchmarkRow:
    """One (case, solver) measurement."""

    case: str
    n_bus: int
    n_branch: int
    n_pv: int
    n_pq: int
    solver: str
    converged: bool
    iterations: int
    t_min: float           # seconds, best of `repeats`
    t_median: float
    t_spread: float        # (max - min) / min, a noise indicator
    t_per_iter: float
    peak_mem_kb: float = np.nan
    jac_rows: int = 0
    jac_nnz: int = 0
    jac_bytes: float = np.nan
    max_mismatch: float = np.nan
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def time_solver(
    solver: Callable[..., PowerFlowResult],
    case: PowerCase,
    options: Optional[SolverOptions] = None,
    repeats: int = 7,
    warmup: int = 1,
    ybus=None,
):
    """Time ``solver`` on ``case``.  Returns ``(result, samples_in_seconds)``."""
    options = options or SolverOptions()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)

    result = None
    for _ in range(max(warmup, 0)):
        result = solver(case, options, ybus)

    samples: List[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = solver(case, options, ybus)
            samples.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()

    if result is not None:
        result.elapsed = min(samples)
    return result, samples


def measure_memory(
    solver: Callable[..., PowerFlowResult],
    case: PowerCase,
    options: Optional[SolverOptions] = None,
    ybus=None,
) -> float:
    """Peak Python heap increment during one solve, in kilobytes."""
    options = options or SolverOptions()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)
    solver(case, options, ybus)          # warm caches so they are not counted
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    solver(case, options, ybus)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (peak - base) / 1024.0


def jacobian_footprint(case: PowerCase, solver_key: str, options: SolverOptions) -> dict:
    """Size and sparsity of the linear system each method has to factorise.

    This is the structural half of the "memory overhead" comparison asked for in
    the project proposal: the two NR variants solve systems of the *same*
    dimension, the rectangular method solves a larger one, and FDLF solves two
    smaller constant ones.
    """
    ybus = build_ybus(case, sparse=options.sparse)

    if solver_key in ("SNR", "PNR"):
        from .solvers import simplified_nr, standard_nr

        mod = standard_nr if solver_key == "SNR" else simplified_nr
        jac = mod.jacobian(case, ybus, case.flat_start())
    elif solver_key == "PNR+":
        from .solvers import simplified_nr as mod

        v = case.flat_start()
        s_eff = mod.effective_schedule(case, ybus, v)
        jac = mod.jacobian_augmented(case, ybus, v, s_eff)
    elif solver_key == "RCI":
        from .solvers import rect_current_nr as mod

        v = case.flat_start()
        jac = mod.jacobian(case, ybus, v, case.q_sch.copy())
    elif solver_key.startswith("FDLF"):
        from .solvers.fast_decoupled import make_b_matrices

        bp, bpp = make_b_matrices(case, solver_key.split("-")[1], options.sparse)
        rows = bp.shape[0] + bpp.shape[0]
        nnz = int(sp.csr_matrix(bp).nnz + sp.csr_matrix(bpp).nnz)
        return {"jac_rows": rows, "jac_nnz": nnz, "jac_bytes": float(nnz * 8)}
    else:                                   # Gauss-Seidel factorises nothing
        return {"jac_rows": 0, "jac_nnz": 0, "jac_bytes": 0.0}

    csr = jac.tocsr() if sp.issparse(jac) else sp.csr_matrix(jac)
    rows = jac.shape[0]
    dense_bytes = float(rows * rows * 8)
    return {
        "jac_rows": rows,
        "jac_nnz": int(csr.nnz),
        "jac_bytes": float(csr.nnz * 8) if options.sparse else dense_bytes,
    }


def benchmark_case(
    case: PowerCase,
    solver_keys: Sequence[str],
    options: Optional[SolverOptions] = None,
    repeats: int = 7,
    with_memory: bool = True,
    per_solver_options: Optional[Dict[str, SolverOptions]] = None,
    per_solver_repeats: Optional[Dict[str, int]] = None,
) -> List[BenchmarkRow]:
    """Benchmark every solver in ``solver_keys`` on one case.

    ``per_solver_repeats`` lets a slow solver be sampled fewer times -- Gauss-Seidel
    can be three orders of magnitude slower than the Newton methods, and repeating
    it as often would dominate the whole sweep for no extra precision.
    """
    base_options = options or SolverOptions()
    per_solver_options = per_solver_options or {}
    per_solver_repeats = per_solver_repeats or {}
    rows: List[BenchmarkRow] = []

    for key in solver_keys:
        opts = per_solver_options.get(key, base_options)
        reps = per_solver_repeats.get(key, repeats)
        ybus = build_ybus(case, sparse=opts.sparse)
        try:
            result, samples = time_solver(SOLVERS[key], case, opts, repeats=reps, ybus=ybus)
        except Exception as exc:                       # keep the sweep going
            rows.append(
                BenchmarkRow(
                    case=case.name, n_bus=case.n_bus, n_branch=case.n_branch,
                    n_pv=len(case.pv), n_pq=len(case.pq), solver=key, converged=False,
                    iterations=0, t_min=np.nan, t_median=np.nan, t_spread=np.nan,
                    t_per_iter=np.nan, notes=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        mem = measure_memory(SOLVERS[key], case, opts, ybus) if with_memory else np.nan
        foot = jacobian_footprint(case, key, opts)
        t_min = min(samples)
        rows.append(
            BenchmarkRow(
                case=case.name,
                n_bus=case.n_bus,
                n_branch=case.n_branch,
                n_pv=len(case.pv),
                n_pq=len(case.pq),
                solver=key,
                converged=result.converged,
                iterations=result.iterations,
                t_min=t_min,
                t_median=statistics.median(samples),
                t_spread=(max(samples) - t_min) / t_min if t_min > 0 else np.nan,
                t_per_iter=t_min / result.iterations if result.iterations else np.nan,
                peak_mem_kb=mem,
                max_mismatch=result.history[-1].max_mismatch if result.history else np.nan,
                notes="" if result.converged else result.message,
                **foot,
            )
        )
    return rows


def benchmark_suite(
    cases: Iterable[PowerCase],
    solver_keys: Sequence[str],
    options: Optional[SolverOptions] = None,
    repeats: int = 7,
    with_memory: bool = True,
    per_solver_options: Optional[Dict[str, SolverOptions]] = None,
    per_solver_repeats: Optional[Dict[str, int]] = None,
):
    """Benchmark a list of cases and return a tidy ``pandas`` DataFrame."""
    import pandas as pd

    rows: List[BenchmarkRow] = []
    for case in cases:
        rows.extend(
            benchmark_case(case, solver_keys, options, repeats, with_memory,
                           per_solver_options, per_solver_repeats)
        )
    return pd.DataFrame([r.as_dict() for r in rows])


def component_timings(case: PowerCase, options: Optional[SolverOptions] = None,
                      repeats: int = 200) -> "pandas.DataFrame":
    """Time one iteration of SNR and PNR broken into its three stages.

    The base paper's claim is specifically about the cost of *rebuilding the
    Jacobian*, and it assumes "other steps of the two NR methods are exactly the
    same" (Section 3).  Total solve time therefore tests the claim only
    indirectly: it also contains the mismatch evaluation and the linear solve,
    which are genuinely identical work for the two methods and dilute whatever
    advantage the Jacobian assembly has.  This function measures the three
    stages separately, at the flat start, so the claim can be tested directly.
    """
    import pandas as pd

    from .solvers import simplified_nr, standard_nr

    options = options or SolverOptions()
    ybus = build_ybus(case, sparse=options.sparse)
    v = case.flat_start()

    def bench(fn) -> float:
        # A single warm-up is not enough here: the first few calls on the larger
        # systems pay one-off allocator growth that otherwise lands in the
        # minimum and inflates the result by 2x or more.
        for _ in range(max(5, repeats // 10)):
            fn()
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    s_eff = simplified_nr.effective_schedule(case, ybus, v)
    pvpq, pq = case.pvpq, case.pq
    stages = {
        "SNR": {
            "mismatch": lambda: standard_nr.mismatch_vector(case, ybus, v),
            "derivatives": lambda: standard_nr.derivative_matrices(case, ybus, v),
        },
        "PNR": {
            "mismatch": lambda: simplified_nr.mismatch_vector(case, ybus, v, s_eff),
            "derivatives": lambda: simplified_nr.derivative_matrices(case, ybus, v, s_eff),
        },
    }

    rows = []
    for key, fns in stages.items():
        t_mis = bench(fns["mismatch"])
        t_der = bench(fns["derivatives"])

        # `assemble_reduced` is bit-for-bit the same work for both methods --
        # same shapes, same index sets -- so it is timed separately rather than
        # charged to either formulation.
        m_a, m_m = fns["derivatives"]()
        t_asm = bench(lambda: assemble_reduced(m_a, m_m, pvpq, pq))

        jac = assemble_reduced(m_a, m_m, pvpq, pq)
        rhs = np.ones(jac.shape[0])
        t_solve = bench(lambda: linear_solve(jac, rhs))

        total = t_mis + t_der + t_asm + t_solve
        rows.append(
            {
                "case": case.name,
                "n_bus": case.n_bus,
                "solver": key,
                "t_mismatch_us": t_mis * 1e6,
                "t_derivatives_us": t_der * 1e6,
                "t_assembly_us": t_asm * 1e6,
                "t_jacobian_us": (t_der + t_asm) * 1e6,
                "t_solve_us": t_solve * 1e6,
                "t_iteration_us": total * 1e6,
                "derivative_share": t_der / total,
            }
        )
    return pd.DataFrame(rows)


def reference_solution(case: PowerCase, tol: float = 1e-13) -> np.ndarray:
    """A tightly converged solution, used as the truth for the error plots."""
    from .solvers import standard_nr

    opts = SolverOptions(tol=tol, criterion="mismatch", max_iter=100)
    res = standard_nr.solve(case, opts)
    return res.v
