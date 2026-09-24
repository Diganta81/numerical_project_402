"""
Extension 1 -- scale testing beyond the paper's test systems.

The project proposal asks to "extend evaluations beyond small networks to
large-scale grids (e.g. IEEE 118-bus)".  The base paper stops at 57 buses, which
is exactly the size range where the effect it is arguing about is still buried
under fixed per-call overheads.  This script runs the same two methods (plus the
augmented-PV variant) from 3 to 300 buses and fits how each cost actually grows.

Three separate questions are asked, because they have different answers:

1. Does the **iteration count** stay flat as the system grows?  For the standard
   method, yes.  For the paper's method, no -- and the reason is the lagged PV
   reactive powers, which is what the augmented variant fixes.
2. Does the **Jacobian derivative evaluation** get relatively cheaper, as the
   paper's FLOP analysis predicts?  Yes, and the advantage grows with size.
3. Does **total solve time** follow?  Only once the system is big enough for
   arithmetic to dominate interpreter overhead.

Outputs
-------
``results/tables/04_scale_results.{csv,md}``      full sweep
``results/tables/04_scaling_exponents.{csv,md}``  fitted growth exponents
``results/figures/04_iterations_vs_size.png``
``results/figures/04_time_vs_size.png``
``results/figures/04_derivative_ratio_vs_size.png``
``results/figures/04_dense_vs_sparse.png``
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import load_case
from powerflow.benchmark import benchmark_suite, component_timings
from powerflow.plotting import plot_grouped_bars, plot_scaling, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.solvers import SolverOptions

LADDER = [
    "case3_saadat", "case5_stagg", "case6ww", "case14_ieee", "case24_ieee_rts",
    "case30_ieee", "case57_ieee", "case118_ieee", "case300_ieee",
]
SOLVER_KEYS = ["SNR", "PNR", "PNR+", "RCI"]
REPEATS = 15


#: Below this size a solve is dominated by fixed NumPy/interpreter call overhead
#: rather than arithmetic, so a growth exponent fitted through the small systems
#: is meaningless (it comes out near 1 no matter what the algorithm costs).
ASYMPTOTIC_FROM = 30


def fit_exponent(n, y, min_n: int = 1):
    """Least-squares slope of ``log y`` against ``log n`` -- the growth exponent."""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0) & (n >= min_n)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(n[ok]), np.log(y[ok]), 1)[0])


def main() -> None:
    use_paper_style()
    cases = [load_case(name) for name in LADDER]
    options = SolverOptions(tol=1e-6, criterion="voltage", max_iter=60)

    # ------------------------------------------------------------- the sweep
    with Section("1. Solver sweep from 3 to 300 buses"):
        df = benchmark_suite(cases, SOLVER_KEYS, options, repeats=REPEATS, with_memory=False)
        df = df.sort_values(["n_bus", "solver"])
        view = df[["case", "n_bus", "n_pv", "solver", "converged", "iterations",
                   "t_min", "t_per_iter"]].copy()
        view["t_ms"] = view.pop("t_min") * 1e3
        view["t_per_iter_ms"] = view.pop("t_per_iter") * 1e3
        print_table(view)
        save_table(
            df, "04_scale_results", "Solver sweep from 3 to 300 buses",
            f"Flat start, 1e-6 p.u. voltage tolerance, best of {REPEATS} repeats with Y-bus "
            "construction excluded. Dense linear algebra throughout; the sparse comparison is "
            "in section 4 below.",
        )

    pivot_iter = df.pivot_table(index="n_bus", columns="solver", values="iterations")
    pivot_time = df.pivot_table(index="n_bus", columns="solver", values="t_min")
    sizes = pivot_iter.index.to_numpy()

    # ------------------------------------------------------ iteration counts
    with Section("2. Iteration count versus system size"):
        print_table(pivot_iter.reset_index())
        path = plot_scaling(
            sizes,
            {k: pivot_iter[k].to_numpy() for k in SOLVER_KEYS},
            title="Iterations to a 1e-6 p.u. voltage tolerance",
            ylabel="Iterations",
            subtitle="The standard method stays flat; the paper's formulation grows with the "
                     "number of PV buses because it lags their reactive powers. Making Q a "
                     "Newton unknown (PNR+) removes the growth entirely.",
            name="04_iterations_vs_size",
            loglog=False,
            xlog=True,
        )
        print(f"\nwrote {path}")
        growth = pivot_iter.loc[300] / pivot_iter.loc[5]
        print("\niterations at n=300 relative to n=5:")
        for k in SOLVER_KEYS:
            print(f"   {k:<5s} {pivot_iter.loc[5, k]:.0f} -> {pivot_iter.loc[300, k]:.0f} "
                  f"({growth[k]:.2f}x)")

    # --------------------------------------------------------- timing growth
    with Section("3. Solve time versus system size"):
        path = plot_scaling(
            sizes,
            {k: pivot_time[k].to_numpy() * 1e3 for k in SOLVER_KEYS},
            title="Total solve time versus system size",
            ylabel="Solve time (ms)",
            subtitle=f"Best of {REPEATS} repeats, dense linear algebra, Y-bus excluded.",
            name="04_time_vs_size",
        )
        print(f"wrote {path}")

        rows = []
        for k in SOLVER_KEYS:
            per_iter = df[df["solver"] == k].sort_values("n_bus")["t_per_iter"].to_numpy()
            rows.append(
                {
                    "solver": k,
                    "time_exponent_all": fit_exponent(sizes, pivot_time[k].to_numpy()),
                    "time_exponent_large": fit_exponent(
                        sizes, pivot_time[k].to_numpy(), ASYMPTOTIC_FROM
                    ),
                    "per_iter_exponent_large": fit_exponent(sizes, per_iter, ASYMPTOTIC_FROM),
                    "iteration_exponent": fit_exponent(sizes, pivot_iter[k].to_numpy()),
                }
            )
        exponents = pd.DataFrame(rows)
        print()
        print_table(exponents)
        save_table(
            exponents, "04_scaling_exponents",
            "Fitted growth exponents, log-log slope of cost against bus count",
            "An exponent of 2 means quadratic growth, 3 cubic. `*_all` is fitted through every "
            f"size and is misleadingly low, because below about {ASYMPTOTIC_FROM} buses a solve "
            "is dominated by fixed interpreter/NumPy call overhead rather than arithmetic. "
            f"`*_large` is fitted from {ASYMPTOTIC_FROM} buses upward and is the meaningful "
            "figure: dense Jacobian assembly is O(n^2) and dense LU is O(n^3), so it lands "
            "between the two.",
        )
        print(f"\n(exponents marked `_large` are fitted from n >= {ASYMPTOTIC_FROM} only; "
              f"below that, fixed call overhead dominates and flattens the fit)")

    # --------------------------------------- derivative-stage ratio versus n
    with Section("4. Does the Jacobian advantage grow with system size?"):
        stage = pd.concat([component_timings(c, options, repeats=150) for c in cases],
                          ignore_index=True)
        wide = stage.pivot_table(index="n_bus", columns="solver",
                                 values=["t_derivatives_us", "t_iteration_us"]).sort_index()
        ratio = pd.DataFrame(
            {
                "n_bus": wide.index.to_numpy(),
                "deriv_SNR_us": wide[("t_derivatives_us", "SNR")].to_numpy(),
                "deriv_PNR_us": wide[("t_derivatives_us", "PNR")].to_numpy(),
                "deriv_ratio": (
                    wide[("t_derivatives_us", "SNR")] / wide[("t_derivatives_us", "PNR")]
                ).to_numpy(),
                "iteration_ratio": (
                    wide[("t_iteration_us", "SNR")] / wide[("t_iteration_us", "PNR")]
                ).to_numpy(),
            }
        )
        print_table(ratio)
        save_table(
            ratio, "04_derivative_ratio",
            "Jacobian derivative evaluation speed-up versus system size",
            "`deriv_ratio` above 1 means the simplified formulation is faster at the step the "
            "paper analyses. The end-to-end `iteration_ratio` stays lower because block "
            "assembly and the LU solve are shared work.",
        )
        path = plot_scaling(
            ratio["n_bus"].to_numpy(),
            {
                "SNR": ratio["deriv_ratio"].to_numpy(),
                "PNR": ratio["iteration_ratio"].to_numpy(),
            },
            title="Speed-up of the simplified formulation versus system size",
            ylabel="Time ratio, standard / simplified",
            subtitle="Upper series: the Jacobian derivative step alone. Lower series: a whole "
                     "iteration. Both above 1.0 favour the simplified method.",
            name="04_derivative_ratio_vs_size",
            loglog=False,
            xlog=True,
        )
        print(f"\nwrote {path}")

    # ------------------------------------------------------ dense vs sparse
    with Section("5. Dense versus sparse linear algebra at scale"):
        big = [load_case(n) for n in ("case57_ieee", "case118_ieee", "case300_ieee")]
        dense = benchmark_suite(big, ["SNR", "PNR"], SolverOptions(tol=1e-6, max_iter=60),
                                repeats=REPEATS, with_memory=False)
        sparse = benchmark_suite(big, ["SNR", "PNR"],
                                 SolverOptions(tol=1e-6, max_iter=60, sparse=True),
                                 repeats=REPEATS, with_memory=False)
        dense["storage"] = "dense"
        sparse["storage"] = "sparse"
        both = pd.concat([dense, sparse], ignore_index=True)
        cmp = both.pivot_table(index=["n_bus", "solver"], columns="storage",
                               values=["t_min", "jac_bytes"]).sort_index()
        summary = pd.DataFrame(
            {
                "n_bus": [i[0] for i in cmp.index],
                "solver": [i[1] for i in cmp.index],
                "dense_ms": cmp[("t_min", "dense")].to_numpy() * 1e3,
                "sparse_ms": cmp[("t_min", "sparse")].to_numpy() * 1e3,
                "speedup": (cmp[("t_min", "dense")] / cmp[("t_min", "sparse")]).to_numpy(),
                "dense_jac_kb": cmp[("jac_bytes", "dense")].to_numpy() / 1024,
                "sparse_jac_kb": cmp[("jac_bytes", "sparse")].to_numpy() / 1024,
            }
        )
        print_table(summary)
        save_table(
            summary, "04_dense_vs_sparse",
            "Dense versus sparse storage on the large systems",
            "Sparsity is what actually makes large power flow tractable, and it is orthogonal "
            "to the paper's reformulation -- both methods benefit equally.",
        )
        labels = [f"{int(r.n_bus)}-bus {r.solver}" for r in summary.itertuples()]
        path = plot_grouped_bars(
            labels,
            {"SNR": summary["dense_ms"].to_numpy(), "PNR": summary["sparse_ms"].to_numpy()},
            title="Dense versus sparse solve time",
            subtitle="Series are the two storage schemes, not the two methods; the method is "
                     "named on each x-axis label.",
            ylabel="Solve time (ms)",
            name="04_dense_vs_sparse",
            value_fmt="%.2f",
            log=True,
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
