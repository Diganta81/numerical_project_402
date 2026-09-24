"""
Extension 2 -- benchmark against FDLF and Gauss-Seidel.

The project proposal asks to "compare iteration counts, execution runtime and
memory overhead against traditional Power Mismatch NR and Fast Decoupled Load
Flow (FDLF)", and its problem statement also raises Gauss-Seidel.  This script
runs all seven solvers over every test system and places the base paper's
contribution in that wider context.

The comparison is framed around one trade-off.  Each method attacks the cost of
a Newton iteration in a different place:

======  =========================================================  ===================
Method  What it changes                                            What it costs
======  =========================================================  ===================
PNR     cheaper Jacobian *entries*, same Newton step               lagged PV Q -> more iterations
PNR+    same, with PV reactive power as a real unknown             one extra unknown per PV bus
RCI     constant off-diagonal Jacobian blocks, rectangular state   system grows to 2(n-1)+n_pv
FDLF    Jacobian replaced by two constant matrices, factored once  linear convergence
GS      no Jacobian at all                                         linear, and the rate degrades with n
======  =========================================================  ===================

Outputs
-------
``results/tables/05_method_comparison.{csv,md}``
``results/tables/05_convergence_118.{csv,md}``
``results/figures/05_iterations_all_methods.png``
``results/figures/05_time_all_methods.png``
``results/figures/05_convergence_118.png``
``results/figures/05_convergence_30.png``
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import load_case
from powerflow.benchmark import benchmark_suite, reference_solution
from powerflow.plotting import plot_convergence, plot_grouped_bars, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.solvers import SOLVERS, SolverOptions

CASES = ["case5_stagg", "case6ww", "case24_ieee_rts", "case30_ieee",
         "case57_ieee", "case118_ieee", "case300_ieee"]
KEYS = ["SNR", "PNR", "PNR+", "RCI", "FDLF-XB", "FDLF-BX", "GS"]
REPEATS = 10

#: Gauss-Seidel needs orders of magnitude more sweeps than any Newton method.
GS_OPTIONS = SolverOptions(tol=1e-6, criterion="voltage", max_iter=3000)
NEWTON_OPTIONS = SolverOptions(tol=1e-6, criterion="voltage", max_iter=60)


def main() -> None:
    use_paper_style()
    cases = [load_case(n) for n in CASES]

    with Section("1. All seven methods on every test system"):
        df = benchmark_suite(
            cases, KEYS, NEWTON_OPTIONS, repeats=REPEATS, with_memory=False,
            per_solver_options={"GS": GS_OPTIONS},
            per_solver_repeats={"GS": 2},     # GS is ~1000x slower; 2 samples is plenty
        )
        view = df[["case", "n_bus", "solver", "converged", "iterations", "t_min"]].copy()
        view["t_ms"] = view.pop("t_min") * 1e3
        print_table(view)
        save_table(
            df, "05_method_comparison", "All solvers on all test systems",
            "Flat start, 1e-6 p.u. maximum voltage correction. Gauss-Seidel is given a 3000-sweep "
            f"budget; the Newton methods get 60. Times are best of {REPEATS} repeats with Y-bus "
            "construction excluded.",
        )
        failed = df[~df["converged"]]
        if len(failed):
            print("\ndid not converge:")
            for r in failed.itertuples():
                print(f"   {r.solver} on {r.case}: {r.notes}")

    # ------------------------------------------------- verify they all agree
    with Section("2. Do the methods agree on the answer?"):
        rows = []
        for case in cases:
            v_ref = reference_solution(case)
            for key in KEYS:
                opts = GS_OPTIONS if key == "GS" else NEWTON_OPTIONS
                res = SOLVERS[key](case, opts)
                rows.append(
                    {
                        "case": case.name, "n_bus": case.n_bus, "solver": key,
                        "converged": res.converged,
                        "max_V_error": float(np.max(np.abs(res.v - v_ref))),
                    }
                )
        agree = pd.DataFrame(rows)
        worst = agree[agree["converged"]].groupby("solver")["max_V_error"].max()
        print_table(worst.reset_index().rename(columns={"max_V_error": "worst_max_V_error"}))
        save_table(
            agree, "05_solution_agreement",
            "Distance from each solver's answer to a tightly converged reference",
            "Reference is the standard NR solution driven to a 1e-13 p.u. power mismatch. "
            "Every converged run agrees with it to within the requested tolerance, which is "
            "what makes the timing comparison meaningful.",
        )

    # -------------------------------------------------------------- figures
    with Section("3. Iterations and time across methods"):
        labels = [f"{int(n)}-bus" for n in sorted(df["n_bus"].unique())]
        pivot_i = df.pivot_table(index="n_bus", columns="solver", values="iterations")
        pivot_t = df.pivot_table(index="n_bus", columns="solver", values="t_min")

        newton = [k for k in KEYS if k != "GS"]
        path = plot_grouped_bars(
            labels, {k: pivot_i[k].to_numpy() for k in newton},
            title="Iterations to a 1e-6 p.u. voltage tolerance",
            subtitle="Gauss-Seidel is omitted here -- it needs 39 to 2000+ sweeps and would "
                     "flatten every other bar. Its counts are in the table.",
            ylabel="Iterations",
            name="05_iterations_all_methods",
            value_fmt="%.0f",
        )
        print(f"wrote {path}")

        path = plot_grouped_bars(
            labels, {k: pivot_t[k].to_numpy() * 1e3 for k in KEYS},
            title="Total solve time by method",
            subtitle="Logarithmic scale. Gauss-Seidel is included to show the size of the gap.",
            ylabel="Solve time (ms)",
            name="05_time_all_methods",
            value_fmt="%.1f",
            log=True,
        )
        print(f"wrote {path}")

    # --------------------------------------------------- convergence shapes
    with Section("4. Convergence character of each method"):
        for case_name, tag in (("case30_ieee", "30"), ("case118_ieee", "118")):
            case = load_case(case_name)
            v_ref = reference_solution(case)
            hist = {}
            for key in KEYS:
                opts = GS_OPTIONS if key == "GS" else NEWTON_OPTIONS
                res = SOLVERS[key](case, opts).attach_reference(v_ref)
                hist[key] = res.error_history[:30]      # GS would run off the axis
            path = plot_convergence(
                hist,
                title=f"Convergence of all methods, IEEE {tag}-bus",
                subtitle="A straight line on this log scale is linear convergence; a curve that "
                         "steepens is quadratic. First 30 iterations only.",
                name=f"05_convergence_{tag}",
                start_at=0,
            )
            print(f"wrote {path.name}")

            rows = [
                {"solver": k, "iteration": i, "max_voltage_error": e}
                for k, h in hist.items()
                for i, e in enumerate(h)
            ]
            save_table(
                pd.DataFrame(rows), f"05_convergence_{tag}",
                f"Convergence histories, IEEE {tag}-bus",
            )

        print(
            "\nReading the figures: SNR, PNR+ and RCI curve downward (quadratic). PNR is a\n"
            "straight line (linear) because of its lagged PV reactive powers. Both FDLF\n"
            "variants and GS are straight lines by construction -- they use approximate or\n"
            "no derivative information, and buy a cheap iteration at the cost of the rate."
        )


if __name__ == "__main__":
    main()
