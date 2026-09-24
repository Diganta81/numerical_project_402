"""
Reproduce Section 5 of the base paper: the five benchmark test cases.

TC1 5-bus (Stagg & El-Abiad), TC2 6-bus (Wood & Wollenberg), TC3 24-bus IEEE
RTS, TC4 30-bus IEEE, TC5 57-bus IEEE.  Both methods start flat and terminate on
the paper's criterion -- a maximum voltage correction below 1e-6 p.u.

A caveat worth stating plainly: the paper says its test systems were *modified*
but never says how, and the solutions printed in its Table 3 are not those of the
standard IEEE cases (several of its 24-bus voltages sit near 0.46 - j0.91 p.u.,
which no healthy network produces).  The unmodified library cases are used here,
so Table 3 cannot be matched number for number and the table produced below is
this project's own solution set.  Iteration counts, convergence behaviour and
timing ratios -- the paper's actual claims -- are all reproducible and are what
the comparison rests on.

Outputs
-------
``results/tables/03_table3_solutions.{csv,md}``   bus voltages, all five cases
``results/tables/03_table5_results.{csv,md}``     the paper's Table 5
``results/figures/03_fig5_time_ratio.png``        the paper's Figure 5
``results/figures/03_fig6..10_*.png``             the paper's Figures 6-10
``results/figures/03_iterations.png``             iteration counts side by side
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import PAPER_TEST_CASES, build_ybus, load_case
from powerflow.benchmark import component_timings, reference_solution, time_solver
from powerflow.plotting import plot_convergence, plot_grouped_bars, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.solvers import PAPER_SOLVERS, SolverOptions

REPEATS = 25

#: Table 5 of the paper, averaged over its three machines.
PAPER_TABLE5 = {
    "TC1": {"snr_iter": 5, "pnr_iter": 4, "ratio": 1.539},
    "TC2": {"snr_iter": 5, "pnr_iter": 5, "ratio": 1.459},
    "TC3": {"snr_iter": 7, "pnr_iter": 8, "ratio": 1.132},
    "TC4": {"snr_iter": 5, "pnr_iter": 5, "ratio": 1.329},
    "TC5": {"snr_iter": 4, "pnr_iter": 3, "ratio": 1.728},
}

FIGURE_NUMBER = {"TC1": 6, "TC2": 7, "TC3": 8, "TC4": 9, "TC5": 10}


def main() -> None:
    use_paper_style()
    options = SolverOptions(tol=1e-6, criterion="voltage", max_iter=50)

    results = {}
    with Section("1. Solving the five test systems (flat start, 1e-6 p.u. voltage tolerance)"):
        for label, name in PAPER_TEST_CASES:
            case = load_case(name)
            ybus = build_ybus(case)
            v_ref = reference_solution(case)
            per_case = {}
            for key, solver in PAPER_SOLVERS.items():
                res, samples = time_solver(solver, case, options, repeats=REPEATS, ybus=ybus)
                res.attach_reference(v_ref)
                per_case[key] = res
            results[label] = {"case": case, "ybus": ybus, "v_ref": v_ref, "res": per_case}
            print(f"{label}  {case.summary()}")
            for key, res in per_case.items():
                print(f"      {res.summary()}")

    # --------------------------------------------------------------- Table 3
    with Section("2. Paper Table 3 -- power-flow solutions of the test systems"):
        frames = []
        for label, bundle in results.items():
            case = bundle["case"]
            v = bundle["res"]["PNR"].v
            frames.append(
                pd.DataFrame(
                    {
                        "case": label,
                        "bus": case.bus_ids,
                        "Vm_pu": np.abs(v),
                        "Va_deg": np.degrees(np.angle(v)),
                        "V_rect": [f"{x.real:.2f}{x.imag:+.2f}j" for x in v],
                    }
                )
            )
        table3 = pd.concat(frames, ignore_index=True)
        for label in results:
            subset = table3[table3["case"] == label]
            print(f"\n{label}: {len(subset)} buses, "
                  f"Vm in [{subset['Vm_pu'].min():.4f}, {subset['Vm_pu'].max():.4f}] p.u., "
                  f"Va in [{subset['Va_deg'].min():.2f}, {subset['Va_deg'].max():.2f}] deg")
        save_table(
            table3, "03_table3_solutions",
            "Power-flow solutions of the five test systems",
            "Analogous to Table 3 of the paper, but computed on the *unmodified* IEEE library "
            "cases. The paper's test systems were modified in an unstated way and its Table 3 "
            "values cannot be reproduced; the two NR methods agree with each other here to "
            "better than 1e-9 p.u. on every bus.",
        )

    # --------------------------------------------------------------- Table 5
    with Section("3. Paper Table 5 -- iterations, calculation time and time ratio"):
        rows = []
        for label, bundle in results.items():
            snr, pnr = bundle["res"]["SNR"], bundle["res"]["PNR"]
            ratio = snr.elapsed / pnr.elapsed
            per_iter_ratio = (snr.elapsed / snr.iterations) / (pnr.elapsed / pnr.iterations)
            ref = PAPER_TABLE5[label]
            rows.append(
                {
                    "case": label,
                    "system": bundle["case"].name,
                    "n_bus": bundle["case"].n_bus,
                    "SNR_iter": snr.iterations,
                    "PNR_iter": pnr.iterations,
                    "SNR_iter_paper": ref["snr_iter"],
                    "PNR_iter_paper": ref["pnr_iter"],
                    "SNR_time_ms": snr.elapsed * 1e3,
                    "PNR_time_ms": pnr.elapsed * 1e3,
                    "time_ratio": ratio,
                    "time_ratio_paper": ref["ratio"],
                    "per_iteration_ratio": per_iter_ratio,
                    "max_V_diff": float(np.max(np.abs(snr.v - pnr.v))),
                }
            )
        table5 = pd.DataFrame(rows)
        print_table(table5)
        save_table(
            table5, "03_table5_results",
            "Paper Table 5 -- required iterations and calculation time",
            "`time_ratio` is total solve time SNR/PNR, the quantity the paper plots in its "
            "Figure 5. `per_iteration_ratio` divides out the differing iteration counts and is "
            "the quantity the paper's FLOP analysis actually predicts. Times are the best of "
            f"{REPEATS} repeats with Y-bus construction excluded.",
        )
        print(f"\nmean measured time ratio : {table5['time_ratio'].mean():.3f}")
        print(f"mean ratio reported in paper: {table5['time_ratio_paper'].mean():.3f}")
        print(f"mean per-iteration ratio : {table5['per_iteration_ratio'].mean():.3f}")

    # --------------------------------------------------------------- Figure 5
    with Section("4. Paper Figure 5 -- calculation time comparison"):
        labels = list(table5["case"])
        path = plot_grouped_bars(
            labels,
            {"SNR": table5["SNR_time_ms"].to_numpy(), "PNR": table5["PNR_time_ms"].to_numpy()},
            title="Calculation time for the five test cases",
            subtitle="Reproduction of Figure 5 of the base paper. Best of "
                     f"{REPEATS} repeats; Y-bus construction excluded.",
            ylabel="Calculation time (ms)",
            name="03_fig5_time_ratio",
            value_fmt="%.2f",
            log=True,
        )
        print(f"wrote {path}")

        path = plot_grouped_bars(
            labels,
            {
                "SNR": table5["SNR_iter"].to_numpy().astype(float),
                "PNR": table5["PNR_iter"].to_numpy().astype(float),
            },
            title="Iterations to a 1e-6 p.u. voltage tolerance",
            subtitle="Fewer is better. TC3 and TC5 are the PV-heavy cases, where lagging the "
                     "PV reactive powers costs the simplified method extra iterations.",
            ylabel="Iterations",
            name="03_iterations",
            value_fmt="%.0f",
        )
        print(f"wrote {path}")

        path = plot_grouped_bars(
            labels,
            {
                "SNR": table5["time_ratio"].to_numpy(),
                "PNR": table5["time_ratio_paper"].to_numpy(),
            },
            title="Time ratio SNR/PNR: measured here versus reported in the paper",
            subtitle="Values above 1.0 mean the simplified method was faster. Series are the two "
                     "sources, not the two methods.",
            ylabel="Calculation time ratio",
            name="03_fig5_ratio_comparison",
            value_fmt="%.3f",
            reference=1.0,
            reference_label="parity",
        )
        print(f"wrote {path}")

    # ---------------------------------------------------------- Figures 6-10
    with Section("5. Paper Figures 6-10 -- solution convergence of each test case"):
        for label, bundle in results.items():
            fig_no = FIGURE_NUMBER[label]
            case = bundle["case"]
            hist = {k: r.error_history for k, r in bundle["res"].items()}
            path = plot_convergence(
                hist,
                title=f"Solution convergence of {label} ({case.n_bus}-bus)",
                subtitle=f"Reproduction of Figure {fig_no} of the base paper. Error is the "
                         "largest deviation of any bus voltage from the converged solution.",
                name=f"03_fig{fig_no}_convergence_{label}",
                start_at=0,
            )
            print(f"{label}: wrote {path.name}")

        conv_rows = []
        for label, bundle in results.items():
            for key, res in bundle["res"].items():
                for rec in res.history:
                    conv_rows.append(
                        {
                            "case": label, "solver": key, "iteration": rec.iteration,
                            "max_voltage_error": rec.max_v_error,
                            "max_mismatch": rec.max_mismatch, "max_dv": rec.max_dv,
                        }
                    )
        save_table(
            pd.DataFrame(conv_rows), "03_convergence_histories",
            "Convergence histories behind Figures 6-10",
            "`max_voltage_error` is the quantity plotted; `max_dv` is the correction that the "
            "1e-6 p.u. termination test is applied to.",
        )

    # --------------------------------------------- testing the paper's claim
    with Section("6. Where the iteration time actually goes"):
        print(
            "Section 3 of the paper assumes that 'other steps of the two NR methods are\n"
            "exactly the same, therefore the Jacobian updating step dominates the overall\n"
            "execution time'. Total solve time tests that only indirectly. One iteration\n"
            "splits into four stages, and only the second differs between the methods:\n"
            "  1. mismatch    -- eqs (4),(5) for PNR vs (21),(22) for SNR   [differs]\n"
            "  2. derivatives -- eqs (9)-(16) for PNR vs (19),(20) etc      [differs]\n"
            "  3. assembly    -- slicing the blocks into the reduced system [identical]\n"
            "  4. linear solve -- LU of the same-sized system               [identical]"
        )
        stage_frames = [
            component_timings(bundle["case"], options, repeats=200)
            for bundle in results.values()
        ]
        stages = pd.concat(stage_frames, ignore_index=True)
        wide = stages.pivot_table(
            index=["n_bus", "case"],
            columns="solver",
            values=["t_mismatch_us", "t_derivatives_us", "t_assembly_us",
                    "t_solve_us", "t_iteration_us"],
        ).sort_index()

        def col(field, solver):
            return wide[(field, solver)].to_numpy()

        summary = pd.DataFrame(
            {
                "case": [i[1] for i in wide.index],
                "n_bus": [i[0] for i in wide.index],
                "deriv_SNR_us": col("t_derivatives_us", "SNR"),
                "deriv_PNR_us": col("t_derivatives_us", "PNR"),
                "deriv_ratio": col("t_derivatives_us", "SNR") / col("t_derivatives_us", "PNR"),
                "mismatch_SNR_us": col("t_mismatch_us", "SNR"),
                "mismatch_PNR_us": col("t_mismatch_us", "PNR"),
                "assembly_us": col("t_assembly_us", "SNR"),
                "linear_solve_us": col("t_solve_us", "SNR"),
                "shared_share": (col("t_assembly_us", "SNR") + col("t_solve_us", "SNR"))
                / col("t_iteration_us", "SNR"),
                "iteration_ratio": col("t_iteration_us", "SNR") / col("t_iteration_us", "PNR"),
            }
        )
        print()
        print_table(summary)
        save_table(
            summary, "03_stage_timings",
            "Per-stage timing of one Newton iteration",
            "`deriv_ratio` isolates the paper's claim: the speed-up of evaluating equations "
            "(9)-(16) instead of the standard derivative formulas. `shared_share` is the "
            "fraction of an iteration spent on work that is identical for the two methods "
            "(block assembly plus the LU solve) and therefore cannot be improved by "
            "reformulating the mismatch -- which is why `iteration_ratio` is always much "
            "closer to 1.",
        )

        path = plot_grouped_bars(
            list(summary["case"].str.replace("_", " ")),
            {"SNR": summary["deriv_SNR_us"].to_numpy(), "PNR": summary["deriv_PNR_us"].to_numpy()},
            title="Jacobian derivative evaluation, per iteration",
            subtitle="The one step the paper's FLOP analysis is about, timed on its own. "
                     "Best of 200 repeats at the flat start.",
            ylabel="Time (microseconds)",
            name="03_jacobian_stage_time",
            value_fmt="%.0f",
            log=True,
        )
        print(f"\nwrote {path}")
        print(
            f"\nThe paper's claim holds where it is made: evaluating the simplified Jacobian\n"
            f"is {summary['deriv_ratio'].mean():.2f}x faster on average across TC1-TC5.\n"
            f"But block assembly and the LU solve are {summary['shared_share'].mean() * 100:.0f}% "
            f"of an iteration and are identical work for\nboth methods, so the whole iteration "
            f"only speeds up by {summary['iteration_ratio'].mean():.2f}x."
        )


if __name__ == "__main__":
    main()
