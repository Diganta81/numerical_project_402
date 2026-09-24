"""
Extension 4 -- robustness under increasing load.

Neither the base paper nor the project proposal tests what happens away from the
comfortable operating point every IEEE case ships with, but the proposal's
problem statement raises "convergence instability" as a motivation, so it is
worth measuring.  Every injection is scaled by a loading factor ``lambda`` and
each method is re-solved from a flat start.  As ``lambda`` grows the network
approaches its voltage-stability limit, the Jacobian becomes ill-conditioned,
and the methods start to fail -- at different points, which is the result.

Two things are reported:

* the **iteration count surface** over ``(method, lambda)``, and
* the **largest loading factor** each method still solves, which is a practical
  robustness score and, for the Newton methods, an estimate of the true
  loadability limit of the network.

All injections are scaled, generation as well as load, so that the power balance
stays intact and the limit found is the network's own rather than the slack bus
running out of headroom.

The sweep separates two very different kinds of failure.  On the 30- and 57-bus
systems every method fails at exactly the same loading factor -- that is the
network's voltage-stability limit, and no formulation can be blamed for it.  On
the 118-bus system, the system with 53 PV buses, they part company sharply:
standard NR, the rectangular method and FDLF all reach 3.0x nominal, while the
paper's formulation gives up at 1.6x.  That is not a physical limit, it is a
loss of convergence, and it is the same lagged-PV mechanism that costs the
method iterations in script 04 -- the fixed point it layers on top of Newton has
a contraction factor that degrades as the system is stressed.

The augmented variant recovers part of that (2.0x) but not all of it, which is
worth noting honestly: making the PV reactive powers Newton unknowns fixes the
*convergence rate* completely, and the *convergence basin* only partly.  A cheap
Jacobian and a fast local rate do not by themselves buy robustness.

Outputs
-------
``results/tables/07_robustness.{csv,md}``        full sweep
``results/tables/07_max_loading.{csv,md}``       loadability score per method
``results/figures/07_iterations_heatmap_*.png``
``results/figures/07_max_loading.png``
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import load_case
from powerflow.plotting import plot_grouped_bars, plot_heatmap, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.solvers import SOLVERS, SolverOptions

CASES = ["case30_ieee", "case57_ieee", "case118_ieee"]
KEYS = ["SNR", "PNR", "PNR+", "RCI", "FDLF-XB"]
LAMBDAS = np.round(np.arange(1.0, 5.01, 0.2), 2)
MAX_ITER = 40
STEP = 0.2


def main() -> None:
    use_paper_style()
    options = SolverOptions(tol=1e-6, criterion="voltage", max_iter=MAX_ITER)

    records = []
    with Section("1. Loading sweep"):
        for name in CASES:
            base = load_case(name)
            print(f"\n{base.summary()}")
            for lam in LAMBDAS:
                case = base.scaled(lam)
                line = [f"  lambda={lam:4.1f}"]
                for key in KEYS:
                    try:
                        res = SOLVERS[key](case, options)
                        ok = res.converged and np.all(np.isfinite(res.v))
                        iters = res.iterations if ok else np.nan
                    except Exception:
                        ok, iters = False, np.nan
                    records.append(
                        {
                            "case": name, "n_bus": base.n_bus, "load_factor": lam,
                            "solver": key, "converged": bool(ok), "iterations": iters,
                        }
                    )
                    line.append(f"{key}={'--' if not ok else int(iters):>3}")
                if abs(round(lam * 5) - lam * 5) < 1e-9:   # print at whole and half steps
                    print("  ".join(line))

    df = pd.DataFrame(records)
    save_table(
        df, "07_robustness", "Iterations versus loading factor",
        f"Every injection scaled by `load_factor`, flat start, 1e-6 p.u. voltage tolerance, "
        f"{MAX_ITER}-iteration budget. Blank iterations mean the method failed to converge.",
    )

    # ------------------------------------------------------------- heatmaps
    with Section("2. Iteration-count surfaces"):
        for name in CASES:
            sub = df[df["case"] == name]
            grid = sub.pivot_table(index="solver", columns="load_factor",
                                   values="iterations", dropna=False).reindex(KEYS)
            tag = name.replace("case", "").replace("_ieee", "")
            path = plot_heatmap(
                grid.to_numpy(),
                row_labels=KEYS,
                col_labels=[f"{c:g}" for c in grid.columns],
                title=f"Iterations versus loading factor, IEEE {tag}-bus",
                subtitle=f"Blank cells did not converge within {MAX_ITER} iterations. "
                         "Darker means more iterations.",
                cbar_label="Iterations",
                name=f"07_iterations_heatmap_{tag}",
            )
            print(f"wrote {path.name}")

    # --------------------------------------------------------- loadability
    with Section("3. Largest loading factor each method still solves"):
        rows = []
        for name in CASES:
            for key in KEYS:
                sub = df[(df["case"] == name) & (df["solver"] == key)].sort_values("load_factor")
                ok = sub[sub["converged"]]["load_factor"]
                fail = sub[~sub["converged"]]["load_factor"]
                # Two measures, because they can differ: a method may fail at one
                # loading factor and succeed at a larger one, which is itself a
                # robustness finding and should not be averaged away.
                contiguous = (fail.min() - STEP) if len(fail) else (ok.max() if len(ok) else np.nan)
                highest = ok.max() if len(ok) else np.nan
                rows.append(
                    {
                        "case": name,
                        "solver": key,
                        "max_load_contiguous": contiguous,
                        "max_load_any": highest,
                        "n_failures": int((~sub["converged"]).sum()),
                        "censored": bool(len(fail) == 0),
                    }
                )
        limits = pd.DataFrame(rows)
        wide = limits.pivot_table(index="case", columns="solver", values="max_load_contiguous")
        wide = wide.reindex(columns=KEYS).reindex(CASES)
        print_table(limits)
        save_table(
            limits, "07_max_loading",
            "Largest loading factor solved, by method",
            "`max_load_contiguous` is the last factor before the *first* failure -- the "
            "practically usable range. `max_load_any` is the largest factor solved at all; where "
            "the two differ, the method failed at some loading and then recovered at a heavier "
            "one, which means it lost convergence rather than hitting a physical limit. "
            f"`censored` marks a method that never failed inside the sweep (up to {LAMBDAS[-1]:g}), "
            "so its true limit is higher than reported.",
        )
        print()
        print_table(wide.reset_index())

        path = plot_grouped_bars(
            [c.replace("case", "IEEE ").replace("_ieee", "-bus") for c in wide.index],
            {k: wide[k].to_numpy() for k in KEYS},
            title="Largest loading factor solved from a flat start",
            subtitle="Higher is more robust. All methods target the same physical limit, so a "
                     "shorter bar means the method lost convergence before the network did.",
            ylabel="Loading factor",
            name="07_max_loading",
            value_fmt="%.1f",
            reference=1.0,
            reference_label="nominal load",
        )
        print(f"\nwrote {path}")

        print("\nloading headroom relative to standard NR (negative = gave up earlier):")
        for case in wide.index:
            deltas = "  ".join(f"{k}={wide.loc[case, k] - wide.loc[case, 'SNR']:+.1f}"
                               for k in KEYS if k != "SNR")
            print(f"   {case:<16s} {deltas}")

        shaky = limits[limits["max_load_any"] > limits["max_load_contiguous"]]
        if len(shaky):
            print("\nmethods that failed at one loading and recovered at a heavier one:")
            for r in shaky.itertuples():
                print(f"   {r.solver} on {r.case}: first failure at "
                      f"{r.max_load_contiguous + STEP:.1f}, yet still solved "
                      f"{r.max_load_any:.1f} -- lost convergence, not a physical limit")


if __name__ == "__main__":
    main()
