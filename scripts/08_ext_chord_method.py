from __future__ import annotations
import os
# Time every method on single-threaded BLAS/LAPACK.  On small matrices a
# multi-threaded BLAS spends more time waking threads than computing, which
# made timings noisy and unfair to the substitution-only chord iterations.
# Must be set before NumPy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time  # noqa: E402
from functools import partial  # noqa: E402

import _bootstrap  # noqa: F401,E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from powerflow import EXTENSION_CASES, PAPER_TEST_CASES, build_ybus, load_case
from powerflow.benchmark import reference_solution, time_solver
from powerflow.plotting import INK_MUTED, INK_SECONDARY, SLOTS, SURFACE, _finish, style, use_paper_style
from powerflow.reporting import Section, print_table
from powerflow.solvers import SolverOptions, simplified_nr, standard_nr
from powerflow.solvers.chord_nr import factorize, substitute
from powerflow.solvers.chord_nr import solve as chord_solve

OUT_DIR = _bootstrap.RESULTS / "extension_chord"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

#: Chord iterations converge linearly, so they get more room than the paper's 50.
OPTIONS = SolverOptions(tol=1e-6, criterion="voltage", max_iter=150)

SOLVERS = {
    "SNR": standard_nr.solve,
    "PNR": simplified_nr.solve,
    "PNR-chord": partial(chord_solve, base="PNR"),
    "SNR-chord": partial(chord_solve, base="SNR"),
}
KEYS = list(SOLVERS)

#: Table 5 of the paper, averaged over its three machines (same values as script 03).
PAPER_TABLE5 = {
    "TC1": {"snr_iter": 5, "pnr_iter": 4, "ratio": 1.539},
    "TC2": {"snr_iter": 5, "pnr_iter": 5, "ratio": 1.459},
    "TC3": {"snr_iter": 7, "pnr_iter": 8, "ratio": 1.132},
    "TC4": {"snr_iter": 5, "pnr_iter": 5, "ratio": 1.329},
    "TC5": {"snr_iter": 4, "pnr_iter": 3, "ratio": 1.728},
}
FIGURE_NUMBER = {"TC1": 6, "TC2": 7, "TC3": 8, "TC4": 9, "TC5": 10}
REPEATS = {"3-bus": 25, "TC1": 25, "TC2": 25, "TC3": 25, "TC4": 25, "TC5": 25, "EX1": 7, "EX2": 3}

CASES = [("3-bus", "case3_saadat")] + list(PAPER_TEST_CASES) + list(EXTENSION_CASES)

#: Colour / marker / dash per method.  SNR and PNR keep the project-wide styles;
#: the two chord variants take the remaining slots of the validated palette
#: (all-pairs CVD check passes for these four hues on the figure surface).
STYLE = {
    "SNR": style("SNR"),
    "PNR": style("PNR"),
    "PNR-chord": {"color": SLOTS[6], "marker": "*", "linestyle": (0, (6, 2, 1, 2))},
    "SNR-chord": {"color": SLOTS[2], "marker": "P", "linestyle": (0, (2, 2))},
}
PAPER_COLOR = INK_MUTED


# ---------------------------------------------------------------- helpers
def save_csv(df: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def save_png(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def contraction_rate(dv: np.ndarray) -> float:
    """Median ratio of successive corrections over the last few steps (linear rate C)."""
    dv = np.asarray([d for d in dv if np.isfinite(d) and d > 0])
    if len(dv) < 3:
        return float("nan")
    ratios = dv[1:] / dv[:-1]
    return float(np.median(ratios[-4:]))


def bench(fn, target_s: float = 0.05) -> float:
    """Median wall time of one call of ``fn``, in microseconds."""
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if time.perf_counter() - t0 > target_s / 5 or n > 50_000:
            break
        n *= 2
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        samples.append((time.perf_counter() - t0) / n)
    return float(np.median(samples) * 1e6)


# ---------------------------------------------------------------- figures
def fig_convergence(histories: dict, title: str, subtitle: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key in KEYS:
        err = np.asarray(histories.get(key, []), dtype=float)
        idx = np.arange(len(err))           # iteration 0 = flat start, as in script 03
        mask = np.isfinite(err) & (err > 0)
        if mask.any():
            ax.semilogy(idx[mask], err[mask], label=key, markevery=max(mask.sum() // 12, 1),
                        **STYLE[key])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Maximum voltage error (p.u.)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="best")
    _finish(fig, ax, title, subtitle)
    save_png(fig, name)


def fig_grouped(categories, series: dict, colors: dict, title: str, subtitle: str,
                ylabel: str, name: str, log=False, reference=None, fmt="{:.2f}",
                missing_label="diverged") -> None:
    keys = list(series)
    x = np.arange(len(categories))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.3 * len(categories) + 2.8, 4.3))
    for i, key in enumerate(keys):
        vals = np.asarray(series[key], dtype=float)
        off = (i - (len(keys) - 1) / 2) * width
        shown = np.where(np.isfinite(vals), vals, 0.0)
        bars = ax.bar(x + off, shown, width * 0.96, label=key, color=colors[key],
                      edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.annotate(fmt.format(val), (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=INK_SECONDARY)
            else:
                ax.annotate(missing_label, (b.get_x() + b.get_width() / 2, 0),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=INK_SECONDARY, rotation=90)
    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=2)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=min(len(keys), 4), fontsize=8)
    _finish(fig, ax, title, subtitle)
    save_png(fig, name)


# ---------------------------------------------------------------- main
def main() -> None:
    use_paper_style()
    runs = {}

    with Section("1. Solving every case with the four methods (flat start, 1e-6 p.u. tolerance)"):
        for label, name in CASES:
            case = load_case(name)
            ybus = build_ybus(case)
            v_ref = reference_solution(case)
            per = {}
            for key, solver in SOLVERS.items():
                res, samples = time_solver(solver, case, OPTIONS, repeats=REPEATS[label], ybus=ybus)
                res.attach_reference(v_ref)
                per[key] = (res, float(np.median(samples)))
            runs[label] = {"case": case, "ybus": ybus, "v_ref": v_ref, "res": per}
            line = "  ".join(
                f"{k}={r.iterations}{'' if r.converged else '(fail)'}/{t * 1e3:.2f}ms"
                for k, (r, t) in per.items()
            )
            print(f"{label:6s} n={case.n_bus:4d} pv={len(case.pv):3d}  {line}")

    # ------------------------------------------------------------ Table 5
    with Section("2. Paper Table 5, extended with the chord method"):
        rows = []
        for label, b in runs.items():
            case, per = b["case"], b["res"]
            paper = PAPER_TABLE5.get(label, {})

            def ms(key):
                res, t = per[key]
                return t * 1e3 if res.converged else np.nan

            def it(key):
                res, _ = per[key]
                return res.iterations if res.converged else np.nan

            def err(key):
                res, _ = per[key]
                return float(np.max(np.abs(res.v - b["v_ref"]))) if res.converged else np.nan

            rows.append({
                "case": label, "system": case.name, "n_bus": case.n_bus, "n_pv": len(case.pv),
                "SNR_iter_paper": paper.get("snr_iter", np.nan),
                "PNR_iter_paper": paper.get("pnr_iter", np.nan),
                "SNR_iter": it("SNR"), "PNR_iter": it("PNR"),
                "PNR_chord_iter": it("PNR-chord"), "SNR_chord_iter": it("SNR-chord"),
                "SNR_ms": ms("SNR"), "PNR_ms": ms("PNR"),
                "PNR_chord_ms": ms("PNR-chord"), "SNR_chord_ms": ms("SNR-chord"),
                "ratio_SNR_over_PNR_paper": paper.get("ratio", np.nan),
                "ratio_SNR_over_PNR": ms("SNR") / ms("PNR"),
                "ratio_SNR_over_PNR_chord": ms("SNR") / ms("PNR-chord"),
                "speedup_PNR_chord_vs_PNR": ms("PNR") / ms("PNR-chord"),
                "speedup_SNR_chord_vs_SNR": ms("SNR") / ms("SNR-chord"),
                "PNR_chord_converged": per["PNR-chord"][0].converged,
                "SNR_chord_converged": per["SNR-chord"][0].converged,
                "PNR_chord_rate": contraction_rate(per["PNR-chord"][0].dv_history),
                "SNR_chord_rate": contraction_rate(per["SNR-chord"][0].dv_history),
                "LU_factorizations_PNR": it("PNR"),
                "LU_factorizations_PNR_chord": per["PNR-chord"][0].extras.get("n_factorizations"),
                "max_V_error_SNR": err("SNR"), "max_V_error_PNR": err("PNR"),
                "max_V_error_PNR_chord": err("PNR-chord"), "max_V_error_SNR_chord": err("SNR-chord"),
            })
        table5 = pd.DataFrame(rows)
        print_table(table5[["case", "n_bus", "n_pv", "SNR_iter", "PNR_iter", "PNR_chord_iter",
                            "SNR_chord_iter", "SNR_ms", "PNR_ms", "PNR_chord_ms",
                            "ratio_SNR_over_PNR_paper", "ratio_SNR_over_PNR",
                            "ratio_SNR_over_PNR_chord", "speedup_PNR_chord_vs_PNR"]])
        save_csv(table5, "08_table5_chord")

        tcs = [lbl for lbl, _ in PAPER_TEST_CASES]
        t5 = table5.set_index("case")
        fig_grouped(
            tcs,
            {"Paper: SNR / PNR (reported)": t5.loc[tcs, "ratio_SNR_over_PNR_paper"],
             "This work: SNR / PNR": t5.loc[tcs, "ratio_SNR_over_PNR"],
             "This work: SNR / PNR-chord": t5.loc[tcs, "ratio_SNR_over_PNR_chord"]},
            {"Paper: SNR / PNR (reported)": PAPER_COLOR, "This work: SNR / PNR": STYLE["PNR"]["color"],
             "This work: SNR / PNR-chord": STYLE["PNR-chord"]["color"]},
            "Calculation time ratio against standard NR",
            "Reproduction of the paper's Figure 5 with the chord method added. A ratio above 1 "
            "means faster than standard NR (dashed line).",
            "Time ratio  SNR time / method time", "08_fig5_time_ratio", reference=1.0,
        )
        order = [lbl for lbl, _ in CASES]
        fig_grouped(
            order,
            {k: t5.loc[order, col] for k, col in
             [("SNR", "SNR_iter"), ("PNR", "PNR_iter"), ("PNR-chord", "PNR_chord_iter"),
              ("SNR-chord", "SNR_chord_iter")]},
            {k: STYLE[k]["color"] for k in KEYS},
            "Iterations to reach the paper's 1e-6 p.u. tolerance",
            "Chord variants trade more (linearly convergent) iterations for cheaper ones. "
            "Bars marked 'diverged' did not converge.",
            "Iterations", "08_iterations", fmt="{:.0f}",
        )

    # ------------------------------------------------------ convergence curves
    with Section("3. Convergence histories (paper Figures 4 and 6-10, chord method added)"):
        rows = []
        for label, b in runs.items():
            for key, (res, _) in b["res"].items():
                for rec in res.history:
                    rows.append({"case": label, "method": key, "iteration": rec.iteration,
                                 "max_voltage_error": rec.max_v_error,
                                 "max_dv": rec.max_dv, "max_mismatch": rec.max_mismatch})
        conv = pd.DataFrame(rows)
        save_csv(conv, "08_convergence_histories")

        def curves(label):
            return {k: np.asarray(r.error_history, dtype=float)
                    for k, (r, _) in runs[label]["res"].items() if r.converged}

        fig_convergence(curves("3-bus"), "Solution convergence, 3-bus example",
                        "Paper Figure 4 with the chord method added. Both chord variants use the "
                        "Jacobian factorised at the flat start for every step.",
                        "08_fig4_convergence_3bus")
        for label, _ in PAPER_TEST_CASES:
            case = runs[label]["case"]
            fig_convergence(
                curves(label), f"Solution convergence of {label} ({case.n_bus}-bus)",
                f"Paper Figure {FIGURE_NUMBER[label]} with the chord method added. Curved lines "
                "are quadratic (Newton); straight lines are linear (chord).",
                f"08_fig{FIGURE_NUMBER[label]}_convergence_{label}",
            )
        print(f"wrote convergence figures for 3-bus and {', '.join(l for l, _ in PAPER_TEST_CASES)}")

    # ------------------------------------------------------------ step costs
    with Section("4. Cost of each step of one iteration (paper's equations, flat start)"):
        rows = []
        for label, b in runs.items():
            case, ybus = b["case"], b["ybus"]
            v = case.flat_start()
            s_eff = simplified_nr.effective_schedule(case, ybus, v)
            jac = simplified_nr.jacobian(case, ybus, v, s_eff)
            f = simplified_nr.mismatch_vector(case, ybus, v, s_eff)
            lu = factorize(jac)
            t_res = bench(lambda: simplified_nr.mismatch_vector(
                case, ybus, v, simplified_nr.effective_schedule(case, ybus, v)))
            t_jac = bench(lambda: simplified_nr.jacobian(case, ybus, v, s_eff))
            t_fac = bench(lambda: factorize(jac))
            t_sub = bench(lambda: substitute(lu, f))
            newton = t_res + t_jac + t_fac + t_sub
            chord = t_res + t_sub
            rows.append({"case": label, "n_bus": case.n_bus, "jacobian_size": jac.shape[0],
                         "residual_us": t_res, "jacobian_build_us": t_jac,
                         "lu_factorization_us": t_fac, "substitution_us": t_sub,
                         "newton_iteration_us": newton, "chord_iteration_us": chord,
                         "iteration_cost_ratio": newton / chord})
        costs = pd.DataFrame(rows)
        print_table(costs)
        save_csv(costs, "08_step_costs")

        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        order = costs["case"].tolist()
        parts = [("residual_us", "Residual F(x)", SLOTS[0]),
                 ("jacobian_build_us", "Build J", SLOTS[1]),
                 ("lu_factorization_us", "LU factorisation", SLOTS[6]),
                 ("substitution_us", "Forward/back substitution", SLOTS[2])]
        x = np.arange(len(order))
        w = 0.38
        newton_total = costs["newton_iteration_us"].to_numpy()
        for j, cols in enumerate([[p[0] for p in parts], ["residual_us", "substitution_us"]]):
            bottom = np.zeros(len(order))
            for col, lab, c in parts:
                vals = costs[col].to_numpy() if col in cols else np.zeros(len(order))
                share = 100 * vals / newton_total
                ax.bar(x + (j - 0.5) * w, share, w * 0.94, bottom=bottom, color=c,
                       edgecolor=SURFACE, linewidth=1.0, zorder=3,
                       label=lab if j == 0 else None)
                bottom += share
            totals = costs["newton_iteration_us" if j == 0 else "chord_iteration_us"].to_numpy()
            for xi, top, tot in zip(x, bottom, totals):
                ax.annotate(f"{'N' if j == 0 else 'C'}\n{tot:.0f}", (xi + (j - 0.5) * w, top),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.5, color=INK_SECONDARY)
        ax.set_xticks(x, order)
        ax.set_ylim(0, 118)
        ax.set_ylabel("Share of one Newton iteration (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", ncols=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
        _finish(fig, ax, "Where the time of one iteration goes",
                "N = one Newton iteration (always 100%), C = one chord iteration, both as a share "
                "of the Newton iteration. Numbers above the bars are microseconds. The chord "
                "method skips building J and its LU factorisation after the first step.")
        save_png(fig, "08_step_costs")

    # ------------------------------------------------------------ speed-up vs n
    with Section("5. Chord speed-up against system size"):
        sp_rows = table5[["case", "system", "n_bus", "n_pv", "speedup_PNR_chord_vs_PNR",
                          "speedup_SNR_chord_vs_SNR", "PNR_chord_converged",
                          "SNR_chord_converged"]].copy()
        print_table(sp_rows)
        save_csv(sp_rows, "08_speedup_vs_size")

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        n = sp_rows["n_bus"].to_numpy()
        for key, col in [("PNR-chord", "speedup_PNR_chord_vs_PNR"),
                         ("SNR-chord", "speedup_SNR_chord_vs_SNR")]:
            y = sp_rows[col].to_numpy(dtype=float)
            ok = np.isfinite(y)
            ax.semilogx(n[ok], y[ok], label=f"{key} vs its Newton method", **STYLE[key])
            for xi in n[~ok]:
                ax.annotate("diverged", (xi, 1.0), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK_SECONDARY)
                ax.plot([xi], [1.0], marker="x", color=STYLE[key]["color"], linestyle="none")
        ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        y_p = sp_rows["speedup_PNR_chord_vs_PNR"].to_numpy(dtype=float)
        for lab, note in [("TC3", "TC3: 10 PV buses,\n57 chord iterations"), ("EX1", "EX1 (118 bus)")]:
            i = sp_rows.index[sp_rows["case"] == lab][0]
            if np.isfinite(y_p[i]):
                ax.annotate(note, (n[i], y_p[i]), xytext=(8, -4), textcoords="offset points",
                            fontsize=7.5, color=INK_SECONDARY, va="top")
        ax.set_xlabel("Total number of buses, $n$")
        ax.set_ylabel("Speed-up  (Newton time / chord time)")
        ax.legend(loc="upper left", fontsize=8)
        _finish(fig, ax, "Chord method speed-up against system size",
                "Above the dashed line the chord method is faster than re-factorising every "
                "iteration.")
        save_png(fig, "08_speedup_vs_size")

    print(f"\nCSV tables in  {TABLE_DIR}\nPNG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
