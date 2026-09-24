from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import build_ybus, load_case
from powerflow.plotting import plot_convergence, use_paper_style
from powerflow.reporting import Section, print_table, save_table, voltage_table
from powerflow.solvers import SolverOptions, simplified_nr, standard_nr

PAPER_YBUS_MAG = np.array(
    [[53.8517, 22.3607, 31.6228], [22.3607, 58.1378, 35.7771], [31.6228, 35.7771, 67.2310]]
)
PAPER_YBUS_ANG = np.array(
    [[-68.20, 116.57, 108.43], [116.57, -63.43, 116.57], [108.43, 116.57, -67.25]]
)
PAPER_Q3_INITIAL = 1.0192

PAPER_ITERATIONS = [
    {
        "mismatch": np.array([-2.8600, 1.3831, 0.2200]),
        "jacobian": np.array(
            [[54.50, -33.28, 22.00], [-32.00, 63.50, -16.00], [30.00, -16.64, -49.50]]
        ),
        "dx": np.array([-0.0460, -0.0088, -0.0294]),
        "x_next": np.array([-0.0460, -0.0088, 0.9706]),
    },
    {
        "mismatch": np.array([-0.0408, 0.0241, -0.0826]),
        "jacobian": np.array(
            [
                [54.3425, -33.4251, 19.4624],
                [-31.7416, 63.2414, -14.5117],
                [26.8874, -16.3466, -50.2945],
            ]
        ),
        "dx": np.array([-0.0011, 0.0001, 0.0010]),
        "x_next": np.array([-0.0471, -0.0087, 0.9716]),
    },
    {
        "mismatch": np.array([6.71e-5, 4.03e-4, -6.3e-6]),
        "jacobian": np.array(
            [
                [54.4238, -33.4238, 19.4172],
                [-31.7908, 63.2908, -14.4766],
                [26.8492, -16.3493, -50.3215],
            ]
        ),
        "dx": np.array([7.23e-6, 1.02e-5, 6.80e-7]),
        "x_next": np.array([-0.04709, -0.00869, 0.97168]),
    },
]

PAPER_TABLE2 = {
    "V2 magnitude (p.u.)": (0.97168, 0.97168),
    "V2 angle (deg)": (-2.696, -2.698),
    "V3 magnitude (p.u.)": (1.04, 1.04),
    "V3 angle (deg)": (-0.4988, -0.4979),
    "Q3 (p.u.)": (1.4617, 1.4618),
}

ROW_LABELS = ["G2", "G3", "H2"]
COL_LABELS = ["d(delta_2)", "d(delta_3)", "d|V_2|"]


def main() -> None:
    use_paper_style()
    case = load_case("case3_saadat")
    ybus = build_ybus(case)

    with Section("1. Bus admittance matrix (Section 4 of the paper)"):
        mag, ang = np.abs(ybus), np.degrees(np.angle(ybus))
        rows = []
        for i in range(3):
            for j in range(i, 3):
                rows.append(
                    {
                        "entry": f"Y{i + 1}{j + 1}",
                        "computed_mag": mag[i, j],
                        "paper_mag": PAPER_YBUS_MAG[i, j],
                        "computed_ang_deg": ang[i, j],
                        "paper_ang_deg": PAPER_YBUS_ANG[i, j],
                        "abs_err_mag": abs(mag[i, j] - PAPER_YBUS_MAG[i, j]),
                    }
                )
        ybus_df = pd.DataFrame(rows)
        print_table(ybus_df)
        save_table(
            ybus_df, "01_ybus_polar", "Y-bus versus the matrix printed in Section 4",
            "Line impedances were back-derived from the printed Y-bus; agreement is "
            "to the 4 decimal places the paper prints.",
        )
        print(f"\nlargest magnitude discrepancy: {ybus_df['abs_err_mag'].max():.2e}")

    with Section("2. Iteration-by-iteration reproduction of the PNR example"):
        v = case.flat_start()
        vm, va = np.abs(v), np.angle(v)
        pvpq, pq = case.pvpq, case.pq
        n_ang = len(pvpq)

        s_eff0 = simplified_nr.effective_schedule(case, ybus, v)
        q3 = s_eff0[2].imag
        print(f"Initial Q_cal,3 (paper eq. 22) = {q3:.4f} p.u.   paper: {PAPER_Q3_INITIAL}")
        print(f"S_sch,3 = {s_eff0[2].real:.4f} + j{q3:.4f} p.u.   paper: 2.0 + j1.0192\n")

        check_rows = []
        for step, expected in enumerate(PAPER_ITERATIONS, start=1):
            s_eff = simplified_nr.effective_schedule(case, ybus, v)
            f = simplified_nr.mismatch_vector(case, ybus, v, s_eff)
            jac = simplified_nr.jacobian(case, ybus, v, s_eff)
            dx = np.linalg.solve(jac, f)

            print(f"--- iteration {step} " + "-" * 58)
            print("current mismatch  [G2, G3, H2]:")
            print(f"   computed : {np.array2string(f, precision=4, suppress_small=False)}")
            print(f"   paper    : {np.array2string(expected['mismatch'], precision=4)}")
            print("Jacobian (rows G2,G3,H2 x cols d2,d3,|V2|):")
            for r in range(3):
                print(
                    f"   computed : [{jac[r, 0]:9.4f} {jac[r, 1]:9.4f} {jac[r, 2]:9.4f}]"
                    f"   paper: [{expected['jacobian'][r, 0]:9.4f}"
                    f" {expected['jacobian'][r, 1]:9.4f} {expected['jacobian'][r, 2]:9.4f}]"
                )
            print("correction vector:")
            print(f"   computed : {np.array2string(dx, precision=6)}")
            print(f"   paper    : {np.array2string(expected['dx'], precision=6)}")

            for name, got, want in (
                ("mismatch", f, expected["mismatch"]),
                ("jacobian", jac.ravel(), expected["jacobian"].ravel()),
                ("dx", dx, expected["dx"]),
            ):
                scale = max(np.max(np.abs(want)), 1e-12)
                check_rows.append(
                    {
                        "iteration": step,
                        "quantity": name,
                        "max_abs_diff": float(np.max(np.abs(got - want))),
                        "max_rel_diff": float(np.max(np.abs(got - want)) / scale),
                    }
                )

            va[pvpq] += dx[:n_ang]
            vm[pq] += dx[n_ang:]
            v = vm * np.exp(1j * va)
            x_now = np.array([va[1], va[2], vm[1]])
            print(f"updated state     : {np.array2string(x_now, precision=5)}")
            print(f"paper             : {np.array2string(expected['x_next'], precision=5)}\n")

        checks = pd.DataFrame(check_rows)
        print_table(checks)
        save_table(
            checks, "01_iterations",
            "Per-iteration agreement with the values printed in Section 4",
            "`max_rel_diff` is normalised by the largest entry of the paper's own vector or "
            "matrix, so it absorbs the paper's 4-decimal rounding.",
        )
        print(f"\nlargest relative discrepancy over all three iterations: "
              f"{checks['max_rel_diff'].max():.2e}")

    with Section("3. Paper Table 2 -- converged solution, both methods"):
        opts = SolverOptions(tol=1e-10)
        res_snr = standard_nr.solve(case, opts, ybus)
        res_pnr = simplified_nr.solve(case, opts, ybus)

        q_snr = res_snr.injections(ybus).imag[2]
        q_pnr = res_pnr.injections(ybus).imag[2]
        computed = {
            "V2 magnitude (p.u.)": (res_snr.vm[1], res_pnr.vm[1]),
            "V2 angle (deg)": (res_snr.va_deg[1], res_pnr.va_deg[1]),
            "V3 magnitude (p.u.)": (res_snr.vm[2], res_pnr.vm[2]),
            "V3 angle (deg)": (res_snr.va_deg[2], res_pnr.va_deg[2]),
            "Q3 (p.u.)": (q_snr, q_pnr),
        }
        table2 = pd.DataFrame(
            [
                {
                    "item": k,
                    "SNR (this work)": computed[k][0],
                    "SNR (paper)": PAPER_TABLE2[k][0],
                    "PNR (this work)": computed[k][1],
                    "PNR (paper)": PAPER_TABLE2[k][1],
                }
                for k in PAPER_TABLE2
            ]
        )
        print_table(table2)
        save_table(
            table2, "01_table2_solution", "Paper Table 2 -- power-flow solution for the example",
            "The paper's third-iteration values are quoted; this work iterates to a 1e-10 p.u. "
            "voltage tolerance, which is why the angles differ in the fourth decimal.",
        )

        print()
        print(voltage_table(case, res_pnr.v).to_string(index=False))

    with Section("4. Paper Figure 4 -- solution convergence"):
        hist = {
            "SNR": res_snr.mismatch_history,
            "PNR": res_pnr.mismatch_history,
        }
        for key, h in hist.items():
            print(f"{key}: " + "  ".join(f"{x:.3e}" for x in h))
        path = plot_convergence(
            hist,
            title="Solution convergence, 3-bus example",
            subtitle="Reproduction of Figure 4 of the base paper. SNR residual is a power "
                     "mismatch (p.u. MVA); PNR residual is a current mismatch (p.u. A).",
            ylabel="Maximum mismatch (p.u.)",
            name="01_fig4_convergence",
            start_at=0,
        )
        print(f"\nwrote {path}")

        conv = pd.DataFrame(
            {
                "iteration": range(max(len(hist["SNR"]), len(hist["PNR"]))),
                "SNR_max_power_mismatch": pd.Series(hist["SNR"]),
                "PNR_max_current_mismatch": pd.Series(hist["PNR"]),
            }
        )
        save_table(conv, "01_fig4_convergence",
                   "Figure 4 data -- maximum mismatch per iteration")


if __name__ == "__main__":
    main()
