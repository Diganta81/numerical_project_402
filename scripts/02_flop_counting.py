"""
Reproduce Section 3 of the base paper: floating-point operation counting.

Produces the paper's Table 1 and Figures 1 and 2 exactly as published, then puts
an audited element-by-element recount of the *same* formulas beside them.  The
recount matters: the paper's headline claim of a cost linear in the bus count
cannot hold for a dense Jacobian, and seeing by how much it is off is the point
of the exercise.  See :mod:`powerflow.flops` for the derivation of each model.

Outputs
-------
``results/tables/02_table1_flops.{csv,md}``      the paper's Table 1
``results/tables/02_flop_models.{csv,md}``       paper vs audited vs sparse
``results/figures/02_fig1_jacobian_flops.png``   the paper's Figure 1
``results/figures/02_fig2_mismatch_flops.png``   the paper's Figure 2
``results/figures/02_flop_models.png``           the three models together
``results/figures/02_predicted_speedup.png``     predicted cost ratio vs n
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from powerflow import EXTENSION_CASES, PAPER_TEST_CASES, build_ybus, load_case
from powerflow.flops import (
    BREAK_EVEN_DEGREE,
    DIAGONAL_COST,
    OFF_DIAGONAL_COST,
    audited_jacobian,
    flop_dataframe,
    paper_figure1,
    paper_figure2,
    paper_table1,
    ratio_vs_degree,
    sparse_jacobian,
)
from powerflow.plotting import plot_flops, plot_scaling, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.ybus import sparsity

N_GRID = np.unique(np.round(np.logspace(0.4, 4, 60)).astype(int))
N_GRID = N_GRID[N_GRID >= 3]


def main() -> None:
    use_paper_style()

    # ------------------------------------------------------------- Table 1
    with Section("1. Paper Table 1 -- FLOPs required to build the Jacobian"):
        rows = []
        for n in (5, 6, 24, 30, 57, 118, 300):
            tbl = paper_table1(n)
            for blk in ("J1", "J2", "J3", "J4"):
                rows.append(
                    {
                        "n": n,
                        "sub_matrix": blk,
                        "SNR_diagonal": tbl["SNR"][blk]["diagonal"],
                        "SNR_off_diagonal": tbl["SNR"][blk]["off_diagonal"],
                        "SNR_total": tbl["SNR"][blk]["total"],
                        "PNR_diagonal": tbl["PNR"][blk]["diagonal"],
                        "PNR_off_diagonal": tbl["PNR"][blk]["off_diagonal"],
                        "PNR_total": tbl["PNR"][blk]["total"],
                    }
                )
            rows.append(
                {
                    "n": n, "sub_matrix": "overall",
                    "SNR_diagonal": np.nan, "SNR_off_diagonal": np.nan,
                    "SNR_total": tbl["SNR"]["overall"]["total"],
                    "PNR_diagonal": np.nan, "PNR_off_diagonal": np.nan,
                    "PNR_total": tbl["PNR"]["overall"]["total"],
                }
            )
        table1 = pd.DataFrame(rows)
        print_table(table1[table1["n"].isin([5, 30, 118])])
        save_table(
            table1, "02_table1_flops", "Paper Table 1 -- multiplication counts per iteration",
            "Symbolic counts from the paper: standard NR totals 10n^2 + O(n) and the proposed "
            "method totals 22(n-2). Evaluated here at each test-system size.",
        )
        n = 118
        t = paper_table1(n)
        print(f"\nAt n = {n}: SNR {t['SNR']['overall']['total']:,.0f} FLOPs, "
              f"PNR {t['PNR']['overall']['total']:,.0f} FLOPs  "
              f"(paper's claimed ratio {t['SNR']['overall']['total'] / t['PNR']['overall']['total']:,.0f}x)")

    # ------------------------------------------------------- Figures 1 and 2
    with Section("2. Paper Figures 1 and 2"):
        snr_j, pnr_j = paper_figure1(N_GRID)
        path = plot_flops(
            N_GRID, {"SNR": snr_j, "PNR": pnr_j},
            title="FLOPs per iteration to update the Jacobian matrix",
            ylabel="FLOPs (multiplications)",
            name="02_fig1_jacobian_flops",
        )
        print(f"wrote {path}   (SNR = 10n^2, PNR = 22(n-2), as published)")

        snr_m, pnr_m = paper_figure2(N_GRID)
        path = plot_flops(
            N_GRID, {"SNR": snr_m, "PNR": pnr_m},
            title="FLOPs per iteration to update the mismatch vectors",
            ylabel="FLOPs (multiplications)",
            name="02_fig2_mismatch_flops",
        )
        print(f"wrote {path}   (SNR = 6n, PNR = 4n + 4, as published)")

    # ------------------------------------------------ audited vs paper model
    with Section("3. Audited recount of the same formulas"):
        print(
            "The paper counts the off-diagonal work of the proposed method once per row\n"
            "rather than once per entry, which is what makes its Table 1 come out linear in n.\n"
            "Equations (9), (11), (13) and (15) each carry a distinct Y_ki, so a dense\n"
            "Jacobian needs at least one multiplication per entry and the true cost is O(n^2)\n"
            "for BOTH methods. Recounting element by element, with the same common\n"
            "sub-expression reuse granted to each:\n"
            f"    off-diagonal pair : SNR {OFF_DIAGONAL_COST['SNR']} mults, "
            f"PNR {OFF_DIAGONAL_COST['PNR']} mults\n"
            f"    per bus (diagonal): SNR {DIAGONAL_COST['SNR']} mults, "
            f"PNR {DIAGONAL_COST['PNR']} mults\n"
            "The proposed method is genuinely cheaper -- by a constant factor approaching\n"
            "9/5 = 1.8 for a dense Jacobian, not by a factor of n."
        )
        models = flop_dataframe([5, 6, 24, 30, 57, 118, 300, 1000])
        print()
        print_table(models)
        save_table(
            models, "02_flop_models",
            "Per-iteration multiplication counts: as published versus audited",
            "`paper_*` columns are the counts as printed in the paper. `audit_*` columns "
            "recount the identical formulas entry by entry. For the Jacobian alone the audited "
            "ratio tends to 9/5 = 1.80; including the mismatch vectors (which save only 4:3) "
            "brings the combined figure to about 1.62. The paper measured 1.13-1.73 in its "
            "Table 5, with 1.728 on its largest (57-bus) case.",
        )
        jac_only = ratio_vs_degree(999)
        print(f"\naudited ratio at n=1000: {models.iloc[-1]['audit_ratio']:.3f} overall "
              f"({jac_only:.3f} for the Jacobian alone, asymptote 9/5 = 1.800; the mismatch "
              f"vectors only save 4:3 and pull the combined figure down)")

        audit_s = np.array([audited_jacobian("SNR", int(k)).total for k in N_GRID], dtype=float)
        audit_p = np.array([audited_jacobian("PNR", int(k)).total for k in N_GRID], dtype=float)
        path = plot_flops(
            N_GRID,
            {
                "SNR": snr_j + snr_m,
                "PNR": pnr_j + pnr_m,
                "SNR audited": audit_s,
                "PNR audited": audit_p,
            },
            title="Per-iteration cost: as published versus audited",
            ylabel="FLOPs (multiplications)",
            name="02_flop_models",
        )
        print(f"wrote {path}")

    # ----------------------------------------- sparse model on the real cases
    with Section("4. Sparse model on the actual test systems"):
        rows = []
        for label, name in PAPER_TEST_CASES + EXTENSION_CASES:
            case = load_case(name)
            stats = sparsity(build_ybus(case, sparse=True))
            s = sparse_jacobian("SNR", case.n_bus, stats["nnz"])
            p = sparse_jacobian("PNR", case.n_bus, stats["nnz"])
            d_s = audited_jacobian("SNR", case.n_bus)
            rows.append(
                {
                    "label": label,
                    "case": name,
                    "n": case.n_bus,
                    "Ybus_nnz": stats["nnz"],
                    "Ybus_density": stats["density"],
                    "dense_SNR": d_s.total,
                    "sparse_SNR": s.total,
                    "sparse_PNR": p.total,
                    "sparse_ratio": s.total / p.total,
                    "dense_over_sparse": d_s.total / s.total,
                }
            )
        sparse_df = pd.DataFrame(rows)
        print_table(sparse_df)
        save_table(
            sparse_df, "02_sparse_flops",
            "Audited FLOP counts over the true Y-bus sparsity pattern",
            "Real networks are extremely sparse, so the quadratic term never materialises in "
            "practice: `dense_over_sparse` shows how much work a dense implementation wastes. "
            "The 7/5 advantage of the proposed method survives, because it is a per-entry "
            "saving and applies to whichever entries are actually stored.",
        )

        path = plot_scaling(
            sparse_df["n"].to_numpy(),
            {"SNR": sparse_df["sparse_SNR"], "PNR": sparse_df["sparse_PNR"]},
            title="Audited FLOPs per iteration over the true Y-bus sparsity",
            ylabel="FLOPs (multiplications)",
            subtitle="Once sparsity is exploited the per-iteration cost grows linearly in n for "
                     "both methods, and the two curves nearly coincide.",
            name="02_sparse_flops",
            point_labels=list(sparse_df["label"]),
        )
        print(f"\nwrote {path}")

    # ------------------------------------------- where the advantage lives
    with Section("5. How the advantage depends on network density"):
        print(
            "The simplified Jacobian is cheaper per matrix entry "
            f"({OFF_DIAGONAL_COST['SNR']} vs {OFF_DIAGONAL_COST['PNR']} multiplications) but\n"
            f"dearer per bus ({DIAGONAL_COST['SNR']} vs {DIAGONAL_COST['PNR']}), because its "
            "closed-form diagonals replace row sums\n"
            "that the standard method accumulates for free. With average nodal degree d the\n"
            "cost ratio is (9d + 4) / (5d + 13), so:\n"
            f"    break-even at d = {BREAK_EVEN_DEGREE:.2f}\n"
            "    dense formulation (d = n-1) tends to 9/5 = 1.80"
        )
        degrees = np.concatenate([np.linspace(1.0, 20.0, 80), np.logspace(1.3, 3, 40)])
        actual_d = (sparse_df["Ybus_nnz"] - sparse_df["n"]) / sparse_df["n"]
        density_df = pd.DataFrame(
            {
                "label": sparse_df["label"],
                "case": sparse_df["case"],
                "n": sparse_df["n"],
                "avg_degree": actual_d,
                "predicted_ratio_sparse": ratio_vs_degree(actual_d),
                "predicted_ratio_dense": ratio_vs_degree(sparse_df["n"] - 1),
            }
        )
        print()
        print_table(density_df)
        save_table(
            density_df, "02_density_ratio",
            "Predicted Jacobian cost ratio SNR/PNR versus network density",
            "`predicted_ratio_dense` applies to a dense implementation such as the paper's; "
            "`predicted_ratio_sparse` applies once Y-bus sparsity is exploited. The paper "
            "measured 1.728 on the 57-bus system, against a dense prediction of "
            f"{ratio_vs_degree(56):.3f}.",
        )

        fig_df = pd.DataFrame({"degree": degrees, "ratio": ratio_vs_degree(degrees)})
        path = plot_scaling(
            fig_df["degree"].to_numpy(),
            {"SNR": fig_df["ratio"].to_numpy()},
            title="Predicted Jacobian cost ratio, standard / simplified",
            ylabel="Cost ratio (>1 favours the simplified method)",
            xlabel="Average off-diagonal entries per row, $d$",
            subtitle=f"Break-even at d = {BREAK_EVEN_DEGREE:.2f}; dense asymptote 9/5 = 1.80. "
                     "Real transmission grids sit at d = 2.5-3.7, only just above break-even.",
            name="02_ratio_vs_degree",
            loglog=False,
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
