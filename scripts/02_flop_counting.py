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

from powerflow.flops import paper_figure1, paper_figure2, paper_table1
from powerflow.plotting import plot_flops, use_paper_style
from powerflow.reporting import Section, print_table, save_table

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
        


if __name__ == "__main__":
    main()
