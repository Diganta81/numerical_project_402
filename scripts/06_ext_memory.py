"""
Extension 3 -- memory overhead.

The project proposal asks for memory overhead to be compared alongside iteration
counts and runtime.  Memory is reported at two levels, because they answer
different questions:

**Structural** -- the size and sparsity of the linear system each method has to
store and factorise.  This is deterministic, implementation-independent and is
the number that matters when sizing a solver for a large grid.  The two NR
variants solve systems of *identical* dimension ``(n-1) + n_pq``: the base
paper's reformulation changes what goes into the Jacobian, not how big it is.
The augmented PV variant adds one row and column per PV bus, the rectangular
method roughly doubles the system, and FDLF replaces one matrix with two smaller
constant ones that are factorised once instead of every iteration.

**Measured** -- peak Python heap growth during a solve, from :mod:`tracemalloc`.

The headline answer to the proposal's question is a null result, and a clean
one: **the standard and simplified NR methods have the same memory footprint**,
structurally *and* as measured (within about 1% on every system tested).  This
is not a surprise once looked at directly -- the peak is set by the dense LU
factorisation of the reduced Jacobian, which is the same size for both -- but it
is worth establishing rather than assuming.  Reformulating the mismatch is a
*time* optimisation; it buys nothing in space.

Where the memory differences actually are:

* **FDLF** uses about a quarter of the memory, because it never forms a complex
  ``n x n`` derivative matrix at all -- only two real, constant, pre-factorised
  ones.
* **RCI** uses less than either NR variant on the large systems despite solving a
  *larger* system, because it works in real arithmetic on ``G`` and ``B`` rather
  than in complex arithmetic.
* **Sparse storage** dwarfs all of it, and is available to every method equally.

Outputs
-------
``results/tables/06_memory.{csv,md}``             measured + structural, all methods
``results/tables/06_ybus_sparsity.{csv,md}``      dense vs sparse Y-bus storage
``results/figures/06_peak_memory.png``
``results/figures/06_jacobian_size.png``
``results/figures/06_dense_vs_sparse_memory.png``
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import pandas as pd

from powerflow import load_case
from powerflow.benchmark import benchmark_suite
from powerflow.plotting import plot_grouped_bars, plot_scaling, use_paper_style
from powerflow.reporting import Section, print_table, save_table
from powerflow.solvers import SolverOptions
from powerflow.ybus import build_ybus, sparsity

CASES = ["case5_stagg", "case6ww", "case24_ieee_rts", "case30_ieee",
         "case57_ieee", "case118_ieee", "case300_ieee"]
KEYS = ["SNR", "PNR", "PNR+", "RCI", "FDLF-XB"]


def main() -> None:
    use_paper_style()
    cases = [load_case(n) for n in CASES]
    options = SolverOptions(tol=1e-6, max_iter=60)

    # --------------------------------------------------- measured + structural
    with Section("1. Peak heap use and linear-system size, dense storage"):
        df = benchmark_suite(cases, KEYS, options, repeats=3, with_memory=True)
        view = df[["case", "n_bus", "solver", "iterations", "peak_mem_kb",
                   "jac_rows", "jac_nnz", "jac_bytes"]].copy()
        view["jac_kb"] = view.pop("jac_bytes") / 1024
        print_table(view)
        save_table(
            df, "06_memory", "Memory overhead by method (dense storage)",
            "`peak_mem_kb` is the peak tracemalloc increment over one solve. `jac_rows`/`jac_kb` "
            "are the dimension and dense footprint of the linear system factorised each "
            "iteration. SNR and PNR match on both counts -- the paper's reformulation changes "
            "the Jacobian's contents, not its size, and the peak is set by the dense LU of that "
            "same-sized system. FDLF is far lower because it never forms a complex n x n "
            "derivative matrix.",
        )

        wide = df.pivot_table(index="n_bus", columns="solver", values="peak_mem_kb")
        print("\npeak heap increment by system size (KiB):")
        print_table(wide.reset_index())
        # Below ~24 buses the totals are a few KiB and are dominated by one-off
        # allocations, so the ratios there are noise rather than signal.
        big = wide[wide.index >= 24]
        drift = (big["PNR"] / big["SNR"] - 1.0).abs().max() * 100
        print(
            f"\nFrom 24 buses upward, PNR differs from SNR by at most {drift:.1f}%: the two"
            f"\nformulations have the same memory footprint, because the peak is set by the dense"
            f"\nLU of a linear system they solve at identical size. That is the answer to the"
            f"\nproposal's question -- reformulating the mismatch is a time optimisation, not a"
            f"\nspace one. (Below 24 buses the totals are a few KiB and the ratios are noise.)"
        )
        print(
            f"\nOn those same systems FDLF uses {(big['FDLF-XB'] / big['SNR']).min():.2f}-"
            f"{(big['FDLF-XB'] / big['SNR']).max():.2f}x the memory of standard NR and RCI "
            f"{(big['RCI'] / big['SNR']).min():.2f}-{(big['RCI'] / big['SNR']).max():.2f}x; "
            f"at 300 buses\nFDLF needs {big['FDLF-XB'].iloc[-1] / 1024:.1f} MiB against "
            f"{big['SNR'].iloc[-1] / 1024:.1f} MiB for either NR variant."
        )

    # ------------------------------------------------------------- figures
    with Section("2. Figures"):
        labels = [f"{int(n)}-bus" for n in sorted(df["n_bus"].unique())]
        pivot_mem = df.pivot_table(index="n_bus", columns="solver", values="peak_mem_kb")
        pivot_rows = df.pivot_table(index="n_bus", columns="solver", values="jac_rows")

        path = plot_grouped_bars(
            labels, {k: pivot_mem[k].to_numpy() for k in KEYS},
            title="Peak heap increment during one solve",
            subtitle="Logarithmic scale, lower is better. SNR and PNR are indistinguishable: "
                     "the peak is set by the dense LU of a system both solve at the same size. "
                     "Only FDLF, which never forms a complex n x n derivative, breaks away.",
            ylabel="Peak allocation (KiB)",
            name="06_peak_memory",
            value_fmt="%.0f",
            log=True,
        )
        print(f"wrote {path}")

        path = plot_scaling(
            pivot_rows.index.to_numpy(),
            {k: pivot_rows[k].to_numpy() for k in KEYS},
            title="Dimension of the linear system solved each iteration",
            ylabel="Rows in the linear system",
            subtitle="SNR and PNR coincide exactly. PNR+ adds one row per PV bus; the "
                     "rectangular method roughly doubles the system; FDLF's two constant "
                     "matrices together have about the same total size but are factorised once.",
            name="06_jacobian_size",
        )
        print(f"wrote {path}")

    # ------------------------------------------------- dense vs sparse storage
    with Section("3. Dense versus sparse storage of Y-bus and the Jacobian"):
        rows = []
        for case in cases:
            stats = sparsity(build_ybus(case, sparse=True))
            n_sys = len(case.pvpq) + len(case.pq)
            jac_nnz = 2 * stats["nnz"]          # roughly, the four real sub-blocks
            rows.append(
                {
                    "case": case.name,
                    "n_bus": case.n_bus,
                    "Ybus_nnz": stats["nnz"],
                    "Ybus_density": stats["density"],
                    "Ybus_dense_kb": stats["dense_bytes"] / 1024,
                    "Ybus_sparse_kb": stats["sparse_bytes"] / 1024,
                    "jac_rows": n_sys,
                    "jac_dense_kb": n_sys * n_sys * 8 / 1024,
                    "jac_sparse_kb": jac_nnz * 8 / 1024,
                    "dense_over_sparse": (n_sys * n_sys * 8) / max(jac_nnz * 8, 1),
                }
            )
        storage = pd.DataFrame(rows)
        print_table(storage)
        save_table(
            storage, "06_ybus_sparsity",
            "Dense versus sparse storage of Y-bus and the Jacobian",
            "Both NR formulations inherit the sparsity of Y-bus, so this saving is available to "
            "either of them and is independent of the base paper's contribution. It is also far "
            "larger than anything the reformulation offers -- see `dense_over_sparse` -- and is "
            "what actually limits how far a dense solver can scale.",
        )

        path = plot_scaling(
            storage["n_bus"].to_numpy(),
            {
                "SNR": storage["jac_dense_kb"].to_numpy(),
                "PNR": storage["jac_sparse_kb"].to_numpy(),
            },
            title="Jacobian storage: dense versus sparse",
            ylabel="Storage (KiB)",
            subtitle="Series are the two storage schemes, which apply equally to both NR "
                     "formulations. Dense storage grows quadratically, sparse linearly.",
            name="06_dense_vs_sparse_memory",
        )
        print(f"wrote {path}")
        big = storage.iloc[-1]
        print(f"\nat {int(big['n_bus'])} buses: dense Jacobian {big['jac_dense_kb']:.0f} KiB "
              f"vs sparse {big['jac_sparse_kb']:.0f} KiB "
              f"({big['dense_over_sparse']:.0f}x)")


if __name__ == "__main__":
    main()
