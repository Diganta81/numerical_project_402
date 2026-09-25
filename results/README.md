# Generated results

Everything here is produced by `python run_all.py` and is safe to delete — it
regenerates in about two minutes. Figures are written as both PNG and PDF; every
figure has a table of the same name carrying the numbers behind it, as CSV (for
analysis) and Markdown (for reading).

## Reproductions of the base paper

| Output | Corresponds to | Shows |
| --- | --- | --- |
| `tables/01_ybus_polar` | Section 4 | Y-bus against the matrix printed in the paper |
| `tables/01_iterations` | Section 4 | per-iteration agreement with the paper's printed mismatch, Jacobian and correction |
| `tables/01_table2_solution` | Table 2 | converged solution, both methods, against the paper |
| `figures/01_fig4_convergence` | **Figure 4** | solution convergence of the 3-bus example |
| `tables/02_table1_flops` | **Table 1** | multiplication counts per iteration, as published |
| `figures/02_fig1_jacobian_flops` | **Figure 1** | Jacobian FLOPs per iteration versus bus count |
| `figures/02_fig2_mismatch_flops` | **Figure 2** | mismatch-vector FLOPs per iteration |
| `tables/03_table3_solutions` | Table 3 | bus voltages for TC1–TC5 (see the caveat below) |
| `tables/03_table5_results` | **Table 5** | iterations, calculation time and time ratio |
| `figures/03_fig5_time_ratio` | **Figure 5** | calculation time for the five test cases |
| `figures/03_fig6…10_convergence_TC*` | **Figures 6–10** | convergence of each test case |

**Caveat on Table 3.** The paper's test systems were "modified" in an unstated
way and its printed solutions are not those of the standard IEEE cases. The
unmodified library cases are used here, so `03_table3_solutions` is this
project's own solution set rather than a reproduction. Iteration counts,
convergence shapes and timing ratios are reproducible and are what the paper's
claims rest on.

## Analysis beyond the paper

| Output | Shows |
| --- | --- |
| `tables/02_flop_models`, `figures/02_flop_models` | the paper's FLOP counts beside an audited element-by-element recount |
| `tables/02_sparse_flops`, `figures/02_sparse_flops` | the audited count over the true Y-bus sparsity pattern |
| `tables/02_density_ratio`, `figures/02_ratio_vs_degree` | how the cost advantage depends on network density; break-even at degree 2.25 |
| `tables/03_stage_timings`, `figures/03_jacobian_stage_time` | one iteration split into mismatch / derivatives / assembly / linear solve |
| `figures/03_fig5_ratio_comparison` | measured time ratio against the ratio the paper reported |
| `figures/03_iterations` | iteration counts for TC1–TC5 side by side |

## Extension 1 — scale testing

| Output | Shows |
| --- | --- |
| `tables/04_scale_results` | all four Newton methods from 3 to 300 buses |
| `figures/04_iterations_vs_size` | **the project's main finding** — the paper's method needs more iterations as PV buses multiply |
| `figures/04_time_vs_size`, `tables/04_scaling_exponents` | measured growth of solve time, with fitted exponents |
| `tables/04_derivative_ratio`, `figures/04_derivative_ratio_vs_size` | the Jacobian advantage does grow with size, as predicted |
| `tables/04_dense_vs_sparse`, `figures/04_dense_vs_sparse` | where sparse storage starts to pay |

## Extension 2 — method comparison

| Output | Shows |
| --- | --- |
| `tables/05_method_comparison` | all seven solvers on all seven systems |
| `tables/05_solution_agreement` | every method's distance from a tightly converged reference |
| `figures/05_iterations_all_methods`, `figures/05_time_all_methods` | iterations and wall-clock by method |
| `figures/05_convergence_30`, `figures/05_convergence_118` | convergence *character*: quadratic curves versus linear straight lines |

## Extension 3 — memory

| Output | Shows |
| --- | --- |
| `tables/06_memory`, `figures/06_peak_memory` | peak heap per solve; SNR and PNR are indistinguishable |
| `figures/06_jacobian_size` | dimension of the linear system each method factorises |
| `tables/06_ybus_sparsity`, `figures/06_dense_vs_sparse_memory` | dense versus sparse storage |

## Extension 4 — robustness

| Output | Shows |
| --- | --- |
| `tables/07_robustness` | iterations against loading factor, every method and system |
| `figures/07_iterations_heatmap_*` | the same as a surface; blank cells did not converge |
| `tables/07_max_loading`, `figures/07_max_loading` | largest loading factor each method still solves |

## Reading the figures

Each method keeps the same colour, marker and dash pattern in every figure, so
identity never rests on colour alone:

| | SNR | PNR | PNR+ | RCI | FDLF-XB | FDLF-BX | GS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| colour | blue | orange | aqua | yellow | magenta | green | violet |
| marker | ● | ■ | ◆ | ▲ | ▼ | ✚ | ✖ |

On a semi-log convergence plot a **straight line is linear convergence** and a
**curve that steepens is quadratic** — that one distinction carries most of the
findings in this project.
