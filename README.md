# Simplified Newton-Raphson Power Flow

CSE 402 (Numerical Analysis, Simulation and Modeling) project.

Reproduction and extension of T. Kulworawanichpong, *"Simplified Newton-Raphson
power-flow solution method"*, Int. J. Electrical Power & Energy Systems 32
(2010) 551-558.

### Commands:

1. Initial setup:
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run tests for Paper example and Jacobian:
```bash
python -m pytest tests/test_paper_example.py
python -m pytest tests/test_jacobians.py
```

3. Run the three bus example:
```bash
python scripts/01_worked_example_3bus.py
```

4. Run the flop counting:
```sh
python scripts/02_flop_counting.py
```

## Repository layout

```
src/powerflow/            the library — no scripts, no I/O side effects
  case.py                 PowerCase: per-unit injections, bus classes, indices
  loader.py               MATPOWER-format JSON -> PowerCase
  ybus.py                 bus admittance matrix (dense or sparse), branch flows
  results.py              PowerFlowResult and per-iteration history
  flops.py                the paper's Table 1, plus an audited recount
  plotting.py             every figure; fixed method -> colour mapping
  reporting.py            CSV + Markdown table output
  solvers/
    base.py               options, iteration bookkeeping, shared assembly
    standard_nr.py        SNR   — power mismatch          (the benchmark)
    simplified_nr.py      PNR   — current mismatch        (the base paper)
    rect_current_nr.py    RCI   — rectangular current injection
    fast_decoupled.py     FDLF  — Stott & Alsac, XB and BX
    gauss_seidel.py       GS    — classical baseline

scripts/                  one script per result, runnable standalone
  01_worked_example_3bus.py    paper Section 4, entry by entry
  02_flop_counting.py          paper Section 3, Table 1, Figures 1–2

data/                     IEEE test systems as MATPOWER-format JSON (see data/README.md)
tests/                    pytest suite, including the paper's printed numbers
results/                  generated figures and tables
```

### FLOP counting — reproduced, and audited

`scripts/02_flop_counting.py` reproduces Table 1 and Figures 1–2 exactly as
published (`10 n² + O(n)` versus `22(n−2)` multiplications per iteration).

It then recounts the same formulas element by element, because **the paper's
headline claim cannot be right as stated**. A dense Jacobian has `O(n²)` entries
and equations (9), (11), (13), (15) give each of them a distinct `Y_ki`, so each
needs at least one multiplication of its own; a cost linear in `n` is impossible.
Table 1 appears to count the off-diagonal work once per *row* rather than once
per *entry*.

Recounting properly, with the same common-subexpression reuse granted to both:

| | per off-diagonal pair | per bus (diagonals) |
| --- | --- | --- |
| Standard NR | 9 multiplications | 4 |
| Simplified NR | 5 multiplications | 13 |

So the ratio is `(9d + 4) / (5d + 13)` for average nodal degree `d`: it
approaches **9/5 = 1.80** for a dense Jacobian, and equals **1.73 at 57 buses**
— against the **1.728** the paper actually measured on its largest case. *The
paper's conclusion survives in full even though its Table 1 does not.*

The recount also exposes a limit the paper never mentions: the simplified method
is cheaper per matrix *entry* but dearer per *bus*, because its closed-form
diagonals replace row sums the standard method accumulates for free. The
advantage breaks even at `d = 2.25`, and real transmission grids sit at
`d ≈ 2.5–3.7` — so once sparsity is exploited, very little of the advantage is
left. See `results/figures/02_ratio_vs_degree.png`.