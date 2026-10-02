# CSE 402 (Numerical Analysis, Simulation and Modeling) project.

## Team

- [H. M. Aktaruzzaman Mukdho - 2105064](https://github.com/hm-aktaruzzaman-mukdho)
- [Ahtashamul Haque - 2105067](https://github.com/ahtasham67)
- [Anika Morshed - 2105068](https://github.com/Anika-34)
- [Diganta Saha Tirtha - 2105081](https://github.com/Diganta81)
- [Jahedul Islam Nayeem - 2105082](https://github.com/Nayeemj496)

**Base paper:** T. Kulworawanichpong, *"Simplified Newton–Raphson power-flow
solution method"*, International Journal of Electrical Power & Energy Systems
**32** (2010) 551–558.

The paper replaces the power mismatch equations of the standard Newton–Raphson
power flow with **nodal current mismatch** equations. The roots are identical,
but the Jacobian entries become much simpler expressions, so each iteration
costs less. This repository implements both methods from scratch, reproduces
every table and figure of the paper, and then carries out the extensions set out
in the project proposal: scale testing to 118 and 300 buses, and benchmarking
against Fast Decoupled Load Flow and Gauss–Seidel on iterations, runtime and
memory.

---

## Quick start

```bash
pip install -r requirements.txt

python run_all.py            # regenerate every table and figure
python -m pytest             # 285 tests, ~3 s

python run_all.py --list     # show the individual steps
python run_all.py 01         # just the 3-bus worked example
```

Results land in `results/figures/` (PNG + PDF) and `results/tables/`
(CSV + Markdown); step 08 writes to `results/extension_chord/`. Every figure has
a companion table with the same name.

Step 08 runs in its own interpreter, because it pins BLAS/LAPACK to one thread
and those environment variables are only read when NumPy is first imported.

---

## Repository layout

```
src/powerflow/            the library — no scripts, no I/O side effects
  case.py                 PowerCase: per-unit injections, bus classes, indices
  loader.py               MATPOWER-format JSON -> PowerCase
  ybus.py                 bus admittance matrix (dense or sparse), branch flows
  results.py              PowerFlowResult and per-iteration history
  flops.py                the paper's Table 1, plus an audited recount
  benchmark.py            timing, memory and per-stage measurement
  plotting.py             every figure; fixed method -> colour mapping
  reporting.py            CSV + Markdown table output
  solvers/
    base.py               options, iteration bookkeeping, shared assembly
    standard_nr.py        SNR   — power mismatch          (the benchmark)
    simplified_nr.py      PNR   — current mismatch        (the base paper)
    rect_current_nr.py    RCI   — rectangular current injection
    fast_decoupled.py     FDLF  — Stott & Alsac, XB and BX
    gauss_seidel.py       GS    — classical baseline
    chord_nr.py           chord — frozen-Jacobian variants of SNR and PNR

scripts/                  one script per result, runnable standalone
  01_worked_example_3bus.py    paper Section 4, entry by entry
  02_flop_counting.py          paper Section 3, Table 1, Figures 1–2
  03_paper_test_cases.py       paper Section 5, Table 5, Figures 5–10
  04_ext_scale.py              extension 1: 118- and 300-bus scale testing
  05_ext_method_comparison.py  extension 2: FDLF and Gauss–Seidel
  06_ext_memory.py             extension 3: memory overhead
  07_ext_robustness.py         extension 4: robustness under load
  08_ext_chord_method.py       extension 5: chord (frozen-Jacobian) iterations

data/                     IEEE test systems as MATPOWER-format JSON (see data/README.md)
tests/                    pytest suite, including the paper's printed numbers
docs/METHOD.md            the two formulations derived side by side
results/                  generated figures and tables
  extension_chord/        step 08 writes here (own figures/ and tables/)
```

**Where to start reading.** `docs/METHOD.md` for the mathematics;
`src/powerflow/solvers/simplified_nr.py` for the paper's method itself;
`scripts/01_worked_example_3bus.py` to watch it reproduce the paper's own
arithmetic.

Every solver exposes the same entry point, so they are interchangeable:

```python
import sys; sys.path.insert(0, "src")
from powerflow import load_case
from powerflow.solvers import SOLVERS, SolverOptions

case = load_case("case118_ieee")
result = SOLVERS["PNR"](case, SolverOptions(tol=1e-6))
print(result.summary(), result.vm.min(), result.va_deg.max())
```

| Key | Method |
| --- | --- |
| `SNR` | Standard NR, power mismatch — the paper's benchmark |
| `PNR` | Simplified NR, current mismatch — **the base paper** |
| `PNR+` | As `PNR`, with PV reactive power as a Newton unknown (this project) |
| `RCI` | Rectangular current injection — the proposal's slide-4 formulation |
| `FDLF-XB`, `FDLF-BX` | Fast decoupled load flow |
| `GS` | Gauss–Seidel |

The chord variants live outside that registry, in
`powerflow.solvers.chord_nr`, because they take an extra `refresh_every`
argument: `chord_nr.solve_snr(case)` and `chord_nr.solve_pnr(case)`.

---

## Part 1 — reproducing the base paper

### The 3-bus worked example (Section 4) — exact

The paper prints its Y-bus, and then the current mismatch vector, the full
Jacobian and the correction vector for three iterations. All of it reproduces:

- **Y-bus** matches all six printed entries to 4 decimals.
- `Q_cal,3 = 1.0192` p.u. at the flat start — matches.
- **Every Jacobian entry at every one of the three iterations** matches to all
  printed digits, e.g. iteration 1

  ```
  computed  [ 54.5000  -33.2800   22.0000 ]    paper  [ 54.50  -33.28   22.00 ]
            [-32.0000   63.5000  -16.0000 ]           [-32.00   63.50  -16.00 ]
            [ 30.0000  -16.6400  -49.5000 ]           [ 30.00  -16.64  -49.50 ]
  ```

- **Table 2**: `V2 = 0.97168 ∠ −2.696°`, `V3 = 1.04 ∠ −0.499°`, `Q3 = 1.4618` p.u.

Residual differences are at the level of the paper's own 4-decimal rounding.
This is the strongest evidence that all eight Jacobian formulas (equations 9–16)
are implemented correctly. It runs as a test: `tests/test_paper_example.py`.

### FLOP counting (Section 3) — reproduced, and audited

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

The paper's own text makes the slip visible. It states that there are
`(n − 1) × (n − 2)` off-diagonal elements in `J1`, charges the standard method
`3 × (n − 1) × (n − 2)` for them — and then charges the proposed method
`2 × (n − 2)`, one row's worth, for the same population.

Apply the paper's own per-element costs to the correct element count and the
Jacobian ratio is `10/6 = 1.67`. Grant both methods the shared-subexpression
reuse above and it is `(9d + 4) / (5d + 13)` for average nodal degree `d`,
approaching **9/5 = 1.80** for a dense Jacobian. Either way the result is a
**constant factor**, and both methods stay `O(n²)`.

That is the size of effect the paper actually measured: its Table 5 reports
total-time ratios of **1.13–1.73**, where its own Table 1 would have predicted
**25.9×** at 57 buses. *The paper's conclusion survives in full even though its
Table 1 does not.* (Its 1.73 is a total solve time ratio that also includes a
4-vs-3 iteration advantage, so it is not directly a per-iteration Jacobian
figure — only the order of magnitude is comparable.)

The recount also exposes a limit the paper never mentions: the simplified method
is cheaper per matrix *entry* but dearer per *bus*, because its closed-form
diagonals replace row sums the standard method accumulates for free. The
advantage breaks even at `d = 2.25`, and real transmission grids sit at
`d ≈ 2.5–3.7` — so once sparsity is exploited, very little of the advantage is
left. See `results/figures/02_ratio_vs_degree.png`.

### The five test cases (Section 5) — partly reproducible

The paper says its test systems were "modified" but never says how, and the
solutions in its Table 3 are not those of the standard IEEE cases (several of
its 24-bus voltages sit near `0.46 − j0.91` p.u. — a normal magnitude, but an
angle of −63° from the slack bus, a spread no normally-operating transmission
network shows). The unmodified library cases are used here, so **Table 3 is not
reproducible number for number**. Iteration counts, convergence shapes and
timing ratios are, and they are what the paper's claims rest on.

The end-to-end timing result is **negative, and honestly so**: across TC1–TC5 the
simplified method is *not* faster overall, because it needs more iterations. But
decomposing one iteration into its four stages shows the paper's actual claim
holding exactly where it is made:

| stage | differs between methods? | share of an iteration (TC1–TC5) |
| --- | --- | --- |
| mismatch evaluation | yes | 7–23 % |
| **Jacobian derivative evaluation** | **yes — 1.21–1.31× faster for PNR** | 18–26 % |
| block assembly | no — identical work | 32–42 % |
| linear solve | no — identical work | 9–42 % |

Section 3 of the paper assumes "the Jacobian updating step dominates the overall
execution time". It does not: **51–74 % of an iteration** (mean 58 %) is spent on
work that is identical for both methods and cannot be improved by reformulating
the mismatch, and that share *rises* with system size. The derivative speed-up
itself is 1.21–1.31× on TC1–TC4 and 1.51–1.60× at 118 and 300 buses; the 57-bus
point measures 0.97× and is an outlier, bracketed by 1.29× at 30 buses and 1.51×
at 118. See `results/tables/03_stage_timings.md` and
`results/tables/04_derivative_ratio.md`.

---

## Part 2 — the proposal's extensions

### Extension 1 — scale testing (`04_ext_scale.py`)

The proposal asks to go beyond the paper's 57-bus ceiling. Running from 3 to 300
buses turns up the project's main finding:

| | 5-bus | 300-bus | growth |
| --- | --- | --- | --- |
| SNR iterations | 4 | 5 | 1.25× |
| **PNR iterations** | **4** | **13** | **3.25×** |
| PNR+ iterations | 4 | 6 | 1.50× |

**The paper's method does not scale, and the reason is its PV bus treatment.**
Recomputing `Q` at PV buses from the previous voltage estimate is a
successive-substitution fixed point bolted onto Newton's method, and the
composite converges only *linearly*. With 53 PV buses (118-bus) or 68 (300-bus),
that dominates everything the cheaper Jacobian buys. The paper could not see
this: its largest case has 6 PV buses.

This also explains the one anomaly in the paper's own Table 5 — TC3, its most
PV-heavy case, is the only one where the proposed method needed *more*
iterations than the standard one (8 versus 7). The same effect, at the only
scale where the paper could observe it.

**The fix** (`PNR+`, `pv_handling="augmented"`) makes `Q_k` a genuine Newton
unknown and enforces both `G_k` and `H_k` at PV buses, adding the Jacobian column
`j / conj(V_k)`. Quadratic convergence is restored at the cost of one unknown per
PV bus, and iteration counts return to within one of standard NR on every system
tested. This is the project's own contribution.

Meanwhile, the Jacobian advantage the paper predicted *does* grow with size —
from 1.13× at 3 buses to 1.51× at 118 and 1.60× at 300 — it is simply outweighed
by the iteration count.

### Extension 2 — FDLF and Gauss–Seidel (`05_ext_method_comparison.py`)

All seven solvers on all seven systems, verified to agree with a tightly
converged reference to better than `1e-6` p.u. before any timing is compared.

The comparison reframes the paper's contribution. Each method attacks the cost
of an iteration somewhere different, and pays somewhere different:

| Method | What it changes | What it costs |
| --- | --- | --- |
| PNR | cheaper Jacobian entries | lagged PV Q → more iterations |
| PNR+ | as PNR, Q a real unknown | one unknown per PV bus |
| RCI | constant off-diagonal blocks | system grows to `2(n−1) + n_pv` |
| FDLF | Jacobian → two constant matrices, factorised once | linear convergence |
| GS | no Jacobian at all | linear, and the rate degrades with `n` |

**FDLF wins the wall clock on the 118-bus system** — FDLF-BX solves it in 2.35 ms
against 2.72 ms for standard NR and 8.42 ms for the paper's method, despite
needing 6 iterations, because it never rebuilds or refactors anything. At 300
buses the ordering flips back: standard NR is fastest at 22.4 ms, against 31.7 ms
for `PNR+`, 33.1 ms for `RCI`, 42.2 ms for FDLF-XB and 55.6 ms for the paper's
method, because FDLF's extra iterations eventually outweigh its cheaper ones.
That is the honest context for the paper: eliminating the Jacobian rebuild beats
making it cheaper over a wide range, *if* you can afford linear convergence.
Gauss–Seidel fails to converge at all on the 300-bus system within 3000 sweeps,
having needed 378 on the 118-bus one.

### Extension 3 — memory overhead (`06_ext_memory.py`)

The answer here is a **null result, and a clean one: the two NR methods have the
same memory footprint.** Structurally they solve linear systems of *identical*
dimension — the reformulation changes the Jacobian's contents, not its size — and
measured peak heap agrees to within 3.2 % on every system from 24 buses up.
Reformulating the mismatch is a time optimisation; it buys nothing in space.

The real memory differences are elsewhere. FDLF uses a quarter to a half as much
(2.9 MiB against 11.4 MiB at 300 buses) because it never forms a complex `n × n`
derivative matrix at all. And sparse storage dwarfs everything: at 300 buses a
dense Jacobian needs **126×** the memory of a sparse one, which is what actually
limits how far a dense solver scales.

### Extension 4 — robustness under load (`07_ext_robustness.py`)

Every injection is scaled by a loading factor up to 5× and each method re-solved
from a flat start. This separates two kinds of failure. On the 30-bus system
**no** method fails anywhere in the sweep, so every one is censored at 5×. On the
57-bus system **every** method fails at exactly the same loading (2.0×) — a
common ceiling reached simultaneously is the network's own voltage-stability
limit, and no formulation can be blamed for it.

On the 118-bus system, the one with 53 PV buses, they part company sharply:

| | SNR | PNR | PNR+ | RCI | FDLF-XB |
| --- | --- | --- | --- | --- | --- |
| max loading solved | 3.0× | **1.6×** | 2.0× | 3.0× | 3.0× |

The paper's formulation gives up at **half** the loading the others reach. That
is not a physical limit but a loss of convergence, driven by the same lagged-PV
mechanism that costs it iterations — the fixed point it layers onto Newton has a
contraction factor that degrades as the system is stressed.

Worth reporting honestly: `PNR+` recovers only part of that (2.0×), not all.
Making the PV reactive powers Newton unknowns fixes the convergence *rate*
completely and the convergence *basin* only partly. A cheap Jacobian and a fast
local rate do not by themselves buy robustness.

### Extension 5 — chord iterations (`08_ext_chord_method.py`)

Extension 1–4 all point the same way: most of an iteration is work the paper's
reformulation cannot touch. So instead of making the Jacobian *cheaper to build*,
don't rebuild it at all. The chord variants build and LU-factorise once at the
flat start and reuse the factors by substitution thereafter.

Timing each stage separately shows why this matters, and it reframes the whole
study — **the cost profile inverts with system size**:

| | 5-bus | 300-bus |
| --- | --- | --- |
| Jacobian build — *what the paper optimises* | 64 % | **30 %** |
| LU factorisation — *what the paper ignores* | 6 % | **67 %** |
| a chord iteration is cheaper by | 3.3× | **33.6×** |

The paper optimises precisely the stage whose share *shrinks* as systems grow.
Making the build 1.6× cheaper removes about 11 % of an iteration at 300 buses;
skipping the build and the factorisation together removes 97 % of one.

**`SNR-chord` is the fastest solver measured anywhere in this project**: 9.79 ms
on the 300-bus system against 27.93 ms for standard NR, a **2.85×** speed-up,
using a *single* LU factorisation instead of the 5–13 the Newton methods perform.
It needs 16 iterations to standard NR's 5 — but each is 33.6× cheaper. The gain
rises monotonically with size from 30 buses up (1.40×, 1.56×, 1.79×, 2.85×),
which is the opposite of the trend for the paper's reformulation.

And the strategy works on the **standard** formulation, not on the paper's.
`PNR-chord` fails outright at 300 buses, and on the PV-dense 24-bus case its
linear rate degrades to 0.785, needing **57** iterations and running 3.3×
*slower* than ordinary PNR. Freezing the Jacobian is already a linear iteration;
the lagged-PV treatment is a second one layered on top, and the two contraction
factors compound — measured rates are 0.24–0.40 for `SNR-chord` on the large
systems against 0.57–0.79 for `PNR-chord`.

Because step 08 pins BLAS to a single thread, its absolute times are not
comparable with extensions 1–4; the ratios within it are.

---

## Verification

`python -m pytest` — 285 tests, about 3 seconds.

The Jacobians are the part most likely to be silently wrong, so they are checked
three ways:

1. the compact complex form against a **literal transcription** of the paper's
   equations (9)–(16) and their standard-method counterparts;
2. both of those against a **finite-difference** Jacobian of the residual, which
   validates the calculus rather than just the transcription — this is what would
   catch a sign error shared by both implementations;
3. the sparse assembly path against the dense one.

On top of that: the paper's printed numbers are regression tests; every solver
must satisfy the power-flow equations and respect voltage set-points on every
case; all seven must agree with each other; and dense and sparse paths must
agree.

Two real bugs were found this way and fixed — the flat start left non-slack
angles at zero on cases whose slack bus has a non-zero reference angle (the IEEE
118-bus system uses 30°), and the tap-ratio test had been written against a case
that turns out to have no off-nominal taps.

---

## Measurement notes

- Times are the **minimum** of repeated runs (the least noise-contaminated
  sample), with median and spread recorded alongside.
- `Y_bus` construction is **outside** every timed region, so measurements isolate
  the iteration loop — the only thing the two methods differ in.
- Warm-up runs are discarded. This matters more than it sounds: an
  under-warmed measurement of the 118-bus Jacobian reads 2× too slow.
- Garbage collection is disabled inside timing loops.
- The paper's absolute times (Table 4, Pentium IV / Athlon XP / Duron hardware)
  cannot be reproduced; only ratios and trends are comparable.
- All figures use a categorical palette validated for colour-vision deficiency
  (worst adjacent-pair separation ΔE 9.1, normal-vision ΔE 19.6), with each
  method holding a fixed colour, marker and dash pattern across every figure so
  identity never rests on colour alone.

## Summary

The base paper's mechanism is real and reproduces exactly: the current-mismatch
formulation gives genuinely cheaper Jacobian entries, and the measured speed-up
of that step is the size an honest FLOP recount of its own equations predicts.
Four things qualify it, none of which the paper was positioned to see at its
57-bus ceiling:

1. The cost model in Table 1 is off by a factor of `n` — it charges the proposed
   method for one row of off-diagonals where it charges the standard method for
   all `(n − 1) × (n − 2)` of them. Corrected, the saving is a constant factor of
   1.67–1.80, not `n/2`.
2. The Jacobian rebuild is only a quarter of an iteration, so the end-to-end gain
   is lost in the noise — and its share falls to 30 % by 300 buses while the
   factorisation it leaves alone rises to 67 %.
3. The PV bus treatment costs iteration count that grows with the number of PV
   buses, which on large systems is worth more than the cheaper Jacobian, and it
   costs convergence basin too. This has a clean fix, implemented here as `PNR+`.
4. Carried to its conclusion — freeze the Jacobian and reuse its factors — the
   paper's own premise yields 2.85× at 300 buses rather than 1.0×, but only on
   the standard formulation it set out to improve upon.
