# Test system data

All cases are stored in the MATPOWER/PYPOWER column format as JSON, so they stay
directly comparable with the case library everyone else benchmarks against.
`src/powerflow/loader.py` is the only place that interprets these columns.

| File | Buses | Role |
| --- | --- | --- |
| `case3_saadat.json` | 3 | The worked numerical example of Section 4 of the base paper (Saadat, Example 6.10). Line impedances back-derived from the Y-bus printed in the paper. |
| `case5_stagg.json` | 5 | Paper test case TC1 (Stagg & El-Abiad). |
| `case6ww.json` | 6 | Paper test case TC2 (Wood & Wollenberg). |
| `case14_ieee.json` | 14 | Not used by the paper; extra coverage for the tests. |
| `case24_ieee_rts.json` | 24 | Paper test case TC3 (IEEE RTS). |
| `case30_ieee.json` | 30 | Paper test case TC4. |
| `case57_ieee.json` | 57 | Paper test case TC5, the largest system in the paper. |
| `case118_ieee.json` | 118 | Extension: the scale test named in the project proposal. |
| `case300_ieee.json` | 300 | Extension: a further scale point. |

## Provenance

`case6ww` through `case300_ieee` were exported unmodified from the PYPOWER 5.1.21
case library (BSD-3-licensed, itself derived from MATPOWER). `case3_saadat` and
`case5_stagg` are transcribed from the textbooks the paper cites as its
references [1] and [2].

## Two things to know before comparing with the paper

**The paper's systems were modified.** Section 5 says its test systems were
"modified" but never says how, and the solutions it prints in Table 3 are not
those of the standard IEEE cases -- several of its 24-bus entries sit near
`0.46 - j0.91` p.u., which no healthy network produces. The unmodified library
cases are used here, so Table 3 is not reproducible number for number. Iteration
counts, convergence behaviour and timing ratios all are.

**Bus 2 of `case5_stagg` carries a PQ generator.** The Stagg & El-Abiad table
gives it a fixed 40 MW / 30 MVAr injection with no voltage set-point, so the
loader treats generators at PQ buses as constant `P, Q` injections rather than
rejecting them.

## Column layout

Standard MATPOWER ordering, 0-indexed:

- **bus**: `bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin`
  (`type`: 1 = PQ, 2 = PV, 3 = slack)
- **gen**: `bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin`
- **branch**: `fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax`
  (`ratio` of 0 means a nominal 1:1 ratio)
