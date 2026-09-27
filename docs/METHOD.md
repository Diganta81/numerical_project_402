# The two formulations, side by side

A compact derivation of what the base paper changes and why it is cheaper.
Everything here is implemented in `src/powerflow/solvers/`; equation numbers are
the paper's.

---

## 1. The problem

For an `n`-bus network with bus admittance matrix `Y`, the unknowns are the
complex bus voltages `V_k = |V_k| e^{j delta_k}`. Bus 1 (the slack) has both its
magnitude and angle fixed; PV buses have `|V|` and `P` fixed; PQ buses have `P`
and `Q` fixed. That leaves

- one angle unknown per non-slack bus, and
- one magnitude unknown per PQ bus,

for a total of `(n-1) + n_pq` unknowns. Both formulations below use *exactly*
these unknowns. They differ only in which residual they drive to zero.

---

## 2. Standard NR — the power mismatch

The classical residual is the power imbalance at each bus:

```
dP_k = P_sch,k - P_cal,k        over all non-slack buses
dQ_k = Q_sch,k - Q_cal,k        over PQ buses only
```

with (paper equations 21, 22)

```
P_cal,k =  sum_i |V_k V_i Y_ki| cos(theta_ki + delta_i - delta_k)
Q_cal,k = -sum_i |V_k V_i Y_ki| sin(theta_ki + delta_i - delta_k)
```

The update is paper equation (18):

```
[ dP ]   [ J1  J2 ] [ d(delta) ]
[ dQ ] = [ J3  J4 ] [ d(|V|)   ]
```

followed by `x <- x + dx` (equation 17). Note `J2` and `J4` hold the
*unnormalised* derivatives with respect to `|V|`, matching the paper's own
equations and hence its FLOP counts.

**The shape of a summand.** Every entry of `J1..J4` contains the product
`|V_k| |V_i| |Y_ki|` and the angle `theta_ki + delta_i - delta_k`. Both the
near-end bus `k` and the far-end bus `i` appear. Worse, the diagonal entries are
*sums over the whole row*.

---

## 3. Simplified NR — the current mismatch

The paper's move is to write the *nodal current* balance instead (equation 2):

```
F_k = conj( S_sch,k / V_k ) - sum_i Y_ki V_i = 0
```

This has the same roots. Multiplying through by `conj(V_k)` gives
`conj(V_k) F_k = conj(dS_k)`, so `F = 0` and `dS = 0` are the same condition —
the reformulation is exact, not an approximation.

Splitting `F_k = G_k + j H_k` into real and imaginary parts gives equations (4)
and (5):

```
G_k = |S_sch,k / V_k| cos(delta_k - phi_k) - sum_i |Y_ki V_i| cos(theta_ki + delta_i)
H_k = |S_sch,k / V_k| sin(delta_k - phi_k) - sum_i |Y_ki V_i| sin(theta_ki + delta_i)
```

where `phi_k = arg(S_sch,k)`. `G` is enforced at every non-slack bus and `H` at
PQ buses only, so the system has exactly the same dimension as before.

**Why it is cheaper.** Compare the two summands:

| | summand | depends on |
| --- | --- | --- |
| power mismatch | `\|V_k V_i Y_ki\| cos(theta_ki + delta_i - delta_k)` | both `k` and `i` |
| current mismatch | `\|Y_ki V_i\| cos(theta_ki + delta_i)` | only `i` |

Dropping `|V_k|` from the product and `-delta_k` from the angle is what makes
every off-diagonal Jacobian entry depend on a single bus index. The Jacobian
entries, with `beta_ki = theta_ki + delta_i`, `gamma_k = delta_k - phi_k` and
`a_k = |S_sch,k| / |V_k|`, are equations (9)–(16):

| entry | off-diagonal (`k != i`) | diagonal |
| --- | --- | --- |
| `J1 = dG/d(delta)` | `-\|V_i Y_ki\| sin(beta_ki)` (9) | `-\|V_k Y_kk\| sin(beta_kk) + a_k sin(gamma_k)` (10) |
| `J2 = dG/d\|V\|` | `\|Y_ki\| cos(beta_ki)` (11) | `\|Y_kk\| cos(beta_kk) + (a_k/\|V_k\|) cos(gamma_k)` (12) |
| `J3 = dH/d(delta)` | `\|V_i Y_ki\| cos(beta_ki)` (13) | `\|V_k Y_kk\| cos(beta_kk) - a_k cos(gamma_k)` (14) |
| `J4 = dH/d\|V\|` | `\|Y_ki\| sin(beta_ki)` (15) | `\|Y_kk\| sin(beta_kk) + (a_k/\|V_k\|) sin(gamma_k)` (16) |

Two savings, one of which the paper does not call out:

1. **Fewer factors per entry.** The product has one factor fewer, and `J2`/`J4`
   need no voltage factor at all.
2. **Closed-form diagonals.** The diagonals are single expressions, not row
   sums. The standard method has to accumulate four separate sums across each
   row, which costs extra products per off-diagonal pair.

**Sign convention.** The paper writes `[G; H] = J [d(delta); d|V|]` and then
`x <- x + dx`, so its `J` is `-dF/dx`. Both Jacobian builders in
`simplified_nr.py` follow this, which is why they reproduce the paper's printed
matrices entry for entry.

**Compact equivalent.** With `A_k = conj(S_sch,k / V_k)`, all eight formulas
collapse to

```
J_delta = j Y diag(V) - diag(j A)        ->  J1 = Re,  J3 = Im
J_V     = Y diag(e^{j delta}) + diag(A/|V|)  ->  J2 = Re,  J4 = Im
```

`tests/test_jacobians.py` checks this against a literal transcription of
(9)–(16) *and* against finite differences.

---

## 4. PV buses: the catch

Equation (2) needs a full complex `S_sch,k`, but a PV bus only schedules `P`.
Section 4 of the paper recomputes `Q` from the *present* voltage estimate at the
start of each iteration, then enforces `G_k` alone.

That works, but it layers a successive-substitution fixed point on top of
Newton's method, and **the composite is only linearly convergent**. Measured on
the IEEE 30-bus system to a `1e-12` tolerance, the correction shrinks by a
roughly constant factor each step instead of squaring.

The project's fix (`pv_handling="augmented"`) makes `Q_k` a genuine Newton
unknown and enforces both `G_k` and `H_k` at PV buses, adding the column

```
dJ/dQ_k = j / conj(V_k)
```

This restores quadratic convergence at the cost of one extra unknown per PV bus.
It also explains the single anomaly in the paper's own Table 5: TC3, its most
PV-heavy case, is the only one where the proposed method needed *more*
iterations than the standard one.

---

## 5. What the other solvers do

| Solver | Residual | State | Jacobian |
| --- | --- | --- | --- |
| `SNR` | power mismatch | `\|V\|, delta` | rebuilt each iteration |
| `PNR` | current mismatch | `\|V\|, delta` | rebuilt, cheaper per entry |
| `PNR+` | current mismatch | `\|V\|, delta, Q_pv` | as `PNR`, plus PV columns |
| `RCI` | current mismatch | `e, f, Q_pv` | constant off-diagonal blocks |
| `FDLF` | decoupled power mismatch | `\|V\|, delta` | two constant matrices, factorised once |
| `GS` | nodal fixed point | `V` | none |

`RCI` is the formulation written on slide 4 of the project proposal (and
reference [26] of the paper): keeping the state in rectangular components
`V = e + jf` makes the *network* part of the Jacobian literally `±G` and `±B`
from `Y_bus`, so only the 2×2 diagonal blocks change between iterations. The
price is a system of size `2(n-1) + n_pv` instead of `(n-1) + n_pq`.
