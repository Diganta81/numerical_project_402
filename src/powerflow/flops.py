r"""
Floating-point operation counting -- Section 3 of the base paper.

The paper argues that the simplified method wins because its Jacobian is
cheaper to rebuild, and it backs that up with the multiplication counts of its
Table 1 and the two log-log plots in Figures 1 and 2.  Following the paper,
"FLOP" here always means a *multiplication or division*; additions and
subtractions are treated as free, and so are the transcendental evaluations.

This module provides three counting models, which is the interesting part.

``paper``
    The counts exactly as published (:func:`paper_table1`, :func:`paper_figure1`,
    :func:`paper_figure2`).  Reproduces Table 1 and Figures 1-2 of the paper.

``audited``
    An element-by-element recount of the *same* formulas
    (:func:`audited_jacobian`).  This is included because the paper's headline
    claim -- that the simplified method needs only ``22(n-2)`` multiplications
    per iteration, i.e. a cost *linear* in the number of buses -- cannot hold
    for a dense Jacobian: the matrix has ``O(n^2)`` entries and equations (9),
    (11), (13) and (15) give every one of them a distinct value ``Y_ki``, so
    each needs at least one multiplication of its own.  Table 1 appears to have
    counted the off-diagonal work once per *row* rather than once per *entry*.
    The audited count keeps both methods dense and gives them the same
    common-subexpression optimisations, so the comparison stays fair.  It
    predicts a dense-Jacobian speed-up approaching ``9/5 = 1.8``, and ``1.73``
    at the 57-bus size of the paper's largest test case -- against the ``1.728``
    the paper actually measured there.  The paper's *conclusion* therefore
    survives in full even though its Table 1 does not.

    The recount also exposes a limit the paper never discusses: the simplified
    method is cheaper per matrix *entry* but dearer per *bus*, because its
    closed-form diagonals (equations 10, 12, 14, 16) replace row sums that the
    standard method accumulates for free.  The advantage therefore depends on
    the average nodal degree and disappears below ``d = 2.25``
    (:func:`ratio_vs_degree`).

``sparse``
    The audited count with the dense ``(n-1)(n-2)`` off-diagonal population
    replaced by the true number of non-zeros of ``Y_bus``
    (:func:`sparse_jacobian`).  This is what any real solver pays, and it is the
    model relevant to the large-scale extension cases.  Transmission grids have
    ``d ~ 2.5-3.7``, only just above break-even, which predicts the simplified
    Jacobian gaining very little once sparsity is exploited -- a prediction the
    measured timings in script 05 bear out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. The paper's published counts (Table 1, Figures 1 and 2)
# ---------------------------------------------------------------------------


def paper_table1(n: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Reproduce Table 1 of the paper for an ``n``-bus system.

    Standard NR (left column of the table)::

        J1, J3:  diagonal 3(n-1)        off-diagonal 3(n-1)(n-2)
        J2, J4:  diagonal 2(n-1)+3      off-diagonal 2(n-1)(n-2)

    Proposed NR (right column)::

        J1, J3:  diagonal 4(n-2)        off-diagonal 2(n-2)
        J2, J4:  diagonal 4(n-2)        off-diagonal  (n-2)
    """
    off = (n - 1) * (n - 2)
    snr = {
        "J1": {"diagonal": 3 * (n - 1), "off_diagonal": 3 * off},
        "J2": {"diagonal": 2 * (n - 1) + 3, "off_diagonal": 2 * off},
        "J3": {"diagonal": 3 * (n - 1), "off_diagonal": 3 * off},
        "J4": {"diagonal": 2 * (n - 1) + 3, "off_diagonal": 2 * off},
    }
    pnr = {
        "J1": {"diagonal": 4 * (n - 2), "off_diagonal": 2 * (n - 2)},
        "J2": {"diagonal": 4 * (n - 2), "off_diagonal": 1 * (n - 2)},
        "J3": {"diagonal": 4 * (n - 2), "off_diagonal": 2 * (n - 2)},
        "J4": {"diagonal": 4 * (n - 2), "off_diagonal": 1 * (n - 2)},
    }
    for tbl in (snr, pnr):
        for blk in tbl.values():
            blk["total"] = blk["diagonal"] + blk["off_diagonal"]
        tbl["overall"] = {"total": sum(b["total"] for b in tbl.values())}
    return {"SNR": snr, "PNR": pnr}


def paper_figure1(n):
    """Jacobian FLOPs per iteration as plotted in Figure 1 (leading order).

    Returns ``(standard, proposed) = (10 n^2, 22 (n-2))``.
    """
    n = np.asarray(n, dtype=float)
    return 10.0 * n**2, 22.0 * (n - 2.0)


def paper_figure2(n):
    """Mismatch-vector FLOPs per iteration as plotted in Figure 2.

    The paper states that equations (4) and (5) together cost ``4n + 4``
    multiplications, while forming ``P_cal`` and ``Q_cal`` with equations (21)
    and (22) costs ``6n``.  Returns ``(standard, proposed)``.
    """
    n = np.asarray(n, dtype=float)
    return 6.0 * n, 4.0 * n + 4.0


# ---------------------------------------------------------------------------
# 2. Audited element-by-element counts of the same formulas
# ---------------------------------------------------------------------------

#: Multiplications per off-diagonal ``(k, i)`` pair, filling **all four**
#: sub-matrices at once and reusing every shared sub-expression.
#:
#: Standard NR, ``alpha = theta_ki + delta_i - delta_k``, 9 products::
#:
#:     u  = |V_i| * |Y_ki|      (1)     m * sin(alpha)   -> J1[k,i] and the J1[k,k] sum
#:     w  = |V_k| * |Y_ki|      (1)     m * cos(alpha)   -> J3[k,i] and the J3[k,k] sum
#:     m  = |V_k| * u           (1)     w * cos(alpha)   -> J2[k,i]
#:                                      w * sin(alpha)   -> J4[k,i]
#:                                      u * cos(alpha)   -> the J2[k,k] sum
#:                                      u * sin(alpha)   -> the J4[k,k] sum
#:
#: Simplified NR, ``beta = theta_ki + delta_i``, 5 products::
#:
#:     u = |V_i| * |Y_ki|       (1)     u      * sin(beta) -> J1[k,i]
#:                                      u      * cos(beta) -> J3[k,i]
#:                                      |Y_ki| * cos(beta) -> J2[k,i]
#:                                      |Y_ki| * sin(beta) -> J4[k,i]
#:
#: Two separate effects make the simplified column shorter.  The summand has no
#: ``|V_k|`` factor, which removes ``w`` and ``m``; and -- the larger saving --
#: the simplified diagonals are *closed-form expressions* (equations 10, 12, 14,
#: 16) rather than sums over the row, so an off-diagonal pair contributes nothing
#: to them at all, while the standard method must accumulate four separate row
#: sums.
OFF_DIAGONAL_COST = {"SNR": 9, "PNR": 5}

#: Multiplications per bus for the four diagonal entries.
#:
#: Standard NR (4): ``|V_k||Y_kk|``, doubled, times ``cos(theta_kk)`` and
#: ``sin(theta_kk)``.  The row sums themselves are additions, already paid for
#: in :data:`OFF_DIAGONAL_COST`.
#:
#: Simplified NR (13): ``|V_k||Y_kk|`` times sine and cosine (3),
#: ``|Y_kk|`` times sine and cosine (2), ``|S_k| = sqrt(P^2+Q^2)`` (2),
#: ``a_k = |S_k|/|V_k|`` (1) times sine and cosine of ``gamma_k`` (2), and
#: ``b_k = a_k/|V_k|`` (1) times the same pair (2).
#:
#: So the simplified method is cheaper per *edge* and dearer per *bus*.  How the
#: two balance out depends on the average nodal degree -- see :func:`ratio_vs_degree`.
DIAGONAL_COST = {"SNR": 4, "PNR": 13}

#: Multiplications per ``(k, i)`` term of the mismatch vectors.
#: Standard NR (equations 21, 22): ``|V_k||V_i||Y_ki|`` then times cos and sin.
#: Simplified NR (equations 4, 5): ``|V_i||Y_ki|`` then times cos and sin.
MISMATCH_COST = {"SNR": 4, "PNR": 3}


@dataclass
class FlopCount:
    """Per-iteration multiplication counts under one model."""

    method: str
    n: int
    jacobian_diagonal: float
    jacobian_off_diagonal: float
    mismatch: float

    @property
    def jacobian(self) -> float:
        return self.jacobian_diagonal + self.jacobian_off_diagonal

    @property
    def total(self) -> float:
        return self.jacobian + self.mismatch


def audited_jacobian(method: str, n: int, n_off: Optional[int] = None) -> FlopCount:
    """Honest per-element count for ``method`` on an ``n``-bus system.

    ``n_off`` is the number of off-diagonal entries actually stored; it defaults
    to the dense population ``(n-1)(n-2)`` used by the paper.
    """
    if method not in OFF_DIAGONAL_COST:
        raise KeyError(f"no audited model for {method!r}")
    n_off = (n - 1) * (n - 2) if n_off is None else n_off
    return FlopCount(
        method=method,
        n=n,
        jacobian_diagonal=DIAGONAL_COST[method] * (n - 1),
        jacobian_off_diagonal=OFF_DIAGONAL_COST[method] * n_off,
        mismatch=MISMATCH_COST[method] * (n_off + n),
    )


def sparse_jacobian(method: str, n: int, nnz: int) -> FlopCount:
    """Audited count over the true sparsity pattern of ``Y_bus``.

    ``nnz`` is the total number of stored entries of ``Y_bus`` (diagonal
    included), as returned by :func:`powerflow.ybus.sparsity`.
    """
    return audited_jacobian(method, n, n_off=max(nnz - n, 0))


def ratio_vs_degree(degree):
    """Predicted Jacobian-cost ratio ``SNR/PNR`` as a function of nodal degree.

    With ``d`` off-diagonal entries per row the two costs per bus are
    ``9d + 4`` and ``5d + 13``, so the ratio is

    .. math::  r(d) = \\frac{9d + 4}{5d + 13}

    which rises from below 1 for a radial network to the dense asymptote
    ``9/5 = 1.8``.  Break-even (``r = 1``) sits at ``d = 2.25``: a network
    sparser than that gives the simplified Jacobian *no* advantage, because its
    closed-form diagonals cost more than the row sums they replace.  Real
    transmission grids sit at ``d ~ 2.5-3.7`` (see the ``02_sparse_flops``
    table), while the dense formulation the paper counts has ``d = n - 1``.
    """
    d = np.asarray(degree, dtype=float)
    return (OFF_DIAGONAL_COST["SNR"] * d + DIAGONAL_COST["SNR"]) / (
        OFF_DIAGONAL_COST["PNR"] * d + DIAGONAL_COST["PNR"]
    )


#: Nodal degree at which the two Jacobians cost the same.
BREAK_EVEN_DEGREE = (DIAGONAL_COST["PNR"] - DIAGONAL_COST["SNR"]) / (
    OFF_DIAGONAL_COST["SNR"] - OFF_DIAGONAL_COST["PNR"]
)


def speedup(n, model: str = "paper") -> np.ndarray:
    """Predicted per-iteration cost ratio ``SNR / PNR``."""
    n = np.asarray(n, dtype=float)
    if model == "paper":
        js, jp = paper_figure1(n)
        ms, mp = paper_figure2(n)
        return (js + ms) / (jp + mp)
    if model == "audited":
        return np.array(
            [audited_jacobian("SNR", int(k)).total / audited_jacobian("PNR", int(k)).total for k in np.atleast_1d(n)]
        )
    raise ValueError(f"unknown model {model!r}")