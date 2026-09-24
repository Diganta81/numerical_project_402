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