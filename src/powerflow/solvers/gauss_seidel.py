r"""
Gauss-Seidel power flow.

Project extension.  The proposal's problem statement contrasts Newton-Raphson
with Gauss-Seidel, so GS is included as the "classical baseline" leg of the
benchmark.  It is the cheapest possible iteration -- no Jacobian, no linear
solve, ``O(nnz)`` work per sweep -- and the most expensive overall, because its
convergence is linear with a rate that degrades as the network grows.  Putting
it beside SNR/PNR/FDLF makes the real point of the base paper visible: cost per
iteration and number of iterations trade off against each other, and the
simplified NR method is an attempt to reduce the former without touching the
latter.

The update, applied bus by bus in place (hence *Seidel*), is the rearranged
nodal equation

.. math::

    V_k \leftarrow \frac{1}{Y_{kk}}
        \left[\overline{\left(\frac{S_{\mathrm{sch},k}}{V_k}\right)}
              - \sum_{i \ne k} Y_{ki} V_i \right],

optionally over-relaxed by an acceleration factor ``alpha``.  At a PV bus the
reactive injection is first estimated from the present voltages and the updated
voltage is then rescaled back onto the magnitude set-point.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..case import PV, PowerCase
from ..results import PowerFlowResult
from ..ybus import build_ybus
from .base import IterationTracker, SolverOptions, initial_voltage, max_abs, s_injected

METHOD = "Gauss-Seidel"
SHORT = "GS"


def solve(
    case: PowerCase,
    options: Optional[SolverOptions] = None,
    ybus=None,
    alpha: float = 1.6,
) -> PowerFlowResult:
    """Solve the power flow of ``case`` by accelerated Gauss-Seidel.

    ``alpha`` is the acceleration (over-relaxation) factor; 1.6 is the value
    recommended in most power-system texts and is what the benchmark uses.
    """
    options = (options or SolverOptions()).validate()
    if ybus is None:
        ybus = build_ybus(case, sparse=options.sparse)
    y = np.asarray(ybus.todense()) if sp.issparse(ybus) else np.asarray(ybus)

    v = initial_voltage(case, options)
    y_diag = np.diag(y).copy()
    non_slack = case.pvpq
    is_pv = case.bus_type == PV
    tracker = IterationTracker(options)

    while True:
        mismatch = case.s_sch - s_injected(y, v)
        # The slack bus absorbs any imbalance, and Q is free at PV buses.
        resid = np.concatenate([mismatch.real[non_slack], mismatch.imag[case.pq]])
        if tracker.begin_step(v, max_abs(resid)):
            break

        v_old = v.copy()
        for k in non_slack:
            if is_pv[k]:
                # Estimate Q from the present voltages, then hold |V| fixed.
                q_k = -np.imag(np.conj(v[k]) * (y[k] @ v))
                s_k = case.p_sch[k] + 1j * q_k
            else:
                s_k = case.s_sch[k]
            v_new = (np.conj(s_k / v[k]) - (y[k] @ v - y_diag[k] * v[k])) / y_diag[k]
            v[k] = v[k] + alpha * (v_new - v[k])
            if is_pv[k]:
                v[k] *= case.vm_set[k] / np.abs(v[k])

        tracker.end_step(max_abs(v - v_old))

    return tracker.finish(METHOD, case, v, alpha=alpha)
