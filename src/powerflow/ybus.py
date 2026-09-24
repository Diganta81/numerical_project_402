"""
Bus admittance matrix construction.

Both the standard and the simplified Newton-Raphson formulations are driven by
the same Y_bus, so it is built once, here, and shared. The branch model is the
usual pi-equivalent with an off-nominal, possibly phase-shifting tap on the
"from" side.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .case import PowerCase


def build_ybus(case: PowerCase, sparse: bool = False):
    """Return the n x n bus admittance matrix of `case`.

    If sparse is True, returns a CSR matrix (used for the large-scale
    extension cases); otherwise a dense ndarray.
    """
    n = case.n_bus
    ys = 1.0 / (case.br_r + 1j * case.br_x)
    bc = case.br_b
    t = case.tap * np.exp(1j * case.shift)

    ytt = ys + 1j * bc / 2.0
    yff = ytt / (t * np.conj(t))
    yft = -ys / np.conj(t)
    ytf = -ys / t

    f, tb = case.f_bus, case.t_bus
    rows = np.concatenate([f, f, tb, tb])
    cols = np.concatenate([f, tb, f, tb])
    vals = np.concatenate([yff, yft, ytf, ytt])

    y = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
    y = y + sp.diags(case.y_shunt)
    return y.tocsr() if sparse else np.asarray(y.todense())


def ybus_polar(ybus):
    """Split Y_bus into magnitude and angle arrays, |Y_ki| and theta_ki."""
    y = np.asarray(ybus.todense()) if sp.issparse(ybus) else np.asarray(ybus)
    return np.abs(y), np.angle(y)


def branch_flows(case: PowerCase, v: np.ndarray):
    """Complex power flows at both ends of every branch, in p.u.: (s_from, s_to)."""
    ys = 1.0 / (case.br_r + 1j * case.br_x)
    bc = case.br_b
    t = case.tap * np.exp(1j * case.shift)
    ytt = ys + 1j * bc / 2.0
    yff = ytt / (t * np.conj(t))
    yft = -ys / np.conj(t)
    ytf = -ys / t

    vf, vt = v[case.f_bus], v[case.t_bus]
    i_f = yff * vf + yft * vt
    i_t = ytf * vf + ytt * vt
    return vf * np.conj(i_f), vt * np.conj(i_t)


def sparsity(ybus) -> dict:
    """Structural statistics of Y_bus, used by the memory-overhead comparison."""
    y = ybus.tocsr() if sp.issparse(ybus) else sp.csr_matrix(ybus)
    n = y.shape[0]
    nnz = int(y.nnz)
    return {
        "n": n,
        "nnz": nnz,
        "density": nnz / (n * n),
        "dense_bytes": n * n * 16,
        "sparse_bytes": nnz * 16 + nnz * 4 + (n + 1) * 4,
    }
