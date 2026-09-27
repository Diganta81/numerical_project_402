"""Extension B: the chord (factorise-once) Newton method."""
from __future__ import annotations

import numpy as np
import pytest

from powerflow import build_ybus
from powerflow.benchmark import reference_solution
from powerflow.solvers import SolverOptions, simplified_nr, standard_nr
from powerflow.solvers import chord_nr

#: Paper Section 4, first correction vector [d(delta_2), d(delta_3), d|V_2|].
PAPER_DX1 = np.array([-0.0460, -0.0088, -0.0294])


@pytest.mark.parametrize("base", ["PNR", "SNR"])
def test_chord_reaches_the_same_solution(small_case, base):
    """Same fixed point as Newton: only the path to it differs."""
    v_ref = reference_solution(small_case)
    res = chord_nr.solve(small_case, SolverOptions(max_iter=150), base=base)
    assert res.converged
    assert np.max(np.abs(res.v - v_ref)) < 1e-5


@pytest.mark.parametrize("base", ["PNR", "SNR"])
def test_chord_factorises_exactly_once(small_case, base):
    res = chord_nr.solve(small_case, SolverOptions(max_iter=150), base=base)
    assert res.extras["n_factorizations"] == 1


def test_first_chord_step_is_the_first_newton_step(case3):
    """Both use J(x0) at the flat start, so iteration 1 is identical -- and matches the paper."""
    one = SolverOptions(max_iter=1)
    chord = chord_nr.solve(case3, one, base="PNR")
    newton = simplified_nr.solve(case3, one)
    assert np.allclose(chord.v, newton.v, atol=1e-12)
    dx = np.array([np.angle(chord.v[1]), np.angle(chord.v[2]), np.abs(chord.v[1]) - 1.0])
    assert np.allclose(dx, PAPER_DX1, atol=5e-4)


def test_chord_converges_linearly_not_quadratically(case3):
    """Successive corrections shrink by a roughly constant factor below 1."""
    res = chord_nr.solve(case3, SolverOptions(tol=1e-12, max_iter=100), base="SNR")
    dv = np.array([d for d in res.dv_history if np.isfinite(d) and d > 1e-13])
    ratios = dv[2:] / dv[1:-1]
    assert np.all(ratios < 1.0)
    assert np.std(ratios[-3:]) < 0.05                    # steady contraction factor
    newton = standard_nr.solve(case3, SolverOptions(tol=1e-12))
    assert res.iterations > newton.iterations


def test_refresh_every_refactorises_on_schedule(case3):
    res = chord_nr.solve(case3, SolverOptions(tol=1e-12, max_iter=100), base="PNR", refresh_every=2)
    assert res.converged
    assert res.extras["n_factorizations"] == (res.iterations + 1) // 2


def test_sparse_and_dense_paths_agree(small_case):
    dense = chord_nr.solve(small_case, SolverOptions(max_iter=150), base="PNR")
    sparse = chord_nr.solve(small_case, SolverOptions(max_iter=150, sparse=True), base="PNR",
                            ybus=build_ybus(small_case, sparse=True))
    assert dense.iterations == sparse.iterations
    assert np.max(np.abs(dense.v - sparse.v)) < 1e-10


def test_rejects_bad_arguments(case3):
    with pytest.raises(ValueError):
        chord_nr.solve(case3, base="GS")
    with pytest.raises(ValueError):
        chord_nr.solve(case3, refresh_every=0)
