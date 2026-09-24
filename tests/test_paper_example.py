from __future__ import annotations

import numpy as np
import pytest

from powerflow import build_ybus
from powerflow.solvers import SolverOptions, simplified_nr, standard_nr

PRINTED = 5e-4          

PAPER_YBUS_MAG = np.array(
    [[53.8517, 22.3607, 31.6228], [22.3607, 58.1378, 35.7771], [31.6228, 35.7771, 67.2310]]
)
PAPER_YBUS_ANG_DEG = np.array(
    [[-68.20, 116.57, 108.43], [116.57, -63.43, 116.57], [108.43, 116.57, -67.25]]
)

PAPER_STEPS = [
    {
        "mismatch": np.array([-2.8600, 1.3831, 0.2200]),
        "jacobian": np.array(
            [[54.50, -33.28, 22.00], [-32.00, 63.50, -16.00], [30.00, -16.64, -49.50]]
        ),
        "dx": np.array([-0.0460, -0.0088, -0.0294]),
    },
    {
        "mismatch": np.array([-0.0408, 0.0241, -0.0826]),
        "jacobian": np.array(
            [[54.3425, -33.4251, 19.4624],
             [-31.7416, 63.2414, -14.5117],
             [26.8874, -16.3466, -50.2945]]
        ),
        "dx": np.array([-0.0011, 0.0001, 0.0010]),
    },
    {
        "mismatch": np.array([6.71e-5, 4.03e-4, -6.3e-6]),
        "jacobian": np.array(
            [[54.4238, -33.4238, 19.4172],
             [-31.7908, 63.2908, -14.4766],
             [26.8492, -16.3493, -50.3215]]
        ),
        "dx": np.array([7.23e-6, 1.02e-5, 6.80e-7]),
    },
]


def test_ybus_matches_paper(case3):
    ybus = build_ybus(case3)
    assert np.allclose(np.abs(ybus), PAPER_YBUS_MAG, atol=1e-3)
    assert np.allclose(np.degrees(np.angle(ybus)), PAPER_YBUS_ANG_DEG, atol=6e-3)


def test_initial_pv_reactive_power_matches_paper(case3):
    ybus = build_ybus(case3)
    s_eff = simplified_nr.effective_schedule(case3, ybus, case3.flat_start())
    assert s_eff[2].imag == pytest.approx(1.0192, abs=PRINTED)
    assert s_eff[2].real == pytest.approx(2.0, abs=PRINTED)


@pytest.mark.parametrize("step", range(3))
def test_iteration_matches_paper(case3, step):
    ybus = build_ybus(case3)
    v = case3.flat_start()
    vm, va = np.abs(v), np.angle(v)
    pvpq, pq = case3.pvpq, case3.pq
    n_ang = len(pvpq)

    for k in range(step + 1):
        s_eff = simplified_nr.effective_schedule(case3, ybus, v)
        f = simplified_nr.mismatch_vector(case3, ybus, v, s_eff)
        jac = simplified_nr.jacobian(case3, ybus, v, s_eff)
        dx = np.linalg.solve(jac, f)
        if k == step:
            break
        va[pvpq] += dx[:n_ang]
        vm[pq] += dx[n_ang:]
        v = vm * np.exp(1j * va)

    expected = PAPER_STEPS[step]
    
    assert np.allclose(f, expected["mismatch"], atol=PRINTED * max(1.0, np.max(np.abs(f))))
    assert np.allclose(jac, expected["jacobian"], atol=1e-3 * np.max(np.abs(jac)))
    assert np.allclose(dx, expected["dx"], atol=PRINTED * max(0.1, np.max(np.abs(dx))))


def test_table2_solution(case3):
    ybus = build_ybus(case3)
    opts = SolverOptions(tol=1e-10)
    snr = standard_nr.solve(case3, opts, ybus)
    pnr = simplified_nr.solve(case3, opts, ybus)

    assert snr.converged and pnr.converged
    assert pnr.vm[1] == pytest.approx(0.97168, abs=1e-5)
    assert pnr.va_deg[1] == pytest.approx(-2.698, abs=5e-3)
    assert pnr.vm[2] == pytest.approx(1.04, abs=1e-9)
    assert pnr.va_deg[2] == pytest.approx(-0.4979, abs=5e-3)
    assert pnr.injections(ybus).imag[2] == pytest.approx(1.4618, abs=PRINTED)
    assert np.max(np.abs(snr.v - pnr.v)) < 1e-9


def test_pv_bus_uses_only_the_real_current_mismatch(case3):
    ybus = build_ybus(case3)
    v = case3.flat_start()
    jac = simplified_nr.jacobian(case3, ybus, v)
    assert jac.shape == (3, 3)
