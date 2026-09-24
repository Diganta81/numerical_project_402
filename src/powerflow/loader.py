"""
Load MATPOWER/PYPOWER-format JSON case files into PowerCase.

The on-disk format is the plain MATPOWER column layout so the vendored IEEE
cases in data/ stay comparable with the standard library everyone else
benchmarks against. All the interpretation happens here, once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .case import ISOLATED, PQ, PowerCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# MATPOWER column indices (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = range(11)

#: Cases used by the base paper, in the order they appear in its Table 5.
PAPER_TEST_CASES = [
    ("TC1", "case5_stagg"),
    ("TC2", "case6ww"),
    ("TC3", "case24_ieee_rts"),
    ("TC4", "case30_ieee"),
    ("TC5", "case57_ieee"),
]

#: Additional systems used by the project extensions (scale testing).
EXTENSION_CASES = [
    ("EX1", "case118_ieee"),
    ("EX2", "case300_ieee"),
]


def available_cases() -> List[str]:
    """Names of every case shipped in data/."""
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))


def load_case(name: str, data_dir: Path | str | None = None) -> PowerCase:
    """Load case `name` (with or without the .json suffix)."""
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    path = Path(name)
    if not path.suffix:
        path = directory / f"{name}.json"
    elif not path.is_absolute():
        path = directory / path
    with open(path) as fh:
        raw = json.load(fh)
    return from_matpower(raw, name=raw.get("name", path.stem))


def load_cases(names: Iterable[str]) -> List[PowerCase]:
    return [load_case(n) for n in names]


def from_matpower(ppc: dict, name: str = "case") -> PowerCase:
    """Convert a MATPOWER-format dict into a PowerCase.

    Handles three things every power-flow loader needs: renumbering external
    bus IDs into contiguous internal indices, aggregating multiple generators
    on the same bus, and taking a PV/slack bus's voltage set-point from its
    generator's Vg rather than the bus table's Vm.
    """
    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    gen = np.asarray(ppc["gen"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)

    in_service = bus[:, BUS_TYPE] != ISOLATED
    if not in_service.all():
        bus = bus[in_service]

    bus_ids = bus[:, BUS_I].astype(int)
    n = len(bus_ids)
    e2i = {int(b): i for i, b in enumerate(bus_ids)}

    bus_type = bus[:, BUS_TYPE].astype(int)

    p_sch = -bus[:, PD] / base_mva
    q_sch = -bus[:, QD] / base_mva
    vm_set = bus[:, VM].copy()
    va_set = np.radians(bus[:, VA])
    q_max = np.zeros(n)
    q_min = np.zeros(n)

    for row in gen:
        if row[GEN_STATUS] <= 0:
            continue
        k = e2i[int(row[GEN_BUS])]
        p_sch[k] += row[PG] / base_mva
        q_sch[k] += row[QG] / base_mva
        q_max[k] += row[QMAX] / base_mva
        q_min[k] += row[QMIN] / base_mva
        if bus_type[k] != PQ:
            vm_set[k] = row[VG]

    y_shunt = (bus[:, GS] + 1j * bus[:, BS]) / base_mva

    live = branch[:, BR_STATUS] != 0 if branch.shape[1] > BR_STATUS else np.ones(len(branch), bool)
    branch = branch[live]
    f_bus = np.array([e2i[int(b)] for b in branch[:, F_BUS]], dtype=int)
    t_bus = np.array([e2i[int(b)] for b in branch[:, T_BUS]], dtype=int)
    tap = branch[:, TAP].copy() if branch.shape[1] > TAP else np.ones(len(branch))
    tap[tap == 0] = 1.0  # MATPOWER convention: 0 means nominal ratio
    shift = np.radians(branch[:, SHIFT]) if branch.shape[1] > SHIFT else np.zeros(len(branch))

    return PowerCase(
        name=name,
        base_mva=base_mva,
        bus_ids=bus_ids,
        bus_type=bus_type,
        p_sch=p_sch,
        q_sch=q_sch,
        vm_set=vm_set,
        va_set=va_set,
        y_shunt=y_shunt,
        q_max=q_max,
        q_min=q_min,
        f_bus=f_bus,
        t_bus=t_bus,
        br_r=branch[:, BR_R],
        br_x=branch[:, BR_X],
        br_b=branch[:, BR_B],
        tap=tap,
        shift=shift,
        br_status=np.ones(len(branch), dtype=bool),
        description=ppc.get("description", ""),
        source=ppc.get("source", ""),
        meta={k: v for k, v in ppc.items() if k not in {"bus", "gen", "branch"}},
    )
