"""
Power-system case container.

A PowerCase is a solver-agnostic description of a network: per-unit scheduled
injections, bus classification and branch data. Every solver consumes this one
object, so the solvers stay directly comparable -- they differ only in the
equations they iterate on, never in the data they see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# MATPOWER bus-type codes
PQ = 1
PV = 2
REF = 3
ISOLATED = 4


@dataclass
class PowerCase:
    """An n-bus power system in per-unit on base_mva.

    Bus quantities are indexed by internal bus index 0..n-1; bus_ids maps back
    to the external bus numbers used by the data file.
    """

    name: str
    base_mva: float

    # bus data (length n)
    bus_ids: np.ndarray
    bus_type: np.ndarray
    p_sch: np.ndarray            # scheduled real injection, p.u. (gen - load)
    q_sch: np.ndarray            # scheduled reactive injection, p.u. (gen - load)
    vm_set: np.ndarray           # voltage magnitude set-point (slack & PV buses)
    va_set: np.ndarray           # voltage angle set-point, rad (slack bus)
    y_shunt: np.ndarray          # bus shunt admittance, p.u.
    q_max: np.ndarray            # aggregate generator Q limits, p.u.
    q_min: np.ndarray

    # branch data (length nl)
    f_bus: np.ndarray
    t_bus: np.ndarray
    br_r: np.ndarray             # series resistance, p.u.
    br_x: np.ndarray             # series reactance, p.u.
    br_b: np.ndarray             # total line charging susceptance, p.u.
    tap: np.ndarray              # off-nominal turns ratio (1.0 = nominal)
    shift: np.ndarray            # phase-shift angle, rad
    br_status: np.ndarray

    description: str = ""
    source: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def n_bus(self) -> int:
        return len(self.bus_ids)

    @property
    def n_branch(self) -> int:
        return len(self.f_bus)

    @property
    def slack(self) -> np.ndarray:
        return np.flatnonzero(self.bus_type == REF)

    @property
    def pv(self) -> np.ndarray:
        return np.flatnonzero(self.bus_type == PV)

    @property
    def pq(self) -> np.ndarray:
        return np.flatnonzero(self.bus_type == PQ)

    @property
    def pvpq(self) -> np.ndarray:
        """Non-slack buses in ascending order -- the buses with an unknown angle."""
        return np.sort(np.concatenate([self.pv, self.pq]))

    @property
    def s_sch(self) -> np.ndarray:
        """Complex scheduled injection P + jQ, in p.u."""
        return self.p_sch + 1j * self.q_sch

    def flat_start(self) -> np.ndarray:
        """Flat-start voltage vector.

        Magnitudes are 1.0 except at slack and PV buses, where the set-point
        is enforced. Angles are the slack bus angle everywhere -- not zero --
        so cases with a non-zero reference angle (e.g. IEEE 118-bus, 30 deg)
        still start from a genuinely flat state.
        """
        vm = np.ones(self.n_bus)
        fixed = np.concatenate([self.slack, self.pv])
        vm[fixed] = self.vm_set[fixed]
        reference = self.va_set[self.slack[0]] if len(self.slack) else 0.0
        va = np.full(self.n_bus, reference)
        return vm * np.exp(1j * va)

    def scaled(self, load_factor: float = 1.0, gen_factor: float | None = None) -> "PowerCase":
        """Return a copy with all injections scaled -- used for stress testing.

        load_factor multiplies negative (net-load) injections; gen_factor
        multiplies positive (net-generation) injections and defaults to
        load_factor so the power balance stays roughly intact.
        """
        import copy

        gen_factor = load_factor if gen_factor is None else gen_factor
        new = copy.deepcopy(self)
        load = self.p_sch < 0
        new.p_sch = np.where(load, self.p_sch * load_factor, self.p_sch * gen_factor)
        new.q_sch = np.where(load, self.q_sch * load_factor, self.q_sch * gen_factor)
        new.name = f"{self.name}@x{load_factor:g}"
        return new

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_bus} buses ({len(self.slack)} slack, "
            f"{len(self.pv)} PV, {len(self.pq)} PQ), {self.n_branch} branches, "
            f"base {self.base_mva:g} MVA"
        )

    def __repr__(self) -> str:
        return f"<PowerCase {self.summary()}>"
