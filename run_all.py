#!/usr/bin/env python
"""Run the whole project end to end.

    python run_all.py                # everything
    python run_all.py --paper        # only the base-paper reproduction (01-03)
    python run_all.py --extensions   # only the proposal extensions (04-08)
    python run_all.py 01 03          # named steps
    python run_all.py --list         # show the steps and exit
"""
from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

#: (number, script, group, description, fresh_process)
#:
#: ``fresh_process`` steps are launched in a new interpreter instead of being
#: executed in this one.  Step 08 needs it: it pins BLAS/LAPACK to a single
#: thread by setting environment variables that are only read when NumPy is
#: first imported.  Run in-process after any earlier step, those variables
#: would arrive too late and the timings would silently differ from a
#: standalone run.
STEPS = [
    ("01", "01_worked_example_3bus.py", "paper",
     "Section 4: the 3-bus worked example, entry by entry", False),
    ("02", "02_flop_counting.py", "paper",
     "Section 3: Table 1 and Figures 1-2, plus an audited recount", False),
    ("03", "03_paper_test_cases.py", "paper",
     "Section 5: TC1-TC5, Table 5 and Figures 5-10", False),
    ("04", "04_ext_scale.py", "extensions",
     "Extension 1: scale testing to 118 and 300 buses", False),
    ("05", "05_ext_method_comparison.py", "extensions",
     "Extension 2: benchmark against FDLF and Gauss-Seidel", False),
    ("06", "06_ext_memory.py", "extensions",
     "Extension 3: memory overhead", False),
    ("07", "07_ext_robustness.py", "extensions",
     "Extension 4: robustness under increasing load", False),
    ("08", "08_ext_chord_method.py", "extensions",
     "Extension 5: chord (frozen-Jacobian) iterations", True),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="*", help="step numbers to run, e.g. 01 03")
    p.add_argument("--paper", action="store_true", help="only the base-paper reproduction")
    p.add_argument("--extensions", action="store_true", help="only the proposal extensions")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    return p.parse_args()


def selected(args):
    if args.list:
        return []
    if args.steps:
        wanted = {s.lstrip("0") or "0" for s in args.steps}
        return [s for s in STEPS if (s[0].lstrip("0") or "0") in wanted]
    if args.paper:
        return [s for s in STEPS if s[2] == "paper"]
    if args.extensions:
        return [s for s in STEPS if s[2] == "extensions"]
    return STEPS


def run_step(script: str, fresh: bool) -> None:
    """Execute one step, in a new interpreter when it asks for one."""
    path = SCRIPTS / script
    if not fresh:
        runpy.run_path(str(path), run_name="__main__")
        return
    completed = subprocess.run([sys.executable, str(path)], cwd=str(SCRIPTS))
    if completed.returncode != 0:
        raise SystemExit(f"step {script} failed with exit code {completed.returncode}")


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available steps:\n")
        for number, script, group, description, fresh in STEPS:
            note = "  (own interpreter)" if fresh else ""
            print(f"  {number}  [{group:<10s}] {description}{note}\n      {script}")
        return 0

    steps = selected(args)
    if not steps:
        print("no steps matched", file=sys.stderr)
        return 2

    started = time.perf_counter()
    timings = []
    for number, script, _group, description, fresh in steps:
        print("\n" + "#" * 78)
        print(f"# STEP {number}: {description}")
        print(f"# {script}")
        print("#" * 78)
        t0 = time.perf_counter()
        run_step(script, fresh)
        timings.append((number, script, time.perf_counter() - t0))

    print("\n" + "=" * 78)
    print("ALL STEPS COMPLETE")
    print("=" * 78)
    for number, script, seconds in timings:
        print(f"  step {number}  {seconds:7.1f} s   {script}")
    print(f"\n  total     {time.perf_counter() - started:7.1f} s")
    print(f"\n  figures -> {ROOT / 'results' / 'figures'}")
    print(f"  tables  -> {ROOT / 'results' / 'tables'}")
    if any(number == "08" for number, _script, _seconds in timings):
        print(f"  chord   -> {ROOT / 'results' / 'extension_chord'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
