from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _d in (FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = ["ROOT", "SRC", "DATA", "RESULTS", "FIG_DIR", "TABLE_DIR"]
