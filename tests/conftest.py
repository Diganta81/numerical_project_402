from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from powerflow import load_case  # noqa: E402

SMALL_CASES = ["case3_saadat", "case5_stagg", "case6ww", "case14_ieee",
               "case24_ieee_rts", "case30_ieee"]
ALL_CASES = SMALL_CASES + ["case57_ieee", "case118_ieee"]


@pytest.fixture(params=SMALL_CASES)
def small_case(request):
    return load_case(request.param)


@pytest.fixture(params=ALL_CASES)
def any_case(request):
    return load_case(request.param)


@pytest.fixture
def case3():
    return load_case("case3_saadat")
