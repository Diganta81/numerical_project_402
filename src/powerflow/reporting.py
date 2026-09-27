from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np

TABLE_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


def _fmt(value, precision: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "--"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{precision - 1}e}"
        return f"{value:.{precision}g}"
    return str(value)


def to_markdown(df, precision: int = 4, index: bool = False) -> str:
    frame = df.reset_index() if index else df
    cols = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], precision) for c in cols) + " |")
    return "\n".join(lines)


def save_table(df, name: str, title: str = "", notes: str = "",
               out_dir: Optional[Path] = None, precision: int = 4,
               index: bool = False) -> Path:
    out_dir = Path(out_dir) if out_dir else TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=index)

    body = [f"# {title or name}", ""]
    if notes:
        body += [notes, ""]
    body += [to_markdown(df, precision=precision, index=index), ""]
    (out_dir / f"{name}.md").write_text("\n".join(body), encoding="utf-8")
    return csv_path


class Section:

    def __init__(self, title: str, stream=sys.stdout):
        self.title = title
        self.stream = stream

    def __enter__(self):
        print(f"\n{'=' * 78}\n{self.title}\n{'=' * 78}", file=self.stream)
        return self

    def __exit__(self, *exc):
        return False


def banner(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 72 - len(text)))


def print_table(df, precision: int = 4, index: bool = False) -> None:
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    rendered = [[_fmt(row[c], precision) for c in frame.columns] for _, row in frame.iterrows()]
    widths = [max(len(cols[j]), *(len(r[j]) for r in rendered)) if rendered else len(cols[j])
              for j in range(len(cols))]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rendered:
        print("  ".join(v.rjust(w) for v, w in zip(r, widths)))


def voltage_table(case, v, precision: int = 4):
    import pandas as pd

    kind = {1: "PQ", 2: "PV", 3: "slack"}
    return pd.DataFrame(
        {
            "bus": case.bus_ids,
            "type": [kind.get(int(t), "?") for t in case.bus_type],
            "Vm_pu": np.abs(v),
            "Va_deg": np.degrees(np.angle(v)),
            "V_rect": [f"{x.real:.4f}{x.imag:+.4f}j" for x in v],
        }
    )
