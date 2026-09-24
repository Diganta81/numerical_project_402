from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# --- categorical slots, in the validated order -------------------------------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Fixed method -> colour assignment.  Never cycle this.
METHOD_COLOR: Dict[str, str] = {
    "SNR": SLOTS[0],
    "PNR": SLOTS[1],
    "PNR+": SLOTS[2],
    "RCI": SLOTS[3],
    "FDLF-XB": SLOTS[4],
    "FDLF-BX": SLOTS[5],
    "GS": SLOTS[6],
}

#: Secondary encoding, so identity never rests on colour alone.
METHOD_MARKER: Dict[str, str] = {
    "SNR": "o", "PNR": "s", "PNR+": "D", "RCI": "^",
    "FDLF-XB": "v", "FDLF-BX": "P", "GS": "X",
}
METHOD_DASH: Dict[str, object] = {
    "SNR": "-", "PNR": "--", "PNR+": "-.", "RCI": (0, (4, 1, 1, 1)),
    "FDLF-XB": ":", "FDLF-BX": (0, (5, 2)), "GS": (0, (1, 1)),
}

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def use_paper_style() -> None:
    """Apply the shared rcParams.  Call once at the top of each script."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def style(key: str) -> dict:
    """Colour/marker/dash kwargs for one method."""
    return {
        "color": METHOD_COLOR.get(key, INK_SECONDARY),
        "marker": METHOD_MARKER.get(key, "o"),
        "linestyle": METHOD_DASH.get(key, "-"),
    }


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) if text else text


#: Point sizes used by the title block, so its height can be computed exactly.
SUBTITLE_PT = 8.5
TITLE_PT = 11.5


def _finish(fig, ax, title: str, subtitle: str = "") -> None:
    if not subtitle:
        ax.set_title(title, color=INK, loc="left", pad=8)
        ax.set_axisbelow(True)
        return

    # Wrap to the axes width so a long note cannot stretch the canvas.
    chars = max(int(fig.get_size_inches()[0] * 15), 40)
    wrapped = _wrap(subtitle, chars)
    ax.set_title(wrapped, color=INK_SECONDARY, fontsize=SUBTITLE_PT,
                 fontweight="normal", loc="left", pad=6)

    n_lines = wrapped.count("\n") + 1
    offset = 6 + n_lines * SUBTITLE_PT * 1.35 + 3
    ax.annotate(title, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, offset), textcoords="offset points",
                ha="left", va="bottom", color=INK,
                fontsize=TITLE_PT, fontweight="semibold")
    ax.set_axisbelow(True)


def save(fig, name: str, out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir) if out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Paper figures
# ---------------------------------------------------------------------------
def plot_flops(n, series: Mapping[str, np.ndarray], title: str, ylabel: str,
               name: str, out_dir=None) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, values in series.items():
        st = style(key) if key in METHOD_COLOR else {"color": SLOTS[len(ax.lines) % len(SLOTS)]}
        ax.loglog(n, values, label=key, markevery=max(len(np.atleast_1d(n)) // 8, 1), **st)
    ax.set_xlabel("Total number of buses, $n$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _finish(fig, ax, title)
    return save(fig, name, out_dir)


def plot_convergence(histories: Mapping[str, Sequence[float]], title: str, name: str,
                     ylabel: str = "Maximum voltage error (p.u.)", out_dir=None,
                     start_at: int = 1, subtitle: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, hist in histories.items():
        hist = np.asarray(hist, dtype=float)
        mask = np.isfinite(hist) & (hist > 0)
        idx = np.arange(len(hist)) + start_at
        ax.semilogy(idx[mask], hist[mask], label=key, **style(key))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="best")
    _finish(fig, ax, title, subtitle)
    return save(fig, name, out_dir)


def plot_grouped_bars(categories: Sequence[str], series: Mapping[str, Sequence[float]],
                      title: str, ylabel: str, name: str, out_dir=None,
                      subtitle: str = "", value_fmt: str = "{:.3g}",
                      log: bool = False, reference: Optional[float] = None,
                      reference_label: str = "") -> Path:
    """Grouped bar chart with direct value labels (base paper Figure 5)."""
    keys = list(series)
    n_g, n_s = len(categories), len(keys)
    x = np.arange(n_g)
    # 2px surface gap between adjacent bars.
    total = 0.78
    width = total / n_s
    gap = 0.02 * width

    fig, ax = plt.subplots(figsize=(1.35 * n_g + 2.6, 4.3))
    for i, key in enumerate(keys):
        offset = (i - (n_s - 1) / 2) * width
        bars = ax.bar(x + offset, series[key], width - gap, label=key,
                      color=METHOD_COLOR.get(key, SLOTS[i % len(SLOTS)]),
                      edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if n_s * n_g <= 24:
            ax.bar_label(bars, fmt=value_fmt, padding=2, fontsize=7.5, color=INK_SECONDARY)

    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=2)
        ax.annotate(reference_label, xy=(1.0, reference), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=INK_MUTED)

    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=min(n_s, 4))
    _finish(fig, ax, title, subtitle)
    return save(fig, name, out_dir)


def plot_scaling(x, series: Mapping[str, Sequence[float]], title: str, ylabel: str,
                 name: str, xlabel: str = "Total number of buses, $n$",
                 out_dir=None, subtitle: str = "", loglog: bool = True,
                 xlog: bool = False,
                 point_labels: Optional[Sequence[str]] = None) -> Path:

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plot = ax.loglog if loglog else (ax.semilogx if xlog else ax.plot)
    for key, values in series.items():
        plot(x, values, label=key, **style(key))
    if point_labels is not None:
        top = max(np.max(np.asarray(v, dtype=float)) for v in series.values())
        for xi, lab in zip(x, point_labels):
            ax.annotate(lab, xy=(xi, top), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_MUTED, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _finish(fig, ax, title, subtitle)
    return save(fig, name, out_dir)


def plot_heatmap(matrix, row_labels, col_labels, title: str, name: str,
                 cbar_label: str = "", out_dir=None, subtitle: str = "",
                 fmt: str = "{:.0f}", mask_value=None) -> Path:
    
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list(
        "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    )
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(0.62 * len(col_labels) + 3.2, 0.46 * len(row_labels) + 2.4))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=blues, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8.5)
    ax.grid(visible=False)
    vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = "--" if not np.isfinite(val) else fmt.format(val)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if np.isfinite(val) and val > 0.62 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, color=INK_SECONDARY, fontsize=9)
    cb.outline.set_visible(False)
    _finish(fig, ax, title, subtitle)
    return save(fig, name, out_dir)
