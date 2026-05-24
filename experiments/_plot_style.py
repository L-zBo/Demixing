"""Shared matplotlib style + colorblind-safe palette for paper/PPT figures.

Outputs:
    apply_paper_style() / apply_ppt_style() — set global rcParams in-place
    COLORS — Wong colorblind-safe palette, keyed by domain semantics
    save_figure() — unified mkdir + savefig + close + print
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Wong (2011) colorblind-safe 8-color palette
# https://www.nature.com/articles/nmeth.1618
_WONG = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Domain-semantic aliases (use these in plotting code, not raw _WONG keys)
COLORS = {
    # Endmembers
    "PE": _WONG["blue"],
    "PP": _WONG["orange"],
    "starch": _WONG["bluish_green"],
    # Methods
    "OLS": _WONG["yellow"],
    "NNLS": _WONG["sky_blue"],
    "FCLS": _WONG["bluish_green"],
    "NMF": _WONG["vermillion"],
    "MCR_hard": "#888888",
    "MCR_semi": "#bbbbbb",
    "PRISM": _WONG["reddish_purple"],
    # Roles
    "baseline": "#888888",
    "improvement": _WONG["reddish_purple"],
    "warning": _WONG["vermillion"],
    "neutral": _WONG["sky_blue"],
}

# Method order canonical for cross-figure consistency
METHOD_ORDER = ("OLS", "NNLS", "FCLS", "NMF", "MCR_hard", "MCR_semi", "PRISM")

# Cyclic palette for unnamed series (matplotlib prop_cycle)
PALETTE_CYCLE = [
    _WONG["blue"],
    _WONG["orange"],
    _WONG["bluish_green"],
    _WONG["vermillion"],
    _WONG["reddish_purple"],
    _WONG["sky_blue"],
    _WONG["yellow"],
    "#888888",
]


_BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """IEEE/Nature paper figure rcParams: small font, dpi 300."""
    plt.rcParams.update(_BASE_RC)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.prop_cycle": plt.cycler(color=PALETTE_CYCLE),
    })


def apply_ppt_style() -> None:
    """PPT figure rcParams: large font, dpi 200, higher contrast."""
    plt.rcParams.update(_BASE_RC)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.prop_cycle": plt.cycler(color=PALETTE_CYCLE),
    })


def save_figure(fig: Figure, out_path: Path, *, dpi: int | None = None, root_for_print: Path | None = None) -> None:
    """Create parent dir, save with bbox_inches=tight, close figure, print path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi} if dpi is not None else {}
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    rel = out_path.relative_to(root_for_print) if root_for_print else out_path
    print(f"Saved {rel}")


def method_color(method: str) -> str:
    """Lookup canonical color by method name (case-insensitive, falls back to neutral)."""
    key = method.replace("-", "_").upper()
    aliases = {k.upper(): v for k, v in COLORS.items()}
    return aliases.get(key, COLORS["neutral"])
