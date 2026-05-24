"""Absent-load false positive comparison — 3-panel grouped bars: NNLS vs PRISM.

Outputs:
    plot_absent_load_bars — write PNG to output_path; reads pre-filtered DataFrame
    with columns sample / absent_endmember / method / max_abundance_absent /
    mean_abundance_absent / frac_pixels_absent_over_10pct.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._plot_style import COLORS, save_figure


def plot_absent_load_bars(df: pd.DataFrame, output_path: Path, *, root_for_print: Path | None = None) -> None:
    df = df.copy()
    df["sample_label"] = df["sample"] + "\n(absent: " + df["absent_endmember"] + ")"
    samples = df["sample_label"].unique().tolist()
    methods = ["NNLS", "PRISM_MID_UNI"]
    method_colors = {"NNLS": COLORS["NNLS"], "PRISM_MID_UNI": COLORS["PRISM"]}
    method_labels = {"NNLS": "NNLS", "PRISM_MID_UNI": "PRISM (v1 UNI)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_specs = [
        ("max_abundance_absent", "Max abundance of absent endmember", "↓ better"),
        ("mean_abundance_absent", "Mean abundance of absent endmember", "↓ better"),
        ("frac_pixels_absent_over_10pct", "Fraction pixels > 10% (false-positive rate)", "↓ better, 0 = perfect"),
    ]

    bar_width = 0.35
    x = np.arange(len(samples))

    for ax, (col, ylabel, hint) in zip(axes, metric_specs):
        for i, method in enumerate(methods):
            sub = df[df["method"] == method].sort_values("sample")
            values = sub[col].to_numpy()
            offset = (i - 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width,
                          label=method_labels[method], color=method_colors[method],
                          edgecolor="black", linewidth=0.6)
            for rect, v in zip(bars, values):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{v:.4f}", ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(samples)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({hint})", fontweight="bold")
        ax.legend(loc="best")

    fig.suptitle("Absent-load False Positive — NNLS vs PRISM on 2 real samples",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, output_path, root_for_print=root_for_print)
