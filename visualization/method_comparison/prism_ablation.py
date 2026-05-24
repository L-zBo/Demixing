"""PRISM 2^3 ablation bar plot — 3-panel grouped bars across MAE / Pearson r / spatial_TV.

Outputs:
    plot_prism_ablation_bars — write PNG to output_path; reads pre-aggregated DataFrame
    with columns config / mae / pearson_r / spatial_tv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._plot_style import COLORS, save_figure


def plot_prism_ablation_bars(df: pd.DataFrame, output_path: Path, *, root_for_print: Path | None = None) -> None:
    labels = df["config"].tolist()
    n = len(labels)
    metrics = [
        ("mae", "MAE", "lower better"),
        ("pearson_r", "Pearson r", "higher better"),
        ("spatial_tv", "Spatial TV", "lower better"),
    ]

    bar_colors = []
    for label in labels:
        if "All-off" in label:
            bar_colors.append(COLORS["baseline"])
        elif "Full v1" in label:
            bar_colors.append(COLORS["PRISM"])
        elif "All-on" in label:
            bar_colors.append(COLORS["warning"])
        else:
            bar_colors.append(COLORS["NNLS"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x = np.arange(n)

    for ax, (col, ylabel, hint) in zip(axes, metrics):
        values = df[col].to_numpy()
        bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.6)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.3f}", ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({hint})", fontweight="bold")
        if col == "pearson_r":
            ax.axhline(0, color="k", linewidth=0.5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["baseline"], label="All-off (NNLS baseline)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["NNLS"], label="Partial PRISM"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["PRISM"], label="Full v1 (UNI, real-data default)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["warning"], label="All-on (STD, synth-only best)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("PRISM 2³ Ablation — Synthetic NOISY 40×40 (PE/PP/starch)",
                 fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_figure(fig, output_path, root_for_print=root_for_print)
