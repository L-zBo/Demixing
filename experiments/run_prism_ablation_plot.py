"""Plot PRISM 2^3 ablation as 3-panel grouped bars (MAE / Pearson r / spatial_TV).

Outputs:
    outputs/showcase/method_comparison/prism_ablation_bars.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._plot_style import COLORS, apply_paper_style, save_figure


def main() -> None:
    apply_paper_style()

    csv_path = ROOT / "outputs/showcase/method_comparison/prism_ablation_summary.csv"
    out_path = ROOT / "outputs/showcase/method_comparison/prism_ablation_bars.png"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
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
    save_figure(fig, out_path, root_for_print=ROOT)


if __name__ == "__main__":
    main()
