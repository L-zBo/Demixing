"""PRISM tv_iters convergence/trade-off curve — 3x2 metric/dataset grid.

Outputs:
    plot_tv_iters_tradeoff — write PNG to output_path; reads pre-filtered DataFrame
    from prism_param_sweep with columns dataset / lambda_tv / tv_iters / mae /
    pearson_r / spatial_tv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._plot_style import save_figure


def plot_tv_iters_tradeoff(df: pd.DataFrame, output_path: Path, *, root_for_print: Path | None = None) -> None:
    datasets = [("formal_v1_als_l2", "NOISY 40×40"),
                ("formal_v1_clean_als_l2", "CLEAN 40×40")]
    metrics = [("mae", "MAE ↓", "lower better"),
               ("pearson_r", "Pearson r ↑", "higher better"),
               ("spatial_tv", "Spatial TV ↓", "lower better")]

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(11, 9), sharex=True)

    for col, (ds, ds_label) in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        lambda_tvs = sorted(sub["lambda_tv"].unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_tvs)))
        for row, (metric, ylabel, _) in enumerate(metrics):
            ax = axes[row, col]
            for color, ltv in zip(cmap, lambda_tvs):
                slc = sub[sub["lambda_tv"] == ltv].sort_values("tv_iters")
                ax.plot(slc["tv_iters"], slc[metric], "o-",
                        color=color, label=f"λ_TV={ltv}")
            ax.axvline(2, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{ds_label}", fontweight="bold")
            if row == len(metrics) - 1:
                ax.set_xlabel("tv_iters")
            if row == 0 and col == len(datasets) - 1:
                ax.legend(loc="best", framealpha=0.9)

    fig.suptitle("PRISM tv_iters trade-off curve (red dashed = v1 default tv_iters=2)",
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    save_figure(fig, output_path, root_for_print=root_for_print)
