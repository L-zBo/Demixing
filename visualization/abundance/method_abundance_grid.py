"""Per-sample abundance grid — methods on rows, endmembers on columns.

Outputs:
    plot_method_abundance_grid — write PNG to output_path; takes pre-computed maps dict
    keyed by method name plus component_names tuple.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization._plot_style import save_figure


def plot_method_abundance_grid(
    sample_name: str,
    component_names: tuple[str, ...],
    maps: dict[str, np.ndarray],
    output_path: Path,
    *,
    method_order: tuple[str, ...] = ("NNLS", "PRISM_OLD", "PRISM_MID", "PRISM_AGG"),
    root_for_print: Path | None = None,
) -> None:
    n_rows = len(method_order)
    n_cols = len(component_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)
    for row, method in enumerate(method_order):
        amap = maps[method]
        for col, name in enumerate(component_names):
            ax = axes[row, col]
            im = ax.imshow(amap[..., col], vmin=0.0, vmax=1.0, cmap="viridis", origin="lower", aspect="equal")
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(method, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{sample_name} — abundance maps: NNLS vs PRISM variants")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path, dpi=140, root_for_print=root_for_print)
