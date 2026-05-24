"""NNLS vs PRISM abundance maps + improvement heatmap (3 row x 5 col grid).

Outputs:
    plot_prism_vs_nnls_comparison — write PNG to output_path; takes raw arrays for
    truth / pred_nnls / pred_prism plus library + (height, width).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.endmembers import EndmemberLibrary
from visualization._plot_style import save_figure


def plot_prism_vs_nnls_comparison(
    dataset_name: str,
    library: EndmemberLibrary,
    truth: np.ndarray,
    pred_nnls: np.ndarray,
    pred_prism: np.ndarray,
    height: int,
    width: int,
    output_path: Path,
    *,
    root_for_print: Path | None = None,
) -> None:
    component_names = library.names
    n_components = len(component_names)
    truth_map = truth.reshape(height, width, n_components)
    nnls_map = pred_nnls.reshape(height, width, n_components)
    prism_map = pred_prism.reshape(height, width, n_components)

    nnls_err = np.abs(nnls_map - truth_map)
    prism_err = np.abs(prism_map - truth_map)
    improvement = nnls_err - prism_err

    fig, axes = plt.subplots(n_components, 5, figsize=(20, 4.0 * n_components))
    if n_components == 1:
        axes = axes[np.newaxis, :]

    abundance_vmin, abundance_vmax = 0.0, 1.0
    err_vmax = float(max(nnls_err.max(), prism_err.max()))
    improvement_vlim = float(np.abs(improvement).max())

    column_titles = ["Truth", "NNLS", "PRISM-full", "|NNLS−Truth|", "Improvement\n(NNLS err − PRISM err)"]

    for row, name in enumerate(component_names):
        for col, (data, vmin, vmax, cmap) in enumerate([
            (truth_map[..., row], abundance_vmin, abundance_vmax, "viridis"),
            (nnls_map[..., row], abundance_vmin, abundance_vmax, "viridis"),
            (prism_map[..., row], abundance_vmin, abundance_vmax, "viridis"),
            (nnls_err[..., row], 0.0, err_vmax, "magma"),
            (improvement[..., row], -improvement_vlim, improvement_vlim, "RdBu_r"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower", aspect="equal")
            if row == 0:
                ax.set_title(column_titles[col])
            if col == 0:
                ax.set_ylabel(name, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    nnls_mae = float(nnls_err.mean())
    prism_mae = float(prism_err.mean())
    rel_drop = (nnls_mae - prism_mae) / max(nnls_mae, 1e-12) * 100.0
    fig.suptitle(
        f"{dataset_name}  ({height}×{width} pixels)   "
        f"NNLS MAE={nnls_mae:.4f}   PRISM MAE={prism_mae:.4f}   relative drop={rel_drop:.1f}%",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path, dpi=140, root_for_print=root_for_print)
