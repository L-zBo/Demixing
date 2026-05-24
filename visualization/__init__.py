"""Visualization plots organized by figure type (mirrors LA_perfect layout).

Top-level re-exports keep the legacy ``from visualization import plot_xxx`` callsite
working while the implementation lives in dedicated subpackages by figure type.
"""
from visualization._plot_style import (
    COLORS,
    METHOD_ORDER,
    PALETTE_CYCLE,
    apply_paper_style,
    apply_ppt_style,
    method_color,
    save_figure,
)
from visualization.abundance import (
    plot_abundance_maps,
    plot_method_abundance_grid,
    plot_prism_vs_nnls_comparison,
)
from visualization.method_comparison import (
    plot_absent_load_bars,
    plot_method_abundance_bars,
    plot_method_metric_bars,
    plot_negative_abundance_pct_bars,
    plot_nmf_endmember_sam_bars,
    plot_prism_ablation_bars,
    plot_synthetic_metric_subplots,
)
from visualization.preprocessing import (
    plot_endmember_fingerprints,
    plot_fingerprint_retention_bars,
    plot_protocol_abundance_grid,
    plot_protocol_cv_bars,
    plot_protocol_reconstruction_r2_bars,
    plot_protocol_spectrum_triptych,
    plot_single_spectrum_preprocessing,
)
from visualization.prism_convergence import plot_tv_iters_tradeoff
from visualization.reconstruction import plot_reconstruction_examples
from visualization.residual import plot_residual_map

__all__ = [
    "COLORS",
    "METHOD_ORDER",
    "PALETTE_CYCLE",
    "apply_paper_style",
    "apply_ppt_style",
    "method_color",
    "plot_abundance_maps",
    "plot_absent_load_bars",
    "plot_endmember_fingerprints",
    "plot_fingerprint_retention_bars",
    "plot_method_abundance_bars",
    "plot_method_abundance_grid",
    "plot_method_metric_bars",
    "plot_negative_abundance_pct_bars",
    "plot_nmf_endmember_sam_bars",
    "plot_prism_ablation_bars",
    "plot_prism_vs_nnls_comparison",
    "plot_protocol_abundance_grid",
    "plot_protocol_cv_bars",
    "plot_protocol_reconstruction_r2_bars",
    "plot_protocol_spectrum_triptych",
    "plot_reconstruction_examples",
    "plot_residual_map",
    "plot_single_spectrum_preprocessing",
    "plot_synthetic_metric_subplots",
    "plot_tv_iters_tradeoff",
    "save_figure",
]
