"""Plot PRISM 2^3 ablation as 3-panel grouped bars (MAE / Pearson r / spatial_TV).

Outputs:
    outputs/showcase/method_comparison/prism_ablation_bars.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization import apply_paper_style, plot_prism_ablation_bars


def main() -> None:
    apply_paper_style()
    csv_path = ROOT / "outputs/showcase/method_comparison/prism_ablation_summary.csv"
    out_path = ROOT / "outputs/showcase/method_comparison/prism_ablation_bars.png"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    plot_prism_ablation_bars(df, out_path, root_for_print=ROOT)


if __name__ == "__main__":
    main()
