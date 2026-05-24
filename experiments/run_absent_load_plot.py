"""Plot absent-load false positive comparison: NNLS vs PRISM on 2 real samples.

Outputs:
    outputs/showcase/method_comparison/absent_load_false_positive_bars.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization import apply_paper_style, plot_absent_load_bars


def main() -> None:
    apply_paper_style()
    csv_path = ROOT / "outputs/showcase/method_comparison/prism_real_cross_sample.csv"
    out_path = ROOT / "outputs/showcase/method_comparison/absent_load_false_positive_bars.png"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["scenario"] == "absent_load"].copy()
    plot_absent_load_bars(df, out_path, root_for_print=ROOT)


if __name__ == "__main__":
    main()
