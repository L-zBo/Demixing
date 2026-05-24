"""Plot PRISM tv_iters convergence/trade-off curves from prism_param_sweep results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization import apply_paper_style, plot_tv_iters_tradeoff


def write_summary(df: pd.DataFrame, datasets: list[tuple[str, str]], out_path: Path) -> None:
    rows = []
    for ds, ds_label in datasets:
        sub = df[df["dataset"] == ds]
        for ltv in sorted(sub["lambda_tv"].unique()):
            slc = sub[sub["lambda_tv"] == ltv].sort_values("tv_iters")
            row = {"dataset": ds_label, "lambda_tv": ltv}
            for _, r in slc.iterrows():
                ti = int(r["tv_iters"])
                row[f"mae_iter{ti}"] = round(float(r["mae"]), 4)
                row[f"pearson_iter{ti}"] = round(float(r["pearson_r"]), 3)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path.relative_to(ROOT)}")


def main() -> None:
    apply_paper_style()

    csv_path = ROOT / "outputs/experiments/prism_param_sweep/prism_param_sweep_full.csv"
    out_dir = ROOT / "outputs/showcase/prism_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["weight_mode"] == "endmember_std"].copy()
    df = df[df["lambda_l2"] == 1e-4]

    datasets = [("formal_v1_als_l2", "NOISY 40×40"),
                ("formal_v1_clean_als_l2", "CLEAN 40×40")]
    plot_tv_iters_tradeoff(df, out_dir / "prism_tv_iters_tradeoff.png", root_for_print=ROOT)
    write_summary(df, datasets, out_dir / "prism_tv_iters_summary.csv")


if __name__ == "__main__":
    main()
