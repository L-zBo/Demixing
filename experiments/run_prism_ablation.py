"""PRISM 三模块 2³ 全因子消融：weight × L2 × TV 三个开关共 8 种组合。

Outputs:
    outputs/showcase/method_comparison/prism_ablation_summary.csv
    8 行：从 All-off (NNLS) 到 All-on，覆盖 leave-one-out 和 add-one/add-two/add-all。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import EndmemberLibrary
from preprocessing.preprocess import DEFAULT_PROTOCOL_NAME, SpectrumRecord, preprocess_record
from unmixing.unmix import prism_unmix_spectra, unmix_spectra


SYNTHETIC_ROOT = ROOT / "outputs/synthetic_unmixing/formal_v1_als_l2"
OUT_PATH = ROOT / "outputs/showcase/method_comparison/prism_ablation_summary.csv"


def load_synthetic_bundle(root: Path):
    import json
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    axis = np.load(root / "axis.npy").astype(np.float32)
    A = np.load(root / "endmember_matrix.npy").astype(np.float32)
    abundances = np.load(root / "abundances.npy").astype(np.float32)
    spectra = np.load(root / "spectra.npy").astype(np.float32)
    lib = EndmemberLibrary(
        names=tuple(meta["component_names"]),
        axis=axis, matrix=A, feature_mode="normalized",
        source_paths={n: Path(n) for n in meta["component_names"]},
    )
    return lib, abundances, spectra, int(meta["height"]), int(meta["width"])


def preprocess_spectra(axis: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    out = []
    for i, s in enumerate(spectra):
        rec = SpectrumRecord(
            relative_path=Path(f"synth_{i:05d}.csv"),
            axis=axis, intensity=s, axis_type="raman_shift_cm-1",
            source_format="synthetic",
            header_axis="RamanShift_cm-1", header_intensity="Intensity",
        )
        _, _, normalized, _ = preprocess_record(rec, protocol_name=DEFAULT_PROTOCOL_NAME)
        out.append(normalized)
    return np.stack(out).astype(np.float32)


def metrics(truth_flat, pred, recon, spectra_norm, H, W):
    err = pred - truth_flat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    rs = []
    for c in range(truth_flat.shape[1]):
        t, p = truth_flat[:, c], pred[:, c]
        if t.std() < 1e-12 or p.std() < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.mean((t - t.mean()) * (p - p.mean())) / (t.std() * p.std())))
    pearson = float(np.mean(rs))
    residual = spectra_norm - recon
    recon_rmse = float(np.sqrt(np.mean(residual * residual)))
    abundance_map = pred.reshape(H, W, -1)
    dx = np.abs(np.diff(abundance_map, axis=1))
    dy = np.abs(np.diff(abundance_map, axis=0))
    tv = float((dx.mean() + dy.mean()) / 2.0)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "pearson_r": round(pearson, 3), "recon_rmse": round(recon_rmse, 5),
            "spatial_tv": round(tv, 4)}


def row_norm(pred):
    clipped = np.clip(pred, 0.0, None)
    s = clipped.sum(axis=1, keepdims=True)
    safe = np.where(s > 1e-12, s, 1.0)
    return (clipped / safe).astype(np.float32)


def run_config(label, fn, spectra, truth_flat, recon_calc, H, W):
    t0 = time.perf_counter()
    pred_raw, recon = fn()
    elapsed = time.perf_counter() - t0
    m = metrics(truth_flat, row_norm(pred_raw), recon, spectra, H, W)
    m["elapsed_s"] = round(elapsed, 3)
    return {"config": label, **m}


def main():
    lib, truth_abundances, spectra_raw, H, W = load_synthetic_bundle(SYNTHETIC_ROOT)
    truth_flat = truth_abundances.reshape(-1, lib.n_endmembers)
    spectra_norm = preprocess_spectra(lib.axis, spectra_raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 2^3 = 8 种组合：weight ∈ {uniform, endmember_std} × L2 ∈ {0, 1e-2} × TV ∈ {off, on}
    configs = [
        ("All-off (NNLS)",        "uniform",       0.0,    0.0,  0),
        ("Only TV",               "uniform",       0.0,    0.10, 2),
        ("Only L2",               "uniform",       1e-2,   0.0,  0),
        ("Only Weight",           "endmember_std", 0.0,    0.0,  0),
        ("Weight+L2",             "endmember_std", 1e-2,   0.0,  0),
        ("Weight+TV",             "endmember_std", 0.0,    0.10, 2),
        ("L2+TV (Full v1, UNI)",  "uniform",       1e-2,   0.10, 2),
        ("All-on (Full STD)",     "endmember_std", 1e-2,   0.10, 2),
    ]

    rows = []
    for label, wm, l2, tv, ti in configs:
        if label == "All-off (NNLS)":
            rows.append(run_config(label,
                lambda: ((r := unmix_spectra(spectra_norm, lib, "nnls")).abundances, r.reconstructed),
                spectra_norm, truth_flat, None, H, W))
        else:
            img_shape = (H, W) if ti > 0 else None
            rows.append(run_config(label,
                lambda wm=wm, l2=l2, tv=tv, ti=ti, img_shape=img_shape: (
                    (r := prism_unmix_spectra(spectra_norm, lib,
                                              image_shape=img_shape,
                                              weight_mode=wm, lambda_l2=l2,
                                              lambda_tv=tv, tv_iters=ti)).abundances,
                    r.reconstructed),
                spectra_norm, truth_flat, None, H, W))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"=== PRISM 2^3 Full-Factorial Ablation (synthetic NOISY 40x40, PE/PP/starch) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
