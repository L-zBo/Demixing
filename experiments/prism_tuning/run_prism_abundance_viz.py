"""Plot NNLS vs PRISM abundance maps + improvement heatmap on synthetic ground-truth datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import EndmemberLibrary
from preprocessing.preprocess import DEFAULT_PROTOCOL_NAME, PREPROCESS_PROTOCOLS, SpectrumRecord, preprocess_record
from unmixing.unmix import prism_unmix_spectra, unmix_spectra
from visualization import apply_paper_style, plot_prism_vs_nnls_comparison


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/prism_abundance_viz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-roots",
        nargs="+",
        type=Path,
        default=[
            ROOT / "outputs/synthetic_unmixing/formal_v1_als_l2",
            ROOT / "outputs/synthetic_unmixing/formal_v1_clean_als_l2",
        ],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--lambda-l2", type=float, default=1e-2)
    parser.add_argument("--lambda-tv", type=float, default=0.10)
    parser.add_argument("--tv-iters", type=int, default=2)
    return parser.parse_args()


def load_synthetic_bundle(synthetic_root: Path):
    metadata = json.loads((synthetic_root / "metadata.json").read_text(encoding="utf-8"))
    component_names = tuple(metadata["component_names"])
    axis = np.load(synthetic_root / "axis.npy")
    endmember_matrix = np.load(synthetic_root / "endmember_matrix.npy")
    abundances = np.load(synthetic_root / "abundances.npy")
    spectra = np.load(synthetic_root / "spectra.npy")
    library = EndmemberLibrary(
        names=component_names,
        axis=axis.astype(np.float32),
        matrix=endmember_matrix.astype(np.float32),
        feature_mode="normalized",
        source_paths={name: Path(name) for name in component_names},
    )
    return library, abundances.astype(np.float32), spectra.astype(np.float32), int(metadata["height"]), int(metadata["width"])


def preprocess_synthetic_spectra(axis: np.ndarray, spectra: np.ndarray, protocol_name: str) -> np.ndarray:
    processed = []
    for index, spectrum in enumerate(spectra):
        record = SpectrumRecord(
            relative_path=Path(f"synthetic_{index:05d}.csv"),
            axis=axis,
            intensity=spectrum,
            axis_type="raman_shift_cm-1",
            source_format="synthetic",
            header_axis="RamanShift_cm-1",
            header_intensity="Intensity",
        )
        _, _, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        processed.append(normalized)
    return np.stack(processed).astype(np.float32)


def row_normalize_nonneg(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, None)
    row_sum = clipped.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-12, row_sum, 1.0)
    return (clipped / safe).astype(np.float32)


def main() -> None:
    apply_paper_style()
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for synthetic_root in args.synthetic_roots:
        library, truth_abundances, spectra_raw, height, width = load_synthetic_bundle(synthetic_root)
        spectra_norm = preprocess_synthetic_spectra(library.axis, spectra_raw, protocol_name=args.protocol)
        truth_flat = truth_abundances.reshape(-1, library.n_endmembers)

        nnls_result = unmix_spectra(spectra_norm, library=library, method="nnls")
        prism_result = prism_unmix_spectra(
            spectra_norm,
            library=library,
            image_shape=(height, width),
            lambda_l2=args.lambda_l2,
            lambda_tv=args.lambda_tv,
            tv_iters=args.tv_iters,
        )

        pred_nnls = row_normalize_nonneg(nnls_result.abundances)
        pred_prism = row_normalize_nonneg(prism_result.abundances)

        output_path = args.output_root / f"{synthetic_root.name}_prism_vs_nnls.png"
        plot_prism_vs_nnls_comparison(
            dataset_name=synthetic_root.name,
            library=library,
            truth=truth_flat,
            pred_nnls=pred_nnls,
            pred_prism=pred_prism,
            height=height,
            width=width,
            output_path=output_path,
            root_for_print=ROOT,
        )


if __name__ == "__main__":
    main()
