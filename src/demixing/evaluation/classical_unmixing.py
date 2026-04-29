from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from demixing.data.endmembers import EndmemberLibrary


UnmixingMethod = Literal["ols", "nnls"]


@dataclass(frozen=True)
class ClassicalUnmixingResult:
    component_names: tuple[str, ...]
    method: str
    coefficients: np.ndarray
    abundances: np.ndarray
    reconstructed: np.ndarray
    residual_l2: np.ndarray
    residual_rmse: np.ndarray
    residual_r2: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for index in range(self.coefficients.shape[0]):
            row: dict[str, float | int | str] = {
                "spectrum_index": index,
                "method": self.method,
                "residual_l2": float(self.residual_l2[index]),
                "residual_rmse": float(self.residual_rmse[index]),
                "residual_r2": float(self.residual_r2[index]),
            }
            for component_index, name in enumerate(self.component_names):
                row[f"coef_{name}"] = float(self.coefficients[index, component_index])
                row[f"abundance_{name}"] = float(self.abundances[index, component_index])
            rows.append(row)
        return pd.DataFrame(rows)


def _ensure_2d_spectra(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim == 1:
        return spectra.reshape(1, -1)
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 1D or 2D, got ndim={spectra.ndim}")
    return spectra


def _normalize_coefficients(coefficients: np.ndarray) -> np.ndarray:
    sums = coefficients.sum(axis=1, keepdims=True)
    return np.divide(
        coefficients,
        sums,
        out=np.zeros_like(coefficients, dtype=np.float32),
        where=sums > 0,
    )


def _compute_r2(spectra: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    centered = spectra - np.mean(spectra, axis=1, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=1)
    residual = spectra - reconstructed
    ss_res = np.sum(residual * residual, axis=1)
    return np.divide(
        ss_tot - ss_res,
        ss_tot,
        out=np.zeros_like(ss_tot, dtype=np.float32),
        where=ss_tot > 0,
    )


def solve_single_spectrum(
    spectrum: np.ndarray,
    library: EndmemberLibrary,
    method: UnmixingMethod = "nnls",
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float32)
    if spectrum.ndim != 1:
        raise ValueError("solve_single_spectrum expects a 1D spectrum.")
    if spectrum.shape[0] != library.n_points:
        raise ValueError(f"Spectrum length {spectrum.shape[0]} does not match library length {library.n_points}.")

    matrix = np.asarray(library.matrix, dtype=np.float32)
    if method == "nnls":
        coefficients, _ = nnls(matrix, spectrum)
        return coefficients.astype(np.float32, copy=False)
    if method == "ols":
        coefficients, _, _, _ = np.linalg.lstsq(matrix, spectrum, rcond=None)
        return coefficients.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported method={method!r}. Expected 'ols' or 'nnls'.")


def unmix_spectra(
    spectra: np.ndarray,
    library: EndmemberLibrary,
    method: UnmixingMethod = "nnls",
) -> ClassicalUnmixingResult:
    spectra = _ensure_2d_spectra(spectra)
    if spectra.shape[1] != library.n_points:
        raise ValueError(f"Spectra length {spectra.shape[1]} does not match library length {library.n_points}.")

    coefficients = np.vstack([solve_single_spectrum(row, library=library, method=method) for row in spectra]).astype(np.float32)
    reconstructed = coefficients @ library.matrix.T
    residual = spectra - reconstructed
    residual_l2 = np.linalg.norm(residual, axis=1).astype(np.float32)
    residual_rmse = np.sqrt(np.mean(residual * residual, axis=1)).astype(np.float32)
    residual_r2 = _compute_r2(spectra, reconstructed).astype(np.float32)
    abundances = _normalize_coefficients(coefficients)
    return ClassicalUnmixingResult(
        component_names=library.names,
        method=method,
        coefficients=coefficients,
        abundances=abundances,
        reconstructed=reconstructed.astype(np.float32, copy=False),
        residual_l2=residual_l2,
        residual_rmse=residual_rmse,
        residual_r2=residual_r2,
    )
