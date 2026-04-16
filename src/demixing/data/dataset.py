from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RamanSpectrumDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest_csv: Path | None = None,
        use_normalized: bool = False,
        min_quality_tier: str | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.use_normalized = use_normalized
        self.samples: list[dict[str, object]] = []

        if manifest_csv is None:
            for csv_path in sorted(self.data_root.rglob("*.csv")):
                if "_reports" in csv_path.parts:
                    continue
                self.samples.append({"relative_path": csv_path.relative_to(self.data_root).as_posix(), "quality_tier": "A", "weight": 1.0})
        else:
            min_rank = {"A": 0, "B": 1, "C": 2}
            threshold = min_rank.get(min_quality_tier or "C", 2)
            with Path(manifest_csv).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                tier = row["quality_tier"]
                if min_rank[tier] > threshold:
                    continue
                self.samples.append(
                    {
                        "relative_path": row["relative_path"],
                        "quality_tier": tier,
                        "weight": float(row["recommended_weight"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        sample = self.samples[index]
        path = self.data_root / str(sample["relative_path"])
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = np.asarray(
            [
                float(row["Intensity_norm_max"] if self.use_normalized else row["Intensity_corrected"])
                for row in rows
            ],
            dtype=np.float32,
        )
        axis = np.asarray([float(row["RamanShift_cm-1"]) for row in rows], dtype=np.float32)
        return {
            "x": torch.from_numpy(values),
            "axis": torch.from_numpy(axis),
            "quality_tier": str(sample["quality_tier"]),
            "weight": float(sample["weight"]),
            "relative_path": str(sample["relative_path"]),
        }
