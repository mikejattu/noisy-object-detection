"""Input seam: a loader over the pre-corrupted image set.

Manifest-authoritative (agreed contract): path / label / corruption / severity /
draw come from the manifest, never from parsing on-disk paths. This decouples me
from the corruption owner's directory scheme — they can re-lay-out freely.

`draw` is the replicate index that `seed` selects (corruptions are pre-rendered,
so the corruption RNG is theirs; my seed only picks which rendered draw runs).

Determinism: a slice is sorted by `image_id` (stable), so dataset ordering is a
pure function of (corruption, severity, seed) regardless of manifest row order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.spec import Preprocess

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("model_pipeline.eval.loader requires pandas (manifest is parquet/csv)") from exc


# Columns the manifest MUST provide (the input-seam contract).
REQUIRED_COLUMNS = ("image_id", "path", "label", "corruption", "severity", "draw")


class SeamError(RuntimeError):
    """Raised when the corruption manifest violates the agreed input seam."""


def load_manifest(manifest_path: str | Path) -> "pd.DataFrame":
    """Load and validate the manifest. Fails loudly on a broken contract."""
    p = Path(manifest_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:
        raise SeamError(f"unsupported manifest format {p.suffix!r} (expected .parquet/.csv/.tsv)")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SeamError(
            f"manifest {p} is missing required columns {missing}. "
            f"Note: 'draw' is the replicate index your seed selects — if it is absent, "
            f"coordinate with the corruption owner (how many draws N, and that draw is reproducible)."
        )
    return df


class NoisyImageSet(Dataset):
    """One (corruption, severity, draw=seed) slice of the pre-corrupted set.

    __getitem__ returns (preprocessed_tensor, label, image_id). The corruption is
    already baked into the image on disk; `preprocess` (the model-specific transform)
    runs strictly after it.
    """

    def __init__(
        self,
        manifest: "pd.DataFrame",
        data_root: str | Path,
        corruption: str,
        severity: int,
        seed: int,
        preprocess: Preprocess,
        expected_n: Optional[int] = None,
    ) -> None:
        sel = manifest[
            (manifest["corruption"] == corruption)
            & (manifest["severity"] == severity)
            & (manifest["draw"] == seed)
        ].sort_values("image_id", kind="stable").reset_index(drop=True)

        if len(sel) == 0:
            avail = sorted(
                manifest[
                    (manifest["corruption"] == corruption) & (manifest["severity"] == severity)
                ]["draw"].unique().tolist()
            )
            raise SeamError(
                f"empty slice for corruption={corruption!r} severity={severity} draw(seed)={seed}. "
                f"Available draws for this (corruption, severity): {avail}"
            )
        if expected_n is not None and len(sel) != expected_n:
            raise SeamError(
                f"slice {corruption!r}/sev{severity}/draw{seed} has {len(sel)} images, "
                f"expected {expected_n} — incomplete render?"
            )

        self._rows = sel
        self._root = Path(data_root)
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int):
        row = self._rows.iloc[i]
        img = Image.open(self._root / row["path"]).convert("RGB")  # corruption already applied
        x = self._preprocess(img)                                  # model-specific transform, AFTER corruption
        return x, int(row["label"]), str(row["image_id"])
