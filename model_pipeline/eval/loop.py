"""The pure eval atom.

`evaluate` is a pure function of (model, corruption, severity, seed): identical
inputs -> identical rows, independent of batch_size / device / num_workers (those
change throughput only). It takes an *already-loaded* classifier rather than a
model key so the driver can load each model once and sweep all its slices — the
classifier is itself a deterministic function of the model key, so purity holds.

Output: long/tidy rows, one per (model, corruption, severity, seed, metric),
bound to the column contract agreed with the analysis owner (RESULT_COLUMNS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from ..models.spec import Classifier
from . import metrics

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("model_pipeline.eval.loop requires pandas (tidy results table)") from exc


# Output seam: tidy column contract. DEFAULT — ratify exact names/dtypes with the analysis owner.
RESULT_COLUMNS = ("model", "corruption", "severity", "seed", "metric", "value", "n", "run_id")

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 256
    num_workers: int = 8
    metrics: tuple[str, ...] = ("top1", "top5", "nll", "ece")
    dtype: str = "fp32"          # fp32 for determinism; fp16/bf16 only if proven value-invariant
    ece_bins: int = 15


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    # warn_only so a missing deterministic kernel degrades to a warning, not a crash.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # For strict cuBLAS determinism also export CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch.


@torch.inference_mode()
def evaluate(
    model_key: str,
    classifier: Classifier,
    dataset: Dataset,
    *,
    corruption: str,
    severity: int,
    seed: int,
    config: EvalConfig,
    device: torch.device,
    run_id: str,
) -> "pd.DataFrame":
    _set_determinism(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,                       # deterministic order; the dataset is pre-sorted by image_id
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,                     # never silently drop the ragged last batch
    )
    dtype = _DTYPES[config.dtype]

    n = 0
    correct = {1: 0.0, 5: 0.0}
    nll = 0.0
    conf_sum = acc_sum = count = None

    for x, y, _ids in loader:
        x = x.to(device, dtype=dtype, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = classifier.logits(x).float()   # canonical imagenet1k order, asserted at load
        n += int(y.numel())

        ck = metrics.topk_correct(logits, y, ks=(1, 5))
        correct[1] += ck[1]
        correct[5] += ck[5]
        nll += metrics.nll_sum(logits, y)

        cs, as_, ct = metrics.ece_bin_stats(logits, y, n_bins=config.ece_bins)
        conf_sum = cs if conf_sum is None else conf_sum + cs
        acc_sum = as_ if acc_sum is None else acc_sum + as_
        count = ct if count is None else count + ct

    values = {
        "top1": correct[1] / n,
        "top5": correct[5] / n,
        "nll": nll / n,
        "ece": metrics.ece_from_bins(conf_sum, acc_sum, count),
    }
    rows = [
        {
            "model": model_key,
            "corruption": corruption,
            "severity": severity,
            "seed": seed,
            "metric": m,
            "value": values[m],
            "n": n,
            "run_id": run_id,
        }
        for m in config.metrics
    ]
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
