"""Driver: fan out the eval atom over models x corruptions x severities x seeds.

Owns everything that is NOT the pure atom:
  - model load lifetime: load each model ONCE, sweep all its slices, then free it;
  - resume: skip any (model, corruption, severity, seed) already in the table, so a
    model is never recomputed to redraw a figure (expensive work stays above the table);
  - provenance: snapshot git commit + config + manifest hash next to the results file.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import torch

from ..models import registry
from .loader import NoisyImageSet, load_manifest
from .loop import EvalConfig, evaluate


def _git_commit() -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None  # not a git repo (e.g. local prototype) — provenance degrades gracefully


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _done_keys(table_path: Path):
    if not table_path.exists():
        return set()
    import pandas as pd

    prev = pd.read_parquet(table_path)
    cols = ["model", "corruption", "severity", "seed"]
    return set(map(tuple, prev[cols].drop_duplicates().itertuples(index=False, name=None)))


def _append_rows(table_path: Path, rows) -> None:
    import pandas as pd

    # Append-and-rewrite: simple and crash-safe enough for this scale.
    # Swap for partitioned parquet if the table grows large.
    if table_path.exists():
        rows = pd.concat([pd.read_parquet(table_path), rows], ignore_index=True)
    rows.to_parquet(table_path, index=False)


def run(cfg: dict) -> None:
    device = torch.device(cfg["device"])
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / cfg["results_file"]
    run_id = cfg["run_id"]
    eval_cfg = EvalConfig(**cfg.get("eval", {}))

    manifest_path = Path(cfg["manifest"])
    manifest = load_manifest(manifest_path)

    # Provenance sidecar next to the results file.
    sidecar = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "manifest": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "config": cfg,
        "models": cfg["models"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (out_dir / f"{run_id}.provenance.json").write_text(json.dumps(sidecar, indent=2, default=str))

    done = _done_keys(table_path)
    grid = list(itertools.product(cfg["corruptions"], cfg["severities"], cfg["seeds"]))

    for model_key in cfg["models"]:
        spec = registry.get(model_key)
        classifier = spec.load(device)  # load ONCE per model
        # (optional) clean-set guard: measure top-1 vs spec.meta.expected_top1 before corruptions.
        try:
            for corruption, severity, seed in grid:
                if (model_key, corruption, severity, seed) in done:
                    continue  # never recompute a row just to redraw
                ds = NoisyImageSet(
                    manifest, cfg["data_root"], corruption, severity, seed, spec.preprocess
                )
                rows = evaluate(
                    model_key,
                    classifier,
                    ds,
                    corruption=corruption,
                    severity=severity,
                    seed=seed,
                    config=eval_cfg,
                    device=device,
                    run_id=run_id,
                )
                _append_rows(table_path, rows)
        finally:
            del classifier
            if device.type == "cuda":
                torch.cuda.empty_cache()
