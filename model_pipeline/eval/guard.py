"""Clean-set sanity guard — the runtime test that the wired recipes are correct.

Loads each model, runs a (subsampled) CLEAN image slice through THAT model's own
preprocess, measures top-1, and checks it lands within tolerance of
`meta.expected_top1`. A wrong normalization / interpolation / label-order bug drops
top-1 by several points, so this fails fast BEFORE any GPU hours go into the
corruption sweep. It is the operational form of "no silent mismatch": compiling
proves syntax, this proves the model actually classifies.

Where do clean images come from? Whatever the corruption owner designates as
uncorrupted. By default the guard reads a `corruption="clean", severity=0` slice
from the same manifest (draw = the first configured seed); override via the
`guard.clean` config block — including `guard.clean.manifest` if clean images live
outside the corruption manifest.

Tolerance note: defaults are tuned to catch GROSS wiring errors (normalization
mismatch typically costs >=5 points), not to validate to 0.5%. On a 1000-image
subsample, binomial noise is ~1.3% (1 sigma) at top-1=0.76, so the default
tolerance of 5 points sits well clear of sampling noise. Tighten with larger n.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..models import registry
from ..models.spec import Classifier, ModelSpec
from . import metrics
from .loader import NoisyImageSet, load_manifest
from .loop import EvalConfig

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class GuardResult:
    model_key: str
    measured_top1: float          # percent
    expected_top1: Optional[float]
    n: int
    tolerance_pct: float
    passed: bool
    note: str = ""

    @property
    def floor(self) -> Optional[float]:
        return None if self.expected_top1 is None else self.expected_top1 - self.tolerance_pct


def resolve_clean_selector(cfg: dict) -> tuple[dict, str]:
    """Return (slice_selector, manifest_path) for the clean set.

    Defaults: corruption='clean', severity=0, draw=first configured seed, main manifest.
    Override any of these via cfg['guard']['clean'] (which may also carry 'manifest')."""
    gcfg = cfg.get("guard", {}) or {}
    raw = dict(gcfg.get("clean", {}) or {})
    manifest_path = raw.pop("manifest", cfg["manifest"])
    selector = {"corruption": "clean", "severity": 0, "draw": (cfg.get("seeds") or [0])[0]}
    selector.update(raw)
    return selector, manifest_path


def deterministic_subset(dataset: Dataset, n: Optional[int], seed: int) -> Dataset:
    """A reproducible n-sample subset (sorted indices, seeded choice). Returns the full
    dataset if n is None or >= len."""
    total = len(dataset)
    if n is None or n >= total:
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n].tolist()
    idx.sort()
    return Subset(dataset, idx)


@torch.inference_mode()
def measure_top1(classifier: Classifier, dataset: Dataset, *, device: torch.device, config: EvalConfig) -> tuple[float, int]:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dtype = _DTYPES[config.dtype]
    correct = 0.0
    n = 0
    for x, y, _ids in loader:
        x = x.to(device, dtype=dtype, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = classifier.logits(x).float()
        correct += metrics.topk_correct(logits, y, ks=(1,))[1]
        n += int(y.numel())
    return 100.0 * correct / max(n, 1), n


def _build_result(spec: ModelSpec, top1: float, n: int, tolerance_pct: float) -> GuardResult:
    exp = spec.meta.expected_top1
    if exp is None:
        return GuardResult(spec.key, top1, None, n, tolerance_pct, passed=True, note="no expected_top1 — not checked")
    passed = top1 >= exp - tolerance_pct
    note = ""
    if top1 > exp + tolerance_pct:
        note = f"exceeds expected+{tolerance_pct:g} — check label source / leakage"
    return GuardResult(spec.key, top1, exp, n, tolerance_pct, passed, note)


def evaluate_clean(
    classifier: Classifier,
    spec: ModelSpec,
    *,
    manifest,
    data_root: str | Path,
    clean: dict,
    config: EvalConfig,
    device: torch.device,
    n: int = 1000,
    subsample_seed: int = 0,
    tolerance_pct: float = 5.0,
) -> GuardResult:
    """Measure clean top-1 for an ALREADY-LOADED classifier (caller owns its lifetime).
    Used by the driver to guard a model with the same instance it is about to sweep."""
    ds = NoisyImageSet(manifest, data_root, clean["corruption"], clean["severity"], clean["draw"], spec.preprocess)
    ds = deterministic_subset(ds, n, subsample_seed)
    top1, n_eval = measure_top1(classifier, ds, device=device, config=config)
    return _build_result(spec, top1, n_eval, tolerance_pct)


def guard_clean_top1(
    model_key: str,
    *,
    manifest,
    data_root: str | Path,
    clean: dict,
    config: EvalConfig,
    device: torch.device,
    n: int = 1000,
    subsample_seed: int = 0,
    tolerance_pct: float = 5.0,
) -> GuardResult:
    """Load the model, measure clean top-1, free it. Standalone (no sweep)."""
    spec = registry.get(model_key)
    classifier = spec.load(device)
    try:
        return evaluate_clean(
            classifier, spec, manifest=manifest, data_root=data_root, clean=clean,
            config=config, device=device, n=n, subsample_seed=subsample_seed, tolerance_pct=tolerance_pct,
        )
    finally:
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()


def print_table(results: list[GuardResult]) -> None:
    print(f"{'model':18} {'measured':>9} {'expected':>9} {'floor':>7} {'n':>6}  status")
    for r in results:
        exp = "-" if r.expected_top1 is None else f"{r.expected_top1:.2f}"
        floor = "-" if r.floor is None else f"{r.floor:.2f}"
        status = "OK" if r.passed else "FAIL"
        tail = f"  {r.note}" if r.note else ""
        print(f"{r.model_key:18} {r.measured_top1:9.2f} {exp:>9} {floor:>7} {r.n:6d}  {status}{tail}")


def run_guards(cfg: dict, *, raise_on_fail: bool = True) -> list[GuardResult]:
    """Guard every configured model (loading each once). Prints a table; raises if any
    model falls below its floor (so it can gate a sweep). Returns all results."""
    device = torch.device(cfg["device"])
    eval_cfg = EvalConfig(**cfg.get("eval", {}))
    gcfg = cfg.get("guard", {}) or {}
    selector, manifest_path = resolve_clean_selector(cfg)
    manifest = load_manifest(Path(manifest_path))
    models = gcfg.get("models", cfg["models"])
    n = gcfg.get("n", 1000)
    tol = gcfg.get("tolerance_pct", 5.0)
    seed = gcfg.get("subsample_seed", 0)

    results = [
        guard_clean_top1(
            key, manifest=manifest, data_root=cfg["data_root"], clean=selector,
            config=eval_cfg, device=device, n=n, subsample_seed=seed, tolerance_pct=tol,
        )
        for key in models
    ]
    print_table(results)
    failed = [r for r in results if not r.passed]
    if failed and raise_on_fail:
        detail = ", ".join(f"{r.model_key}={r.measured_top1:.2f}<{r.floor:.2f}" for r in failed)
        raise RuntimeError(f"clean-set guard failed for {len(failed)} model(s): {detail}")
    return results


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Clean-set sanity guard for the wired model recipes")
    parser.add_argument("config", help="path to a YAML config (see configs/example.yaml)")
    parser.add_argument("--no-raise", action="store_true", help="report failures but exit 0")
    a = parser.parse_args()
    try:
        import yaml
    except ImportError:
        sys.exit("guard CLI requires pyyaml: pip install pyyaml")
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    results = run_guards(cfg, raise_on_fail=not a.no_raise)
    sys.exit(0 if all(r.passed for r in results) else 1)
