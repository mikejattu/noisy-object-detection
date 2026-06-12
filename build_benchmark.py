#!/usr/bin/env python3
"""
build_benchmark.py — Spatial-frequency-organized corruption benchmark.

Input-data contract (binding interface with the eval pipeline):
  Required manifest columns (exact names, exact dtypes):
    image_id   str    — stable source ID; equal across all (corruption, severity, draw) of one source
    path       str    — image path relative to output_dir  (= data_root in eval config)
    label      int    — ImageNet-1k synset-sorted class index, 0–999
    corruption str    — corruption name, or "clean"
    severity   int    — 0 for clean; 1–5 for corruptions
    draw       int    — replicate index; stochastic corruptions emit 0..num_draws-1;
                        deterministic corruptions emit draw=0 only (one physical file,
                        one manifest row per seed so the eval grid is always dense)

  Stochastic corruptions (gaussian_noise, shot_noise):
    N independent draws per (image, corruption, severity); each seeded deterministically by
      sha256(f"{image_id}:{corruption}:{severity}:{draw}")[:8] interpreted as hex → uint32
    so the dataset is byte-reproducible from the manifest alone.

  Deterministic corruptions (all others):
    Single draw=0; same physical file; manifest emits one row per configured seed
    (all pointing to the same path) so the eval grid (corruptions × severities × seeds) is dense.

  Primary output:  <output_dir>/metadata/manifest.parquet   (pyarrow)
  Secondary:       <output_dir>/metadata/manifest.csv

Dataset:
  --dataset_name imagenette (default) — 10 ImageNet-1k classes, auto-downloaded via
                                        torchvision.datasets.Imagenette; labels remapped
                                        to the correct ImageNet-1k synset-sorted indices.
  --dataset_name imagenet             — full ImageNet-1k val; requires --imagenet_root
                                        pointing to an ILSVRC-format directory.

Usage:
  python build_benchmark.py
  python build_benchmark.py --output_dir ./output --num_images 50 --num_draws 3 --seed 42
  python build_benchmark.py --dataset_name imagenet --imagenet_root /data/imagenet
  python build_benchmark.py --generate_adversarial --use_gpu --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision  # noqa: F401
from PIL import Image
from tqdm import tqdm


# ============================================================
# IMAGENET-1K LABEL MAPPING FOR IMAGENETTE
# Imagenette's wnid_to_idx gives 0-9; these are the correct
# synset-sorted ImageNet-1k positions for those 10 WNIDs.
# ============================================================

IMAGENETTE_WNID_TO_IN1K: dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


# ============================================================
# FREQUENCY-AXIS CONFIG
# ============================================================

# (band, frequency_rank, description)
# frequency_rank: ordering from highest to lowest dominant freq
FREQUENCY_CONFIG: dict[str, tuple[str, int, str]] = {
    "gaussian_noise":   ("high", 1, "Additive Gaussian white noise — broadband, high-freq dominant"),
    "shot_noise":       ("high", 2, "Poisson photon noise — broadband, high-freq dominant"),
    "jpeg_compression": ("high", 3, "JPEG quantization artifacts — high-freq blocking/ringing"),
    "pixelate":         ("mid",  4, "Downsample→upsample destroys mid/high spatial detail"),
    "defocus_blur":     ("mid",  5, "Gaussian PSF low-pass — attenuates mid and high frequencies"),
    "motion_blur":      ("mid",  6, "Directional motion blur — edge-band, directional freq loss"),
    "fog":              ("low",  7, "Atmospheric haze — global luminance veil, low-freq dominant"),
    "contrast":         ("low",  8, "Contrast reduction toward mean — purely global, low-freq"),
}

CORRUPTION_ORDER: list[str] = list(FREQUENCY_CONFIG.keys())

# Which corruptions involve randomness and need N independent draws.
STOCHASTIC_CORRUPTIONS: frozenset[str] = frozenset({"gaussian_noise", "shot_noise"})

SEVERITY_PARAMS: dict[str, dict[str, list]] = {
    "gaussian_noise":   {"sigma":   [8, 16, 32, 52, 80]},
    "shot_noise":       {"scale":   [60, 40, 25, 15, 8]},
    "jpeg_compression": {"quality": [75, 58, 40, 25, 10]},
    "pixelate":         {"factor":  [2, 4, 6, 8, 12]},
    "defocus_blur":     {"radius":  [2, 4, 6, 8, 10]},
    "motion_blur":      {"length":  [8, 14, 20, 26, 34],
                         "angle":   [0,  5, 10, 15, 20]},
    "fog":              {"alpha":   [0.15, 0.30, 0.45, 0.60, 0.75]},
    "contrast":         {"factor":  [0.85, 0.65, 0.45, 0.28, 0.12]},
}

BAND_ORDER = ["high", "mid", "low"]
BAND_DIR   = {"high": "high_frequency", "mid": "mid_frequency", "low": "low_frequency"}
NUM_SEVERITIES = 5


# ============================================================
# SEEDS
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def draw_rng_seed(image_id: str, corruption: str, severity: int, draw: int) -> int:
    """Deterministic numpy seed for one (image, corruption, severity, draw) triple.

    Logged key format makes the seeding scheme byte-reproducible from the manifest alone:
      sha256(f"{image_id}:{corruption}:{severity}:{draw}")[:8] → uint32
    """
    key = f"{image_id}:{corruption}:{severity}:{draw}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**32)


# ============================================================
# DATASET LOADING
# ============================================================

def load_imagenette(dataset_root: Path) -> list[dict]:
    """Auto-download Imagenette (full size) and return all val samples.

    Labels are remapped to ImageNet-1k synset-sorted indices (0-999) so
    the eval pipeline's top-1 measure is comparable to published baselines.
    """
    from torchvision.datasets import Imagenette

    print("Loading Imagenette val split (download=True, full size)...")
    try:
        ds = Imagenette(root=str(dataset_root), split="val", size="full", download=True)
    except Exception as exc:
        print(f"  Full-size Imagenette failed ({exc}); trying 320px...")
        try:
            ds = Imagenette(root=str(dataset_root), split="val", size="320px", download=True)
        except Exception as exc2:
            print(f"  320px Imagenette also failed: {exc2}")
            return []

    # _samples: list of (path_str, local_label 0-9)
    # ds.wnids: list of WNIDs in the order they were assigned local labels
    wnids = ds.wnids  # e.g. ["n01440764", "n02102040", ...]

    samples: list[dict] = []
    for path_str, local_label in ds._samples:
        wnid = wnids[local_label]
        in1k_label = IMAGENETTE_WNID_TO_IN1K.get(wnid)
        if in1k_label is None:
            continue  # unknown WNID — skip
        img_stem = Path(path_str).stem
        samples.append({
            "image_id":   img_stem,
            "image_path": str(path_str),
            "label":      in1k_label,
            "wnid":       wnid,
            "dataset":    "imagenette",
            "split":      "val",
        })

    print(f"  {len(samples)} val images in Imagenette ({len(ds.wnids)} classes).")
    return samples


def load_imagenet1k_val(imagenet_root: Path) -> list[dict]:
    """Load ImageNet-1k val from an ILSVRC-format directory.

    Expects: imagenet_root/val/{wnid}/*.JPEG  (devkit not required —
    torchvision.datasets.ImageNet only needs the image folders).
    Labels are the synset-sorted indices torchvision assigns, consistent
    with timm and the eval-pipeline model recipes.
    """
    from torchvision.datasets import ImageNet

    print(f"Loading ImageNet-1k val from {imagenet_root}...")
    try:
        ds = ImageNet(root=str(imagenet_root), split="val")
    except Exception as exc:
        print(f"  ImageNet load failed: {exc}")
        return []

    samples: list[dict] = []
    for path_str, label in ds.imgs:
        samples.append({
            "image_id":   Path(path_str).stem,
            "image_path": str(path_str),
            "label":      int(label),
            "wnid":       Path(path_str).parent.name,
            "dataset":    "imagenet1k",
            "split":      "val",
        })

    print(f"  {len(samples)} val images in ImageNet-1k.")
    return samples


def load_dataset(name: str, dataset_root: Path, imagenet_root: Path | None) -> list[dict]:
    if name == "imagenet":
        if imagenet_root is None:
            print("ERROR: --imagenet_root is required when --dataset_name imagenet")
            sys.exit(1)
        return load_imagenet1k_val(imagenet_root)
    else:
        return load_imagenette(dataset_root)


def sample_images(samples: list[dict], n: int, seed: int) -> list[dict]:
    rng   = random.Random(seed)
    if len(samples) < n:
        raise ValueError(f"Need {n} images but dataset only has {len(samples)}.")
    return rng.sample(samples, n)


# ============================================================
# CORRUPTION FUNCTIONS
# Each: (img: np.ndarray uint8 H×W×3, severity: int 1-5) → np.ndarray uint8
# Stochastic corruptions use np.random; caller seeds np.random before calling.
# ============================================================

def corrupt_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = SEVERITY_PARAMS["gaussian_noise"]["sigma"][severity - 1]
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def corrupt_shot_noise(img: np.ndarray, severity: int) -> np.ndarray:
    scale = SEVERITY_PARAMS["shot_noise"]["scale"][severity - 1]
    f     = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(f * scale).astype(np.float32) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def corrupt_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    quality = SEVERITY_PARAMS["jpeg_compression"]["quality"][severity - 1]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def corrupt_pixelate(img: np.ndarray, severity: int) -> np.ndarray:
    factor = SEVERITY_PARAMS["pixelate"]["factor"][severity - 1]
    h, w   = img.shape[:2]
    sh, sw = max(1, h // factor), max(1, w // factor)
    small  = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return  cv2.resize(small, (w,  h), interpolation=cv2.INTER_NEAREST)


def corrupt_defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = SEVERITY_PARAMS["defocus_blur"]["radius"][severity - 1]
    ksize  = 2 * radius + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=float(radius))


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    c, rad = length // 2, math.radians(angle_deg)
    for i in range(length):
        offset = i - c
        x = int(round(c + offset * math.cos(rad)))
        y = int(round(c - offset * math.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def corrupt_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    length = SEVERITY_PARAMS["motion_blur"]["length"][severity - 1]
    angle  = SEVERITY_PARAMS["motion_blur"]["angle"][severity - 1]
    return np.clip(cv2.filter2D(img, -1, _motion_kernel(length, angle)), 0, 255).astype(np.uint8)


def corrupt_fog(img: np.ndarray, severity: int) -> np.ndarray:
    alpha    = SEVERITY_PARAMS["fog"]["alpha"][severity - 1]
    h        = img.shape[0]
    gradient = np.linspace(alpha * 0.4, 0.0, h, dtype=np.float32)[:, None, None]
    blend    = np.clip(alpha + gradient, 0.0, 1.0)
    out = img.astype(np.float32) * (1.0 - blend) + np.full_like(img, 255, np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = SEVERITY_PARAMS["contrast"]["factor"][severity - 1]
    mean   = img.astype(np.float32).mean()
    return np.clip(mean + (img.astype(np.float32) - mean) * factor, 0, 255).astype(np.uint8)


CORRUPTION_FN: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise":   corrupt_gaussian_noise,
    "shot_noise":       corrupt_shot_noise,
    "jpeg_compression": corrupt_jpeg,
    "pixelate":         corrupt_pixelate,
    "defocus_blur":     corrupt_defocus_blur,
    "motion_blur":      corrupt_motion_blur,
    "fog":              corrupt_fog,
    "contrast":         corrupt_contrast,
}


def apply_corruption(img_np: np.ndarray, name: str, severity: int,
                     image_id: str, draw: int) -> np.ndarray:
    """Apply a single corruption, seeding np.random deterministically for stochastic ones."""
    if name in STOCHASTIC_CORRUPTIONS:
        np.random.seed(draw_rng_seed(image_id, name, severity, draw))
    return CORRUPTION_FN[name](img_np.copy(), severity)


# ============================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================

def make_output_dirs(out: Path, num_draws: int) -> None:
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for name, (band, _, _) in FREQUENCY_CONFIG.items():
        for sev in range(1, NUM_SEVERITIES + 1):
            if name in STOCHASTIC_CORRUPTIONS:
                # One draw_N/ subfolder per replicate; each holds exactly 50 files.
                for draw in range(num_draws):
                    (out / BAND_DIR[band] / name / f"severity_{sev}" / f"draw_{draw}").mkdir(
                        parents=True, exist_ok=True)
            else:
                (out / BAND_DIR[band] / name / f"severity_{sev}").mkdir(parents=True, exist_ok=True)
    for atk in ["fgsm", "pgd", "low_freq_pgd"]:
        (out / "attacks" / atk).mkdir(parents=True, exist_ok=True)
    for sub in ["annotations", "metadata", "previews", "spectra"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def _img_relpath(output_dir: Path, abs_path: Path) -> str:
    """Return the path relative to output_dir for storage in the manifest."""
    return str(abs_path.relative_to(output_dir))


# ============================================================
# MANIFEST ROW BUILDER
# Returns a dict with the 6 required contract columns + provenance extras.
# ============================================================

def _row(
    image_id: str, rel_path: str,
    label: int, corruption: str, severity: int, draw: int,
    # provenance extras
    original_path: str, dataset: str, split: str, wnid: str,
    frequency_band: str, frequency_rank: int,
    width: int, height: int,
) -> dict:
    return {
        # --- Required contract columns (exact names, exact dtypes) ---
        "image_id":       image_id,
        "path":           rel_path,
        "label":          int(label),
        "corruption":     corruption,
        "severity":       int(severity),
        "draw":           int(draw),
        # --- Provenance extras (allowed by contract; ignored by eval pipeline) ---
        "frequency_band": frequency_band,
        "frequency_rank": int(frequency_rank),
        "original_path":  original_path,
        "dataset_name":   dataset,
        "split_name":     split,
        "wnid":           wnid,
        "width":          int(width),
        "height":         int(height),
    }


# ============================================================
# BENCHMARK GENERATION
# ============================================================

def generate_benchmark(
    samples: list[dict],
    output_dir: Path,
    num_draws: int = 3,
    force: bool = False,
) -> pd.DataFrame:
    """Generate the full (corruption × severity × draw) grid and return the manifest.

    Every leaf folder contains exactly 50 images (one per source image):

    Stochastic corruptions (gaussian_noise, shot_noise):
      severity_N/draw_0/{image_id}.png   ← 50 images, independent noise draw 0
      severity_N/draw_1/{image_id}.png   ← 50 images, independent noise draw 1
      severity_N/draw_2/{image_id}.png   ← 50 images, independent noise draw 2
      Each draw seeded by sha256(image_id:corruption:severity:draw).

    Deterministic corruptions (all others):
      severity_N/{image_id}.png          ← 50 images, draw=0
      Manifest row replicated for each draw index so the eval grid is always dense.
    """
    make_output_dirs(output_dir, num_draws)
    rows: list[dict] = []

    for sample in tqdm(samples, desc="Corrupting images"):
        image_id = sample["image_id"]
        img_path = Path(sample["image_path"])
        label    = sample["label"]
        dataset  = sample["dataset"]
        split    = sample["split"]
        wnid     = sample.get("wnid", "")

        img_np = np.array(Image.open(img_path).convert("RGB"))
        h, w   = img_np.shape[:2]

        def _save(arr: np.ndarray, abs_path: Path) -> str:
            if force or not abs_path.exists():
                Image.fromarray(arr).save(abs_path)
            return _img_relpath(output_dir, abs_path)

        def add(rel: str, corruption: str, severity: int, draw: int,
                band: str = "none", rank: int = 0) -> None:
            rows.append(_row(image_id, rel, label, corruption, severity, draw,
                             str(img_path), dataset, split, wnid,
                             band, rank, w, h))

        # ── Clean (draw=0, severity=0) ────────────────────────────────────────
        clean_abs = output_dir / "clean" / f"{image_id}.png"
        rel = _save(img_np, clean_abs)
        add(rel, "clean", 0, 0)

        # ── 8 corruptions × 5 severities × draws ────────────────────────────
        for name in CORRUPTION_ORDER:
            band, rank, _ = FREQUENCY_CONFIG[name]
            sev_dir = output_dir / BAND_DIR[band] / name

            for sev in range(1, NUM_SEVERITIES + 1):

                if name in STOCHASTIC_CORRUPTIONS:
                    # Each draw gets its own subfolder; every folder holds exactly 50 files.
                    for draw in range(num_draws):
                        abs_path = sev_dir / f"severity_{sev}" / f"draw_{draw}" / f"{image_id}.png"
                        if force or not abs_path.exists():
                            corrupted = apply_corruption(img_np, name, sev, image_id, draw)
                            Image.fromarray(corrupted).save(abs_path)
                        rel = _img_relpath(output_dir, abs_path)
                        add(rel, name, sev, draw, band, rank)
                else:
                    # One file; manifest row replicated per draw so the eval grid
                    # (corruptions × severities × seeds) is always dense.
                    abs_path = sev_dir / f"severity_{sev}" / f"{image_id}.png"
                    if force or not abs_path.exists():
                        corrupted = apply_corruption(img_np, name, sev, image_id, 0)
                        Image.fromarray(corrupted).save(abs_path)
                    rel = _img_relpath(output_dir, abs_path)
                    for draw in range(num_draws):
                        add(rel, name, sev, draw, band, rank)

    df = pd.DataFrame(rows)
    # Enforce dtypes — the eval pipeline does integer equality filtering
    df["label"]    = df["label"].astype(int)
    df["severity"] = df["severity"].astype(int)
    df["draw"]     = df["draw"].astype(int)
    return df


# ============================================================
# MANIFEST EXPORT (parquet primary, CSV backup)
# ============================================================

def save_manifest(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)

    parquet_path = meta_dir / "manifest.parquet"
    csv_path     = meta_dir / "manifest.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


# ============================================================
# EVAL CONFIG (update example.yaml to match this benchmark)
# ============================================================

def write_eval_config(output_dir: Path, corruptions: list[str],
                      num_draws: int) -> Path:
    """Write a ready-to-use eval config for this benchmark run."""
    import yaml  # optional; only used for the convenience config writer

    seeds = list(range(num_draws))
    cfg = {
        "run_id": "freq-corruption-v1",
        "device": "cuda",
        "data_root": str(output_dir.resolve()),
        "manifest": str((output_dir / "metadata" / "manifest.parquet").resolve()),
        "results_dir": "model_pipeline/results",
        "results_file": "results.parquet",
        "models": [
            "resnet50_in1k", "resnet50_sin", "vit_b16_in1k",
            "convnext_b_in1k", "clip_vit_b16", "resnet50_augmix",
        ],
        "corruptions": corruptions,
        "severities": list(range(1, NUM_SEVERITIES + 1)),
        "seeds": seeds,
        "eval": {
            "batch_size": 256,
            "num_workers": 8,
            "metrics": ["top1", "top5", "nll", "ece"],
            "dtype": "fp32",
            "ece_bins": 15,
        },
        "guard": {
            "enabled": True,
            "n": 1000,
            "tolerance_pct": 5.0,
            "subsample_seed": 0,
            "clean": {"corruption": "clean", "severity": 0, "draw": 0},
        },
    }

    config_path = output_dir / "metadata" / "eval_config.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        return config_path
    except ImportError:
        warnings.warn("pyyaml not installed — skipping eval_config.yaml generation.")
        return config_path


# ============================================================
# ADVERSARIAL ATTACKS (optional branch; saves to attacks/)
# ============================================================

def _load_faster_rcnn(device: torch.device):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    return model.to(device).eval()


def _freeze_bn(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()


def _lowpass(delta: torch.Tensor, cutoff: float = 0.1) -> torch.Tensor:
    C, H, W = delta.shape
    fft  = torch.fft.rfft2(delta.unsqueeze(0))
    _, _, fH, fW = fft.shape
    hc, wc = max(1, int(fH * cutoff)), max(1, int(fW * cutoff))
    mask = torch.zeros_like(fft)
    mask[:, :, :hc, :wc] = 1.0
    mask[:, :, -hc:, :wc] = 1.0
    return torch.fft.irfft2(fft * mask, s=(H, W)).squeeze(0)


def run_adversarial(
    samples: list[dict],
    output_dir: Path,
    device: torch.device,
    low_freq: bool = False,
    force: bool = False,
    eps: float = 8.0 / 255.0,
    pgd_steps: int = 10,
    pgd_alpha: float = 2.0 / 255.0,
) -> list[dict]:
    try:
        model = _load_faster_rcnn(device)
    except Exception as exc:
        warnings.warn(f"Could not load Faster R-CNN: {exc}. Skipping adversarial.")
        return []

    rows: list[dict] = []

    for sample in tqdm(samples, desc="Adversarial"):
        image_id = sample["image_id"]
        img_np   = np.array(Image.open(sample["image_path"]).convert("RGB"))
        h, w     = img_np.shape[:2]
        label    = sample["label"]
        dataset  = sample["dataset"]
        split    = sample["split"]
        wnid     = sample.get("wnid", "")

        # Use bounding box spanning full image as a dummy target for loss computation
        targets = [{
            "boxes":  torch.tensor([[0.0, 0.0, float(w), float(h)]], device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
        }]

        img_t = (torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).to(device))

        def _save_adv(adv: torch.Tensor, folder: str, attack: str) -> None:
            adv_np = (adv.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            abs_path = output_dir / "attacks" / folder / f"{image_id}.png"
            if force or not abs_path.exists():
                Image.fromarray(adv_np).save(abs_path)
            rel = _img_relpath(output_dir, abs_path)
            rows.append(_row(image_id, rel, label, attack, 1, 0,
                             sample["image_path"], dataset, split, wnid,
                             "adversarial", 99, w, h))

        def _det_loss(x: torch.Tensor) -> torch.Tensor:
            model.train()
            _freeze_bn(model)
            loss_dict = model([x], targets)
            model.eval()
            return sum(loss_dict.values())

        # FGSM
        try:
            x = img_t.clone().detach().requires_grad_(True)
            _det_loss(x).backward()
            with torch.no_grad():
                _save_adv((img_t + eps * x.grad.sign()).clamp(0, 1), "fgsm", "fgsm")
        except Exception as exc:
            warnings.warn(f"FGSM failed for {image_id}: {exc}")

        # PGD (with optional low-freq constraint)
        try:
            delta = torch.zeros_like(img_t).uniform_(-eps, eps)
            for _ in range(pgd_steps):
                delta = delta.detach().requires_grad_(True)
                _det_loss((img_t + delta).clamp(0, 1)).backward()
                with torch.no_grad():
                    delta = delta + pgd_alpha * delta.grad.sign()
                    if low_freq:
                        delta = _lowpass(delta, cutoff=0.1)
                    delta = delta.clamp(-eps, eps)
            folder = "low_freq_pgd" if low_freq else "pgd"
            with torch.no_grad():
                _save_adv((img_t + delta.detach()).clamp(0, 1), folder, folder)
        except Exception as exc:
            warnings.warn(f"PGD failed for {image_id}: {exc}")

    model.eval()
    return rows


# ============================================================
# VISUALIZATION
# ============================================================

def save_all_corruptions_grid(sample: dict, output_dir: Path, severity: int = 3) -> None:
    """3×3 grid: clean + 8 corruptions at one representative severity."""
    img_id = sample["image_id"]
    img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    axes = axes.flatten()

    axes[0].imshow(img_np)
    axes[0].set_title("clean", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, name in enumerate(CORRUPTION_ORDER, start=1):
        band, _, _ = FREQUENCY_CONFIG[name]
        corrupted  = apply_corruption(img_np, name, severity, img_id, draw=0)
        axes[i].imshow(corrupted)
        axes[i].set_title(f"{name}\n[{band}] sev={severity}", fontsize=8)
        axes[i].axis("off")

    fig.suptitle(f"{img_id} (label={sample['label']}) — all corruptions @ sev {severity}", fontsize=10)
    fig.savefig(output_dir / "previews" / f"{img_id}_all_sev{severity}.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_band_grid(samples: list[dict], output_dir: Path, severity: int = 3, n: int = 3) -> None:
    """Rows = frequency band; cols = corruptions within that band."""
    band_corruptions = {
        b: [n for n in CORRUPTION_ORDER if FREQUENCY_CONFIG[n][0] == b]
        for b in BAND_ORDER
    }
    max_cols = max(len(v) for v in band_corruptions.values())

    for sample in samples[:n]:
        img_id = sample["image_id"]
        img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))
        fig, axes = plt.subplots(len(BAND_ORDER), max_cols,
                                 figsize=(max_cols * 5, len(BAND_ORDER) * 4),
                                 constrained_layout=True)
        for row_i, band in enumerate(BAND_ORDER):
            corruptions = band_corruptions[band]
            for col_i in range(max_cols):
                ax = axes[row_i, col_i]
                if col_i < len(corruptions):
                    name = corruptions[col_i]
                    ax.imshow(apply_corruption(img_np, name, severity, img_id, 0))
                    ax.set_title(name, fontsize=8)
                    if col_i == 0:
                        ax.set_ylabel(f"{band}-freq", fontsize=9, fontweight="bold")
                ax.axis("off")
        fig.suptitle(f"{img_id} — by frequency band @ sev {severity}", fontsize=10)
        fig.savefig(output_dir / "previews" / f"{img_id}_by_band_sev{severity}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


def save_stochastic_draws_grid(samples: list[dict], output_dir: Path,
                                severity: int = 3, n_imgs: int = 2) -> None:
    """Compare 3 draws of each stochastic corruption for the same image/severity."""
    for sample in samples[:n_imgs]:
        img_id = sample["image_id"]
        img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))
        names  = list(STOCHASTIC_CORRUPTIONS)
        n_rows = len(names)
        n_cols = 4   # clean + draw 0, 1, 2

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 5, n_rows * 4),
                                 constrained_layout=True)
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row_i, name in enumerate(names):
            axes[row_i, 0].imshow(img_np)
            axes[row_i, 0].set_title("clean", fontsize=8)
            axes[row_i, 0].set_ylabel(name, fontsize=9, fontweight="bold")
            axes[row_i, 0].axis("off")
            for draw in range(3):
                ax = axes[row_i, draw + 1]
                ax.imshow(apply_corruption(img_np, name, severity, img_id, draw))
                ax.set_title(f"draw={draw}", fontsize=8)
                ax.axis("off")

        fig.suptitle(f"{img_id} — stochastic draw comparison @ sev {severity}", fontsize=10)
        fig.savefig(output_dir / "previews" / f"{img_id}_draws_sev{severity}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


def save_adversarial_grid(samples: list[dict], output_dir: Path, n: int = 3) -> None:
    for sample in samples[:n]:
        img_id = sample["image_id"]
        img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))
        panels = [("clean", img_np)]
        for atk in ["fgsm", "pgd", "low_freq_pgd"]:
            p = output_dir / "attacks" / atk / f"{img_id}.png"
            if p.exists():
                panels.append((atk, np.array(Image.open(p).convert("RGB"))))
        if len(panels) == 1:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), constrained_layout=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Adversarial: {img_id}", fontsize=10)
        fig.savefig(output_dir / "previews" / f"{img_id}_adversarial.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# FOURIER SPECTRA (sanity check)
# ============================================================

def _log_mag_spectrum(img_rgb: np.ndarray, target_hw: tuple[int, int] = (512, 512)) -> np.ndarray:
    H, W  = target_hw
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    spec  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    return cv2.resize(spec, (W, H), interpolation=cv2.INTER_LINEAR)


def _avg_spectrum(images: list[np.ndarray]) -> np.ndarray:
    stack = [_log_mag_spectrum(img) for img in images]
    return np.mean(stack, axis=0) if stack else np.zeros((512, 512))


def save_spectra(samples: list[dict], output_dir: Path, severity: int = 3,
                 max_imgs: int = 20) -> None:
    clean_imgs = [
        np.array(Image.open(s["image_path"]).convert("RGB"))
        for s in samples[:max_imgs]
    ]
    spectra: dict[str, np.ndarray] = {"clean": _avg_spectrum(clean_imgs)}
    for name in CORRUPTION_ORDER:
        spectra[name] = _avg_spectrum([
            apply_corruption(img, name, severity, "", 0) for img in clean_imgs
        ])

    # All-corruptions figure
    entries = list(spectra.items())
    ncols = 5
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), constrained_layout=True)
    axes = axes.flatten()
    for i, (name, spec) in enumerate(entries):
        axes[i].imshow(spec, cmap="inferno", origin="upper")
        label = name + (f"\n[{FREQUENCY_CONFIG[name][0]}]" if name in FREQUENCY_CONFIG else "")
        axes[i].set_title(label, fontsize=8)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Avg log-magnitude Fourier spectra (sev={severity})", fontsize=12)
    fig.savefig(output_dir / "spectra" / f"all_corruptions_sev{severity}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Band-aggregated figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, band in zip(axes, BAND_ORDER):
        band_names = [n for n in CORRUPTION_ORDER if FREQUENCY_CONFIG[n][0] == band]
        ax.imshow(np.mean([spectra[n] for n in band_names], axis=0), cmap="inferno", origin="upper")
        ax.set_title(f"{band}-frequency band", fontsize=10)
        ax.axis("off")
    fig.suptitle("Band-aggregated Fourier spectra", fontsize=12)
    fig.savefig(output_dir / "spectra" / "band_aggregate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MANIFEST VALIDATION (fail-fast check before writing)
# ============================================================

REQUIRED_COLUMNS = ("image_id", "path", "label", "corruption", "severity", "draw")


def validate_manifest(df: pd.DataFrame, output_dir: Path, num_images: int, num_draws: int) -> None:
    """Smoke-test the manifest against the contract before saving."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"BUG: manifest missing required columns {missing}")

    # dtype check — the eval pipeline uses integer equality filtering
    for col in ("label", "severity", "draw"):
        if not pd.api.types.is_integer_dtype(df[col]):
            raise RuntimeError(f"BUG: column '{col}' must be integer dtype, got {df[col].dtype}")

    # label range
    bad_labels = df[(df["label"] < 0) | (df["label"] > 999)]
    if len(bad_labels):
        raise RuntimeError(f"BUG: {len(bad_labels)} rows have label outside 0-999")

    # all paths resolvable
    missing_files = [p for p in df["path"].unique() if not (output_dir / p).exists()]
    if missing_files:
        raise RuntimeError(f"BUG: {len(missing_files)} manifest paths not found on disk: {missing_files[:5]}")

    # clean slice sanity
    clean = df[(df["corruption"] == "clean") & (df["severity"] == 0) & (df["draw"] == 0)]
    if len(clean) != num_images:
        raise RuntimeError(
            f"BUG: expected {num_images} clean rows (corruption=clean, severity=0, draw=0), "
            f"got {len(clean)}"
        )

    # Every corruption/severity slice must have exactly num_draws draw indices
    for name in CORRUPTION_ORDER:
        slice_ = df[(df["corruption"] == name) & (df["severity"] == 1)]
        draws  = sorted(slice_["draw"].unique().tolist())
        if draws != list(range(num_draws)):
            raise RuntimeError(
                f"BUG: corruption {name!r} at severity 1 has draws {draws}, "
                f"expected {list(range(num_draws))}"
            )

    # Every leaf folder must contain exactly num_images files.
    # Stochastic: leaf = draw_N/  (num_draws subfolders per severity, each 50 files)
    # Deterministic: leaf = severity_N/  (50 files flat)
    for name in CORRUPTION_ORDER:
        band, _, _ = FREQUENCY_CONFIG[name]
        for sev in range(1, NUM_SEVERITIES + 1):
            sev_dir = output_dir / BAND_DIR[band] / name / f"severity_{sev}"
            if name in STOCHASTIC_CORRUPTIONS:
                for draw in range(num_draws):
                    folder = sev_dir / f"draw_{draw}"
                    files  = list(folder.glob("*.png"))
                    if len(files) != num_images:
                        raise RuntimeError(
                            f"BUG: {folder} has {len(files)} files, expected {num_images}"
                        )
            else:
                files = list(sev_dir.glob("*.png"))
                if len(files) != num_images:
                    raise RuntimeError(
                        f"BUG: {sev_dir} has {len(files)} files, expected {num_images}"
                    )

    print("  Manifest validation passed.")
    print(f"  clean/                       : {num_images} files")
    for name in CORRUPTION_ORDER:
        band, _, _ = FREQUENCY_CONFIG[name]
        if name in STOCHASTIC_CORRUPTIONS:
            leaf = output_dir / BAND_DIR[band] / name / "severity_1" / "draw_0"
            print(f"  {name:22}: {len(list(leaf.glob('*.png')))} files/draw folder × {num_draws} draws  (stochastic)")
        else:
            leaf = output_dir / BAND_DIR[band] / name / "severity_1"
            print(f"  {name:22}: {len(list(leaf.glob('*.png')))} files/severity folder  (deterministic)")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the spatial-frequency corruption benchmark (eval-pipeline contract v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_root",         default="./data",
                   help="Root dir for dataset download (Imagenette) or caching")
    p.add_argument("--output_dir",           default="./output",
                   help="Benchmark root (= data_root for the eval pipeline)")
    p.add_argument("--num_images",           type=int, default=50,
                   help="Source images to sample")
    p.add_argument("--num_draws",            type=int, default=3,
                   help="Replicate draws per stochastic corruption (must match eval config seeds list length)")
    p.add_argument("--seed",                 type=int, default=42,
                   help="Master sampling seed (Python / NumPy / PyTorch)")
    p.add_argument("--dataset_name",         default="imagenette",
                   choices=["imagenette", "imagenet"],
                   help="Source dataset (imagenette auto-downloads; imagenet requires --imagenet_root)")
    p.add_argument("--imagenet_root",        default=None,
                   help="Path to an ILSVRC-format ImageNet directory (required for --dataset_name imagenet)")
    p.add_argument("--use_gpu",              action="store_true",
                   help="Use GPU for adversarial attacks")
    p.add_argument("--generate_adversarial", action="store_true",
                   help="Generate FGSM and PGD adversarial examples (slow on CPU)")
    p.add_argument("--low_freq_attack",      action="store_true",
                   help="Constrain PGD perturbation to low spatial frequencies")
    p.add_argument("--force",                action="store_true",
                   help="Overwrite existing output files")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output_dir)
    imagenet_root = Path(args.imagenet_root) if args.imagenet_root else None
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Dataset:     {args.dataset_name}")
    print(f"Output root: {output_dir.resolve()}")
    print(f"data_root == output_dir (set this in eval config)")
    print()

    # 1. Load & sample ────────────────────────────────────────────────────────
    all_samples = load_dataset(args.dataset_name, dataset_root, imagenet_root)
    if not all_samples:
        print("ERROR: no samples loaded. Exiting.")
        sys.exit(1)

    sampled = sample_images(all_samples, args.num_images, args.seed)
    assert len(sampled) == args.num_images
    print(f"Sampled {len(sampled)} images (seed={args.seed}).")
    print(f"Label range: {min(s['label'] for s in sampled)}–{max(s['label'] for s in sampled)}")
    print()

    # 2. Corruption benchmark ─────────────────────────────────────────────────
    print("Generating corruption benchmark...")
    df = generate_benchmark(sampled, output_dir, num_draws=args.num_draws, force=args.force)

    # 3. Adversarial (optional) ───────────────────────────────────────────────
    adv_rows: list[dict] = []
    if args.generate_adversarial:
        print("\nGenerating adversarial examples (slow on CPU)...")
        try:
            adv_rows = run_adversarial(
                sampled, output_dir, device,
                low_freq=args.low_freq_attack, force=args.force,
            )
            print(f"  {len(adv_rows)} adversarial images generated.")
        except Exception as exc:
            warnings.warn(f"Adversarial generation failed: {exc}")

    if adv_rows:
        df = pd.concat([df, pd.DataFrame(adv_rows)], ignore_index=True)
        df["label"]    = df["label"].astype(int)
        df["severity"] = df["severity"].astype(int)
        df["draw"]     = df["draw"].astype(int)

    # 4. Validate manifest ────────────────────────────────────────────────────
    print("\nValidating manifest...")
    validate_manifest(df, output_dir, args.num_images, args.num_draws)

    # 5. Save manifest ────────────────────────────────────────────────────────
    parquet_path, csv_path = save_manifest(df, output_dir)
    print(f"Manifest → {parquet_path}  ({len(df)} rows)")
    print(f"         → {csv_path}  (CSV backup)")

    # 6. Write eval config ────────────────────────────────────────────────────
    try:
        config_path = write_eval_config(output_dir, CORRUPTION_ORDER, args.num_draws)
        print(f"Eval config → {config_path}")
    except Exception:
        pass

    # 7. Preview grids ────────────────────────────────────────────────────────
    print("\nGenerating preview grids...")
    for s in tqdm(sampled[:3], desc="All-corruption grids"):
        save_all_corruptions_grid(s, output_dir, severity=3)
    save_band_grid(sampled, output_dir, severity=3, n=3)
    save_stochastic_draws_grid(sampled, output_dir, severity=3, n_imgs=2)
    if adv_rows:
        save_adversarial_grid(sampled, output_dir, n=3)

    # 8. Fourier spectra ──────────────────────────────────────────────────────
    print("\nComputing Fourier magnitude spectra...")
    save_spectra(sampled, output_dir, severity=3)

    # 9. Summary ──────────────────────────────────────────────────────────────
    n_stochastic_files = sum(
        args.num_images * NUM_SEVERITIES * args.num_draws
        for name in CORRUPTION_ORDER if name in STOCHASTIC_CORRUPTIONS
    )
    n_det_files = sum(
        args.num_images * NUM_SEVERITIES
        for name in CORRUPTION_ORDER if name not in STOCHASTIC_CORRUPTIONS
    )
    n_clean_files = args.num_images

    out_abs = output_dir.resolve()
    print(f"""
{'=' * 60}
BENCHMARK COMPLETE
{'=' * 60}
  Source images:         {len(sampled)} ({args.dataset_name})
  Stochastic corruptions:{len(STOCHASTIC_CORRUPTIONS)} ({', '.join(sorted(STOCHASTIC_CORRUPTIONS))})
    draws per (image, severity): {args.num_draws}  →  {n_stochastic_files} files
  Deterministic corrs:   {len(CORRUPTION_ORDER) - len(STOCHASTIC_CORRUPTIONS)}
    draw=0 only            →  {n_det_files} files
  Clean files:           {n_clean_files}
  Manifest rows:         {len(df)}
  ── Eval pipeline binding ──────────────────────────────
  data_root:   {out_abs}
  manifest:    {out_abs / 'metadata' / 'manifest.parquet'}
  seeds:       {list(range(args.num_draws))}
  corruptions: {CORRUPTION_ORDER}
  ── Outputs ────────────────────────────────────────────
  Previews:    {out_abs / 'previews'}
  Spectra:     {out_abs / 'spectra'}"""
          + (f"\n  Attacks:     {out_abs / 'attacks'}" if adv_rows else "")
          + f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
