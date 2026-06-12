#!/usr/bin/env python3
"""
build_benchmark.py — Spatial-frequency-organized object-detection corruption benchmark.

Samples N images from Pascal VOC 2007 (default) or COCO val2017, then generates
8 corruptions × 5 severities organized by dominant spatial-frequency band.

Usage:
    python build_benchmark.py
    python build_benchmark.py --output_dir ./output --num_images 50 --seed 42
    python build_benchmark.py --generate_adversarial --low_freq_attack --use_gpu
    python build_benchmark.py --dataset_name coco_val2017 --force
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision  # noqa: F401  (version check)
from PIL import Image
from tqdm import tqdm


# ============================================================
# FREQUENCY-AXIS CONFIG
# ============================================================

# (band, frequency_rank, description)
# frequency_rank orders corruptions from highest to lowest dominant freq.
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

CORRUPTION_ORDER: list[str] = list(FREQUENCY_CONFIG.keys())  # sorted by freq_rank

# Severity parameters, index 0 → severity 1, index 4 → severity 5
SEVERITY_PARAMS: dict[str, dict[str, list]] = {
    "gaussian_noise":   {"sigma":   [8, 16, 32, 52, 80]},
    "shot_noise":       {"scale":   [60, 40, 25, 15, 8]},     # lower = noisier
    "jpeg_compression": {"quality": [75, 58, 40, 25, 10]},    # lower = worse
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


# ============================================================
# DATASET LOADING
# ============================================================

def _parse_voc_xml(ann_path: str) -> list[dict]:
    """Parse a Pascal VOC XML annotation file → list of object dicts."""
    root = ET.parse(ann_path).getroot()
    objects: list[dict] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        diff_el  = obj.find("difficult")
        bndbox   = obj.find("bndbox")
        if name_el is None or bndbox is None:
            continue
        try:
            x1 = int(float(bndbox.find("xmin").text))
            y1 = int(float(bndbox.find("ymin").text))
            x2 = int(float(bndbox.find("xmax").text))
            y2 = int(float(bndbox.find("ymax").text))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            objects.append({
                "class":     name_el.text.strip(),
                "bbox":      [x1, y1, x2, y2],
                "difficult": int(diff_el.text.strip()) if diff_el is not None else 0,
            })
    return objects


def load_voc2007(dataset_root: Path) -> list[dict]:
    """Download (if needed) and return all VOC 2007 test samples with ≥1 object."""
    from torchvision.datasets import VOCDetection

    print("Loading Pascal VOC 2007 test split...")
    try:
        ds = VOCDetection(
            root=str(dataset_root),
            year="2007",
            image_set="test",
            download=True,
        )
    except Exception as exc:
        print(f"  VOCDetection failed: {exc}")
        return []

    samples: list[dict] = []
    for img_path, ann_path in zip(ds.images, ds.annotations):
        try:
            objects = _parse_voc_xml(ann_path)
        except Exception:
            continue
        if not objects:
            continue
        samples.append({
            "image_id":   Path(img_path).stem,
            "image_path": str(img_path),
            "objects":    objects,
            "dataset":    "voc2007",
            "split":      "test",
        })

    print(f"  {len(samples)} annotated images in VOC 2007 test.")
    return samples


def load_coco_val2017(dataset_root: Path) -> list[dict]:
    """Download (if needed) and return all COCO val2017 samples with ≥1 object."""
    coco_dir = dataset_root / "coco"
    img_dir  = coco_dir / "val2017"
    ann_file = coco_dir / "annotations" / "instances_val2017.json"

    if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
        img_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading COCO val2017 images (~1 GB)...")
        url  = "http://images.cocodataset.org/zips/val2017.zip"
        dest = coco_dir / "val2017.zip"
        urllib.request.urlretrieve(url, str(dest))
        with zipfile.ZipFile(dest) as z:
            z.extractall(coco_dir)
        dest.unlink()

    if not ann_file.exists():
        ann_file.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading COCO annotations...")
        url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        dest = coco_dir / "annotations.zip"
        urllib.request.urlretrieve(url, str(dest))
        with zipfile.ZipFile(dest) as z:
            z.extractall(coco_dir)
        dest.unlink()

    print("Parsing COCO val2017 annotations...")
    with open(ann_file) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    cat_map    = {cat["id"]: cat["name"]      for cat in coco["categories"]}

    img_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    samples: list[dict] = []
    for img_id, anns in img_to_anns.items():
        fname = id_to_file.get(img_id)
        if not fname:
            continue
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        objects = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                objects.append({
                    "class":     cat_map.get(ann["category_id"], "unknown"),
                    "bbox":      [int(x), int(y), int(x + w), int(y + h)],
                    "difficult": int(ann.get("iscrowd", 0)),
                })
        if objects:
            samples.append({
                "image_id":   str(img_id),
                "image_path": str(img_path),
                "objects":    objects,
                "dataset":    "coco_val2017",
                "split":      "val",
            })

    print(f"  {len(samples)} annotated images in COCO val2017.")
    return samples


def load_dataset(name: str, root: Path) -> list[dict]:
    if name == "voc2007":
        samples = load_voc2007(root)
        if not samples:
            print("VOC 2007 failed — falling back to COCO val2017.")
            samples = load_coco_val2017(root)
    else:
        samples = load_coco_val2017(root)
    return samples


def sample_images(samples: list[dict], n: int, seed: int) -> list[dict]:
    rng   = random.Random(seed)
    valid = [s for s in samples if s["objects"]]
    if len(valid) < n:
        raise ValueError(f"Need {n} images; only {len(valid)} available with objects.")
    return rng.sample(valid, n)


# ============================================================
# CORRUPTION FUNCTIONS
# Each: (img: np.ndarray uint8 H×W×3, severity: int 1-5) → np.ndarray uint8
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
    _, buf     = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    out_bgr    = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


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
    c      = length // 2
    rad    = math.radians(angle_deg)
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
    kernel = _motion_kernel(length, angle)
    out    = cv2.filter2D(img, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_fog(img: np.ndarray, severity: int) -> np.ndarray:
    alpha = SEVERITY_PARAMS["fog"]["alpha"][severity - 1]
    h     = img.shape[0]
    # Fog denser at the top (atmospheric scatter perspective)
    gradient  = np.linspace(alpha * 0.4, 0.0, h, dtype=np.float32)[:, None, None]
    blend     = np.clip(alpha + gradient, 0.0, 1.0)
    fog_color = np.full_like(img, 255, dtype=np.float32)
    out = img.astype(np.float32) * (1.0 - blend) + fog_color * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = SEVERITY_PARAMS["contrast"]["factor"][severity - 1]
    mean   = img.astype(np.float32).mean()
    out    = mean + (img.astype(np.float32) - mean) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


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


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

def make_output_dirs(out: Path) -> None:
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for name, (band, _, _) in FREQUENCY_CONFIG.items():
        for sev in range(1, NUM_SEVERITIES + 1):
            (out / BAND_DIR[band] / name / f"severity_{sev}").mkdir(parents=True, exist_ok=True)
    for atk in ["fgsm", "pgd", "low_freq_pgd"]:
        (out / "attacks" / atk).mkdir(parents=True, exist_ok=True)
    for sub in ["annotations", "metadata", "previews", "spectra"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


# ============================================================
# MANIFEST ROW BUILDER
# ============================================================

def _row(
    img_id: str, orig_path: str, out_path: Path,
    dataset: str, split: str,
    corruption_type: str, band: str, rank: int,
    severity: int, branch: str, attack_type: str | None,
    w: int, h: int, n_obj: int, classes: list[str],
) -> dict:
    return {
        "original_image_id": img_id,
        "original_path":     orig_path,
        "output_path":       str(out_path),
        "dataset_name":      dataset,
        "split_name":        split,
        "corruption_type":   corruption_type,
        "frequency_band":    band,
        "frequency_rank":    rank,
        "severity":          severity,
        "branch_type":       branch,
        "attack_type":       attack_type,
        "width":             w,
        "height":            h,
        "num_objects":       n_obj,
        "class_names":       json.dumps(classes),
    }


# ============================================================
# BENCHMARK GENERATION
# ============================================================

def generate_benchmark(
    samples: list[dict],
    output_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    make_output_dirs(output_dir)
    rows: list[dict] = []

    for sample in tqdm(samples, desc="Corrupting images"):
        img_id   = sample["image_id"]
        img_path = Path(sample["image_path"])
        img_np   = np.array(Image.open(img_path).convert("RGB"))
        h, w     = img_np.shape[:2]
        objects  = sample["objects"]
        classes  = [o["class"] for o in objects]
        dataset  = sample["dataset"]
        split    = sample["split"]

        # Annotations JSON (bboxes are unchanged; corruptions are non-geometric)
        ann_out = output_dir / "annotations" / f"{img_id}.json"
        if force or not ann_out.exists():
            ann_out.write_text(json.dumps({
                "image_id": img_id,
                "dataset":  dataset,
                "split":    split,
                "width":    w,
                "height":   h,
                "objects":  objects,
            }, indent=2))

        def _save(arr: np.ndarray, path: Path) -> None:
            if force or not path.exists():
                Image.fromarray(arr).save(path)

        # Clean copy
        clean_path = output_dir / "clean" / f"{img_id}.png"
        _save(img_np, clean_path)
        rows.append(_row(img_id, str(img_path), clean_path,
                         dataset, split, "clean", "none", 0, 0,
                         "clean", None, w, h, len(objects), classes))

        # 8 corruptions × 5 severities
        for name in CORRUPTION_ORDER:
            band, rank, _ = FREQUENCY_CONFIG[name]
            fn = CORRUPTION_FN[name]
            for sev in range(1, NUM_SEVERITIES + 1):
                out_path = (
                    output_dir / BAND_DIR[band] / name / f"severity_{sev}" / f"{img_id}.png"
                )
                if not force and out_path.exists():
                    rows.append(_row(img_id, str(img_path), out_path,
                                     dataset, split, name, band, rank, sev,
                                     "main", None, w, h, len(objects), classes))
                    continue
                corrupted = fn(img_np.copy(), sev)
                _save(corrupted, out_path)
                rows.append(_row(img_id, str(img_path), out_path,
                                 dataset, split, name, band, rank, sev,
                                 "main", None, w, h, len(objects), classes))

    return pd.DataFrame(rows)


# ============================================================
# ADVERSARIAL ATTACKS
# ============================================================

def _load_faster_rcnn(device: torch.device):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device).eval()
    return model


def _freeze_bn(model: torch.nn.Module) -> None:
    """Keep BN layers in eval mode so batch-size=1 doesn't blow up variance."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()


def _make_targets(objects: list[dict], img_hw: tuple[int, int],
                  device: torch.device) -> list[dict] | None:
    h, w = img_hw
    boxes, labels = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        x1 = max(0.0, min(float(x1), w - 1))
        y1 = max(0.0, min(float(y1), h - 1))
        x2 = max(0.0, min(float(x2), float(w)))
        y2 = max(0.0, min(float(y2), float(h)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
    if not boxes:
        return None
    return [{
        "boxes":  torch.tensor(boxes,  dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.int64,   device=device),
    }]


def _detection_loss(model, img_chw: torch.Tensor,
                    targets: list[dict]) -> torch.Tensor:
    """Forward in train mode (BN frozen) → total detection loss."""
    model.train()
    _freeze_bn(model)
    loss_dict = model([img_chw], targets)
    return sum(loss_dict.values())


def _lowpass(delta: torch.Tensor, cutoff: float = 0.1) -> torch.Tensor:
    """Low-pass filter a (C, H, W) perturbation via rfft2."""
    C, H, W = delta.shape
    d     = delta.unsqueeze(0)           # (1, C, H, W)
    fft   = torch.fft.rfft2(d)
    _, _, fH, fW = fft.shape
    hc    = max(1, int(fH * cutoff))
    wc    = max(1, int(fW * cutoff))
    mask  = torch.zeros_like(fft)
    mask[:, :, :hc,   :wc] = 1.0
    mask[:, :, -hc:,  :wc] = 1.0
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
        img_id   = sample["image_id"]
        img_path = sample["image_path"]
        img_np   = np.array(Image.open(img_path).convert("RGB"))
        h, w     = img_np.shape[:2]
        objects  = sample["objects"]
        classes  = [o["class"] for o in objects]
        dataset  = sample["dataset"]
        split    = sample["split"]

        targets = _make_targets(objects, (h, w), device)
        if targets is None:
            continue

        img_t = (torch.from_numpy(img_np)
                 .permute(2, 0, 1)
                 .float()
                 .div(255.0)
                 .to(device))  # (C, H, W)

        def _save_adv(adv: torch.Tensor, folder: str, attack: str) -> None:
            adv_np = (adv.clamp(0, 1)
                        .permute(1, 2, 0)
                        .cpu().numpy() * 255).astype(np.uint8)
            out_path = output_dir / "attacks" / folder / f"{img_id}.png"
            if force or not out_path.exists():
                Image.fromarray(adv_np).save(out_path)
            rows.append(_row(img_id, img_path, out_path,
                             dataset, split, attack, "adversarial", 99, 1,
                             "adversarial", attack, w, h, len(objects), classes))

        # --- FGSM ---
        try:
            x = img_t.clone().detach().requires_grad_(True)
            loss = _detection_loss(model, x, targets)
            loss.backward()
            with torch.no_grad():
                adv = (img_t + eps * x.grad.sign()).clamp(0, 1)
            _save_adv(adv, "fgsm", "fgsm")
        except Exception as exc:
            warnings.warn(f"FGSM failed for {img_id}: {exc}")

        # --- PGD (with optional low-freq constraint) ---
        try:
            delta = torch.zeros_like(img_t).uniform_(-eps, eps)
            for _ in range(pgd_steps):
                delta = delta.detach().requires_grad_(True)
                x     = (img_t + delta).clamp(0, 1)
                loss  = _detection_loss(model, x, targets)
                loss.backward()
                with torch.no_grad():
                    delta = delta + pgd_alpha * delta.grad.sign()
                    if low_freq:
                        delta = _lowpass(delta, cutoff=0.1)
                    delta = delta.clamp(-eps, eps)
            adv    = (img_t + delta.detach()).clamp(0, 1)
            folder = "low_freq_pgd" if low_freq else "pgd"
            _save_adv(adv, folder, folder)
        except Exception as exc:
            warnings.warn(f"PGD failed for {img_id}: {exc}")

    model.eval()
    return rows


# ============================================================
# VISUALIZATION
# ============================================================

def save_all_corruptions_grid(
    sample: dict, output_dir: Path, severity: int = 3
) -> None:
    """3×3 grid: clean + all 8 corruptions at one severity level."""
    img_id = sample["image_id"]
    img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    axes = axes.flatten()

    axes[0].imshow(img_np)
    axes[0].set_title("clean", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, name in enumerate(CORRUPTION_ORDER, start=1):
        band, _, _ = FREQUENCY_CONFIG[name]
        ax = axes[i]
        ax.imshow(CORRUPTION_FN[name](img_np.copy(), severity))
        ax.set_title(f"{name}\n[{band}] sev={severity}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{img_id} — all corruptions @ severity {severity}", fontsize=11)
    fig.savefig(output_dir / "previews" / f"{img_id}_all_sev{severity}.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_band_grid(
    samples: list[dict], output_dir: Path, severity: int = 3, n_imgs: int = 3
) -> None:
    """Rows = frequency band, cols = corruptions within each band."""
    band_corruptions = {
        b: [n for n in CORRUPTION_ORDER if FREQUENCY_CONFIG[n][0] == b]
        for b in BAND_ORDER
    }
    max_cols = max(len(v) for v in band_corruptions.values())

    for sample in samples[:n_imgs]:
        img_id = sample["image_id"]
        img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))

        fig, axes = plt.subplots(
            len(BAND_ORDER), max_cols,
            figsize=(max_cols * 5, len(BAND_ORDER) * 4),
            constrained_layout=True,
        )

        for row_i, band in enumerate(BAND_ORDER):
            corruptions = band_corruptions[band]
            for col_i in range(max_cols):
                ax = axes[row_i, col_i]
                if col_i < len(corruptions):
                    name = corruptions[col_i]
                    ax.imshow(CORRUPTION_FN[name](img_np.copy(), severity))
                    ax.set_title(name, fontsize=8)
                    if col_i == 0:
                        ax.set_ylabel(f"{band}-freq", fontsize=9, fontweight="bold")
                ax.axis("off")

        fig.suptitle(f"{img_id} — by frequency band @ severity {severity}", fontsize=11)
        fig.savefig(output_dir / "previews" / f"{img_id}_by_band_sev{severity}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


def save_adversarial_grid(
    samples: list[dict], output_dir: Path, n_imgs: int = 3
) -> None:
    """Clean vs. FGSM vs. PGD vs. low-freq PGD, one image per figure."""
    attack_folders = ["fgsm", "pgd", "low_freq_pgd"]
    for sample in samples[:n_imgs]:
        img_id = sample["image_id"]
        img_np = np.array(Image.open(sample["image_path"]).convert("RGB"))
        panels = [("clean", img_np)]
        for atk in attack_folders:
            p = output_dir / "attacks" / atk / f"{img_id}.png"
            if p.exists():
                panels.append((atk, np.array(Image.open(p).convert("RGB"))))
        if len(panels) == 1:
            continue

        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(5 * len(panels), 5),
                                 constrained_layout=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Adversarial: {img_id}", fontsize=11)
        fig.savefig(output_dir / "previews" / f"{img_id}_adversarial.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# FOURIER SPECTRA
# ============================================================

def _log_mag_spectrum(img_rgb: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(fshift))


def _avg_spectrum(
    images: list[np.ndarray], target_hw: tuple[int, int] = (512, 512)
) -> np.ndarray:
    H, W = target_hw
    stack = []
    for img in images:
        spec = _log_mag_spectrum(img)
        spec = cv2.resize(spec, (W, H), interpolation=cv2.INTER_LINEAR)
        stack.append(spec)
    return np.mean(stack, axis=0) if stack else np.zeros(target_hw)


def save_spectra(
    samples: list[dict], output_dir: Path, severity: int = 3, max_imgs: int = 20
) -> None:
    clean_imgs = [
        np.array(Image.open(s["image_path"]).convert("RGB"))
        for s in samples[:max_imgs]
    ]

    spectra: dict[str, np.ndarray] = {"clean": _avg_spectrum(clean_imgs)}
    for name in CORRUPTION_ORDER:
        fn = CORRUPTION_FN[name]
        spectra[name] = _avg_spectrum([fn(img.copy(), severity) for img in clean_imgs])

    # All-corruptions figure
    entries = list(spectra.items())
    ncols   = 5
    nrows   = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 4),
                              constrained_layout=True)
    axes = axes.flatten()
    for i, (name, spec) in enumerate(entries):
        axes[i].imshow(spec, cmap="inferno", origin="upper")
        label = name + (f"\n[{FREQUENCY_CONFIG[name][0]}]"
                        if name in FREQUENCY_CONFIG else "")
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
        agg = np.mean([spectra[n] for n in band_names], axis=0)
        ax.imshow(agg, cmap="inferno", origin="upper")
        ax.set_title(f"{band}-frequency band aggregate", fontsize=10)
        ax.axis("off")
    fig.suptitle("Band-aggregated Fourier spectra", fontsize=12)
    fig.savefig(output_dir / "spectra" / "band_aggregate.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a spatial-frequency-organized object-detection corruption benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_root",          default="./data",
                   help="Root directory for dataset downloads")
    p.add_argument("--output_dir",            default="./output",
                   help="Root directory for benchmark output")
    p.add_argument("--num_images",            type=int, default=50,
                   help="Number of images to sample")
    p.add_argument("--seed",                  type=int, default=42,
                   help="Random seed (Python, NumPy, PyTorch)")
    p.add_argument("--dataset_name",          default="voc2007",
                   choices=["voc2007", "coco_val2017"])
    p.add_argument("--use_gpu",               action="store_true",
                   help="Use GPU if available (falls back to CPU)")
    p.add_argument("--generate_adversarial",  action="store_true",
                   help="Generate FGSM and PGD adversarial examples")
    p.add_argument("--low_freq_attack",       action="store_true",
                   help="Constrain PGD perturbation to low spatial frequencies")
    p.add_argument("--force",                 action="store_true",
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
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"Device:   {device}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Dataset:  {args.dataset_name}")
    print(f"Output:   {output_dir.resolve()}")
    print()

    # 1. Load & sample ─────────────────────────────────────────────────────────
    all_samples = load_dataset(args.dataset_name, dataset_root)
    if not all_samples:
        print("ERROR: No samples loaded. Exiting.")
        sys.exit(1)

    sampled = sample_images(all_samples, args.num_images, args.seed)
    assert len(sampled) == args.num_images, (
        f"Expected {args.num_images} samples, got {len(sampled)}"
    )
    print(f"Sampled {len(sampled)} images (seed={args.seed}).\n")

    # 2. Corruption benchmark ──────────────────────────────────────────────────
    print("Generating corruption benchmark...")
    df = generate_benchmark(sampled, output_dir, force=args.force)
    print(f"  {len(df)} output images (1 clean + 8 corruptions × 5 severities = 41 per image).\n")

    # 3. Adversarial (optional) ────────────────────────────────────────────────
    adv_rows: list[dict] = []
    if args.generate_adversarial:
        print("Generating adversarial examples (this may be slow on CPU)...")
        try:
            adv_rows = run_adversarial(
                sampled, output_dir, device,
                low_freq=args.low_freq_attack,
                force=args.force,
            )
            print(f"  {len(adv_rows)} adversarial images generated.\n")
        except Exception as exc:
            warnings.warn(f"Adversarial generation failed: {exc}")

    # 4. Save manifest CSV ─────────────────────────────────────────────────────
    if adv_rows:
        df = pd.concat([df, pd.DataFrame(adv_rows)], ignore_index=True)
    manifest_path = output_dir / "metadata" / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Manifest → {manifest_path}  ({len(df)} rows)")

    # 5. Preview grids ─────────────────────────────────────────────────────────
    print("\nGenerating preview grids...")
    for s in tqdm(sampled[:3], desc="All-corruption grids"):
        save_all_corruptions_grid(s, output_dir, severity=3)
    save_band_grid(sampled, output_dir, severity=3, n_imgs=3)
    if adv_rows:
        save_adversarial_grid(sampled, output_dir, n_imgs=3)

    # 6. Fourier spectra ───────────────────────────────────────────────────────
    print("\nComputing Fourier magnitude spectra...")
    save_spectra(sampled, output_dir, severity=3)

    # 7. Summary ───────────────────────────────────────────────────────────────
    out_abs = output_dir.resolve()
    print(f"""
{'=' * 58}
BENCHMARK COMPLETE
{'=' * 58}
  Images sampled:      {len(sampled)}
  Corruption types:    {len(CORRUPTION_ORDER)} × {NUM_SEVERITIES} severities
  Total manifest rows: {len(df)}
  Output root:         {out_abs}
  Manifest CSV:        {manifest_path}
  Annotations (JSON):  {out_abs / 'annotations'}
  Preview grids:       {out_abs / 'previews'}
  Fourier spectra:     {out_abs / 'spectra'}"""
          + (f"\n  Adversarial output:  {out_abs / 'attacks'}" if adv_rows else "")
          + f"\n{'=' * 58}")


if __name__ == "__main__":
    main()
