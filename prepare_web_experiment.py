#!/usr/bin/env python3
"""
Prepare web-experiment images from the corruption benchmark output.

Selects 1 source image per Imagenette class (10 total) plus 3 practice
images, then copies clean + 8 corruption types at severity 3 into
docs/images/, resized to JPEG for fast page loads.

Also writes docs/experiment_config.js so index.html can build its trial list.

Usage:
    python prepare_web_experiment.py [--output_dir output] [--docs_dir docs]
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

WNID_TO_CLASS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

CORRUPTION_DISPLAY = {
    "clean": "Clean",
    "gaussian_noise": "Gaussian Noise",
    "shot_noise": "Shot Noise",
    "jpeg_compression": "JPEG Compression",
    "pixelate": "Pixelation",
    "defocus_blur": "Defocus Blur",
    "motion_blur": "Motion Blur",
    "fog": "Fog",
    "contrast": "Contrast",
}

MAIN_SEVERITY = 3
IMG_MAX_SIZE = 400


def resize_and_save(src: Path, dst: Path, quality: int = 85) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = IMG_MAX_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(dst, "JPEG", quality=quality, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--docs_dir", default="docs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    docs = Path(args.docs_dir)
    img_dir = docs / "images"

    manifest = pd.read_parquet(out / "metadata" / "manifest.parquet")
    clean_df = manifest[manifest["corruption"] == "clean"].drop_duplicates("image_id")

    # 1 image per class for the main experiment
    main_ids = []
    for wnid in sorted(WNID_TO_CLASS.keys()):
        candidates = clean_df[clean_df["wnid"] == wnid]["image_id"].tolist()
        if candidates:
            main_ids.append(sorted(candidates)[0])

    # 3 practice images from the remainder
    remaining = [iid for iid in clean_df["image_id"].tolist() if iid not in main_ids]
    random.shuffle(remaining)
    practice_ids = remaining[:3]

    corruptions = sorted(c for c in manifest["corruption"].unique() if c != "clean")
    main_trials, practice_trials = [], []

    def build_record(iid, corruption, severity, web_path, label, class_name):
        return {
            "image_id": iid,
            "corruption": corruption,
            "severity": severity,
            "label": label,
            "class_name": class_name,
            "web_path": web_path,
        }

    for iid in main_ids + practice_ids:
        row = clean_df[clean_df["image_id"] == iid].iloc[0]
        label = int(row["label"])
        class_name = WNID_TO_CLASS[row["wnid"]]

        # Clean
        src = out / row["path"]
        web_path = f"images/clean/{iid}.jpg"
        resize_and_save(src, img_dir / "clean" / f"{iid}.jpg")
        rec = build_record(iid, "clean", 0, web_path, label, class_name)
        (practice_trials if iid in practice_ids else main_trials).append(rec)

        # Corrupted (main images only)
        if iid not in main_ids:
            continue
        for corruption in corruptions:
            rows = manifest[
                (manifest["image_id"] == iid)
                & (manifest["corruption"] == corruption)
                & (manifest["severity"] == MAIN_SEVERITY)
                & (manifest["draw"] == 0)
            ]
            if rows.empty:
                print(f"  WARNING: no row for {iid} / {corruption} / sev{MAIN_SEVERITY}")
                continue
            crow = rows.iloc[0]
            src = out / crow["path"]
            web_path = f"images/{corruption}/sev{MAIN_SEVERITY}/{iid}.jpg"
            dst = img_dir / corruption / f"sev{MAIN_SEVERITY}" / f"{iid}.jpg"
            resize_and_save(src, dst)
            main_trials.append(build_record(iid, corruption, MAIN_SEVERITY, web_path, label, class_name))

    config = {
        "trials": main_trials,
        "practice_trials": practice_trials,
        "class_names": [WNID_TO_CLASS[w] for w in sorted(WNID_TO_CLASS)],
        "corruption_display": CORRUPTION_DISPLAY,
    }
    (docs / "experiment_config.js").write_text(
        f"const EXPERIMENT_CONFIG = {json.dumps(config, indent=2)};\n"
    )

    total_images = len(main_trials) + len(practice_trials)
    print(f"Done.")
    print(f"  Main trials  : {len(main_trials):4d}  (10 images × {len(corruptions)+1} conditions)")
    print(f"  Practice     : {len(practice_trials):4d}")
    print(f"  JPEG files   : {total_images}")
    print(f"  Config       : {docs / 'experiment_config.js'}")


if __name__ == "__main__":
    main()
