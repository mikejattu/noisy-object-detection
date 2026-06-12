"""
Stage 3 — ConvNet Representation Extraction
============================================
Extracts layer-wise activations from a pretrained ResNet50 for a dataset of
object images organised by difficulty condition (difficult: 0 or 1).

Dataset structure
-----------------
Annotations and images share the same folder:
    output/annotations/000098.json   ← annotation
    output/annotations/000098.jpg    ← corresponding image

Each annotation JSON has the structure:
    {
        "image_id": "000098",
        "dataset":  "voc2007",
        "objects": [
            { "class": "cat", "difficult": 0, ... }
        ]
    }

The 'difficult' flag lives inside each object. Since every image has a single
consistent value across all its objects, we read it from objects[0].

Pipeline context
----------------
- Inputs  : output/annotations/ folder containing .json and .jpg files
- Outputs : per-condition activation tensors saved as .npz files,
            one file per layer of interest per condition
- Feeds into: Stage 4 (noise calibration) and Stage 5 (geometric comparison)

Usage
-----
python extract_representations.py \\
    --annotations_dir  output/annotations \\
    --output_dir       results/activations \\
    --layers           layer1 layer2 layer3 layer4 avgpool \\
    --batch_size       32 \\
    --seed             42

Output structure produced
-------------------------
results/activations/
    layer1/
        condition_0.npz    # keys: 'activations' (N, D), 'image_ids' (N,)
        condition_1.npz
    layer2/
        ...
    metadata.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_difficult(ann: dict, json_path: Path) -> int:
    """
    Extract the 'difficult' flag from an annotation dict.

    The flag lives inside objects[0]['difficult']. Every image is expected
    to have at least one object and a consistent difficult value across all
    objects. We validate this and warn if values are inconsistent.
    """
    objects = ann.get("objects", [])

    if len(objects) == 0:
        raise ValueError(
            f"{json_path}: 'objects' array is empty. "
            "Cannot determine difficulty condition."
        )

    values = [obj.get("difficult") for obj in objects]

    if any(v is None for v in values):
        raise ValueError(
            f"{json_path}: one or more objects are missing the 'difficult' field."
        )

    unique = set(values)
    if len(unique) > 1:
        # Inconsistent values — warn and take majority vote
        majority = max(unique, key=values.count)
        print(
            f"  [WARNING] {json_path.name}: inconsistent 'difficult' values "
            f"{values} — using majority vote: {majority}"
        )
        return int(majority)

    value = int(values[0])
    if value not in (0, 1):
        raise ValueError(
            f"{json_path}: 'difficult' = {value} is not binary (expected 0 or 1)."
        )
    return value


def load_and_group_annotations(
    annotations_dir: Path,
) -> dict[int, list[dict]]:
    """
    Parse all JSON files in annotations_dir, extract the difficult flag
    from each, and resolve the corresponding image path (same folder,
    same stem, .jpg extension).

    Returns a dict: {0: [item, ...], 1: [item, ...]}
    where each item has keys: image_id, image_path, difficult.

    Skips annotations whose image file cannot be found.
    """
    EXTENSIONS = [".jpg", ".jpeg", ".png"]
    groups: dict[int, list[dict]] = {0: [], 1: []}

    json_files = sorted(annotations_dir.glob("*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON annotation files found in {annotations_dir}"
        )

    missing_images = 0
    for json_path in json_files:
        with open(json_path) as f:
            ann = json.load(f)

        # Resolve image path — same folder, same stem, image extension
        image_path = None
        for ext in EXTENSIONS:
            candidate = annotations_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            missing_images += 1
            if missing_images <= 5:
                print(
                    f"  [WARNING] No image found for {json_path.stem} "
                    f"(tried {EXTENSIONS}) — skipping."
                )
            continue

        difficult = parse_difficult(ann, json_path)

        # Prefer explicit image_id field, fall back to filename stem
        image_id = str(ann.get("image_id", json_path.stem))

        groups[difficult].append({
            "image_id":   image_id,
            "image_path": image_path,
            "difficult":  difficult,
        })

    if missing_images > 5:
        print(f"  [WARNING] ... and {missing_images - 5} more missing images.")

    print(f"  Condition 0 (not difficult): {len(groups[0])} images")
    print(f"  Condition 1 (difficult):     {len(groups[1])} images")

    return groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ConditionDataset(Dataset):
    """
    Loads images for a single difficulty condition.

    Parameters
    ----------
    items     : list of dicts with keys 'image_id' and 'image_path'
    transform : torchvision transform
    """

    def __init__(self, items: list[dict], transform=None):
        self.items     = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img  = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, item["image_id"]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model and hook registration
# ---------------------------------------------------------------------------

# Map human-readable layer names to ResNet50 submodule paths.
# All five layers extracted by default — best-matching layer determined
# in Stage 4, not here.
RESNET50_LAYERS = {
    "layer1":  "layer1",   # early/mid features
    "layer2":  "layer2",   # mid features
    "layer3":  "layer3",   # mid/high features
    "layer4":  "layer4",   # high-level features
    "avgpool": "avgpool",  # global average pool — 2048-d summary vector
}


class ActivationExtractor:
    """
    Registers forward hooks on named ResNet50 submodules and collects
    activations batch-by-batch.

    Spatial dims are flattened: (B, C, H, W) -> (B, C*H*W).
    avgpool output (B, C, 1, 1) flattens to (B, C).

    Call consolidate() after the full forward pass to concatenate batches.
    Call remove_hooks() after consolidation to clean up.
    """

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.layer_names = layer_names
        self.activations: dict[str, list[np.ndarray]] = {n: [] for n in layer_names}
        self._hooks: list = []
        named_modules = dict(model.named_modules())
        for name in layer_names:
            if name not in RESNET50_LAYERS:
                raise ValueError(
                    f"Unknown layer '{name}'. "
                    f"Choose from: {list(RESNET50_LAYERS.keys())}"
                )
            submodule = named_modules[RESNET50_LAYERS[name]]
            hook = submodule.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            acts = output.detach().cpu().flatten(start_dim=1).numpy()
            self.activations[name].append(acts)
        return hook

    def consolidate(self) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate(batches, axis=0)
            for name, batches in self.activations.items()
        }

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def extract_activations(
    annotations_dir: str | Path,
    output_dir:      str | Path,
    layer_names:     list[str],
    batch_size:      int = 32,
    num_workers:     int = 4,
    device:          str = "cpu",
) -> dict:
    """
    Main extraction function.

    For each difficulty condition (0 and 1), runs all images through
    ResNet50 and saves activations per layer as compressed .npz files.

    Returns a metadata dict summarising the run.
    """
    annotations_dir = Path(annotations_dir)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in layer_names:
        (output_dir / name).mkdir(exist_ok=True)

    # Parse and group
    print("Parsing annotations...")
    groups = load_and_group_annotations(annotations_dir)

    # Load model
    print("\nLoading pretrained ResNet50 (ImageNet weights)...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    transform = build_transform()
    metadata  = {
        "model":                  "resnet50",
        "weights":                "IMAGENET1K_V1",
        "annotations_dir":        str(annotations_dir),
        "conditions":             {str(k): len(v) for k, v in groups.items()},
        "layers":                 layer_names,
        "layer_shapes":           {},
        "image_ids_by_condition": {},
    }

    for condition, items in groups.items():
        if len(items) == 0:
            print(f"\n[WARNING] Condition {condition} has no images — skipping.")
            continue

        label = "not difficult" if condition == 0 else "difficult"
        print(f"\nCondition {condition} ({label}) — {len(items)} images")

        dataset = ConditionDataset(items, transform=transform)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,        # keep order deterministic for image_ids alignment
            num_workers=num_workers,
            pin_memory=(device != "cpu"),
        )

        extractor   = ActivationExtractor(model, layer_names)
        all_ids: list[str] = []

        with torch.no_grad():
            for images, ids in tqdm(loader, desc=f"  Extracting"):
                images = images.to(device)
                _      = model(images)
                all_ids.extend(ids)

        extractor.remove_hooks()
        consolidated = extractor.consolidate()

        for name, acts in consolidated.items():
            out_path = output_dir / name / f"condition_{condition}.npz"
            np.savez_compressed(
                out_path,
                activations=acts,
                image_ids=np.array(all_ids),
            )
            print(f"  Saved {name}: shape {acts.shape} → {out_path}")

            if name not in metadata["layer_shapes"]:
                metadata["layer_shapes"][name] = list(acts.shape[1:])

        metadata["image_ids_by_condition"][str(condition)] = all_ids

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: Extract ResNet50 activations grouped by difficulty condition. "
            "Reads annotations and images from the same folder."
        )
    )
    parser.add_argument(
        "--annotations_dir", type=str, required=True,
        help="Folder containing .json annotation files and .jpg image files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/activations",
        help="Root directory for saving activation files."
    )
    parser.add_argument(
        "--layers", type=str, nargs="+",
        default=["layer1", "layer2", "layer3", "layer4", "avgpool"],
        help=f"ResNet50 layers to extract. Options: {list(RESNET50_LAYERS.keys())}"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda / cpu)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("Stage 3 — ConvNet Representation Extraction")
    print("=" * 60)
    print(f"  Annotations dir : {args.annotations_dir}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Layers          : {args.layers}")
    print(f"  Device          : {args.device}")
    print(f"  Batch size      : {args.batch_size}")
    print("=" * 60)

    metadata = extract_activations(
        annotations_dir = args.annotations_dir,
        output_dir      = args.output_dir,
        layer_names     = args.layers,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        device          = args.device,
    )

    print("\nDone.")
    for cond, count in metadata["conditions"].items():
        label = "not difficult" if int(cond) == 0 else "difficult"
        print(f"  Condition {cond} ({label}): {count} images extracted.")
