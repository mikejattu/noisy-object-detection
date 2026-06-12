"""
Stage 4 — Noise Calibration
============================
Addresses the noise equivalence problem by establishing principled
correspondences between the two conditions (difficult: 0 and 1) across
three measurement systems:

    1. LPIPS perceptual distance  — how different do the images look to a
                                    perceptual similarity model?
    2. ConvNet activation distance — how different are ResNet50 activations
                                    across conditions, per layer?
    3. Human RDM correlation      — how well does each layer's RDM correlate
                                    with human similarity judgements?
                                    [PLACEHOLDER — slot in RDMs when ready]

Outputs
-------
For each layer, reports:
    - Mean LPIPS distance per condition
    - Mean pairwise Euclidean distance between condition activation centroids
    - Spearman correlation with human RDM (placeholder: NaN until RDMs available)
    - A calibration summary recommending the best-matching layer

All metrics saved to results/calibration/calibration_report.json and a
human-readable results/calibration/calibration_report.csv.

Pipeline context
----------------
- Inputs  : results/activations/   (Stage 3 output)
            data/images/           (original images for LPIPS)
            data/human_rdm.npz     (PLACEHOLDER — not required yet)
- Outputs : results/calibration/calibration_report.json
                                   calibration_report.csv
                                   lpips_distances.npz
- Feeds into: Stage 5 (geometric comparison)

Usage
-----
python stage4/noise_calibration.py \\
    --activations_dir  results/activations \\
    --annotations_dir  output/annotations \\
    --output_dir       results/calibration \\
    --human_rdm        data/human_rdm.npz \\   # optional — skip until ready
    --layers           layer1 layer2 layer3 layer4 avgpool \\
    --device           cuda
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from torchvision import transforms
from tqdm import tqdm

# LPIPS requires the lpips package: pip install lpips
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn(
        "lpips not installed. LPIPS distances will be skipped.\n"
        "Install with: pip install lpips",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_activations(activations_dir: Path, layer: str) -> dict[int, dict]:
    """
    Load condition_0.npz and condition_1.npz for a given layer.

    Returns:
        {
            0: {'activations': np.ndarray (N, D), 'image_ids': np.ndarray (N,)},
            1: {...}
        }
    """
    result = {}
    for condition in (0, 1):
        path = activations_dir / layer / f"condition_{condition}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Activation file not found: {path}\n"
                "Run Stage 3 first."
            )
        data = np.load(path, allow_pickle=True)
        result[condition] = {
            "activations": data["activations"],
            "image_ids":   data["image_ids"],
        }
    return result


def load_image(image_id: str, annotations_dir: Path) -> Image.Image | None:
    """Resolve image path from annotations directory (same folder as JSONs)."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = annotations_dir / f"{image_id}{ext}"
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    return None


# ---------------------------------------------------------------------------
# LPIPS distances
# ---------------------------------------------------------------------------

LPIPS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # LPIPS expects [-1, 1]
])


def compute_lpips_distances(
    condition_data:   dict[int, dict],
    annotations_dir:  Path,
    device:           str = "cpu",
    max_pairs:        int = 500,
) -> dict:
    """
    Compute LPIPS perceptual distances between images in condition 0 and
    condition 1.

    Rather than all pairwise comparisons (which is expensive), we sample
    up to max_pairs cross-condition pairs and report summary statistics.

    Returns a dict with keys:
        mean_cross_condition  — mean LPIPS distance between conditions
        std_cross_condition
        mean_within_0         — mean LPIPS within condition 0 (baseline)
        mean_within_1         — mean LPIPS within condition 1 (baseline)
        n_pairs_computed
    """
    if not LPIPS_AVAILABLE:
        return {"error": "lpips not installed"}

    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    def load_tensor(image_id: str) -> torch.Tensor | None:
        img = load_image(str(image_id), annotations_dir)
        if img is None:
            return None
        return LPIPS_TRANSFORM(img).unsqueeze(0).to(device)

    ids_0 = list(condition_data[0]["image_ids"])
    ids_1 = list(condition_data[1]["image_ids"])

    # Sample cross-condition pairs
    rng       = np.random.default_rng(42)
    n_pairs   = min(max_pairs, len(ids_0) * len(ids_1))
    idx_0     = rng.choice(len(ids_0), size=n_pairs, replace=True)
    idx_1     = rng.choice(len(ids_1), size=n_pairs, replace=True)

    cross_distances = []
    with torch.no_grad():
        for i, j in tqdm(
            zip(idx_0, idx_1),
            total=n_pairs,
            desc="  LPIPS cross-condition",
        ):
            t0 = load_tensor(ids_0[i])
            t1 = load_tensor(ids_1[j])
            if t0 is None or t1 is None:
                continue
            d = loss_fn(t0, t1).item()
            cross_distances.append(d)

    # Sample within-condition baselines (condition 0)
    within_0 = []
    n_within  = min(200, len(ids_0) * (len(ids_0) - 1) // 2)
    wi_idx    = rng.choice(len(ids_0), size=(n_within, 2), replace=True)
    with torch.no_grad():
        for i, j in tqdm(wi_idx, desc="  LPIPS within condition 0"):
            if i == j:
                continue
            t0a = load_tensor(ids_0[i])
            t0b = load_tensor(ids_0[j])
            if t0a is None or t0b is None:
                continue
            within_0.append(loss_fn(t0a, t0b).item())

    # Sample within-condition baselines (condition 1)
    within_1 = []
    n_within  = min(200, len(ids_1) * (len(ids_1) - 1) // 2)
    wi_idx    = rng.choice(len(ids_1), size=(n_within, 2), replace=True)
    with torch.no_grad():
        for i, j in tqdm(wi_idx, desc="  LPIPS within condition 1"):
            if i == j:
                continue
            t1a = load_tensor(ids_1[i])
            t1b = load_tensor(ids_1[j])
            if t1a is None or t1b is None:
                continue
            within_1.append(loss_fn(t1a, t1b).item())

    return {
        "mean_cross_condition": float(np.mean(cross_distances)) if cross_distances else None,
        "std_cross_condition":  float(np.std(cross_distances))  if cross_distances else None,
        "mean_within_0":        float(np.mean(within_0))        if within_0 else None,
        "mean_within_1":        float(np.mean(within_1))        if within_1 else None,
        "n_pairs_computed":     len(cross_distances),
    }


# ---------------------------------------------------------------------------
# Activation distances
# ---------------------------------------------------------------------------

def compute_activation_distances(condition_data: dict[int, dict]) -> dict:
    """
    Compute Euclidean distances between condition activation centroids,
    and mean pairwise distances within each condition.

    Uses centroids rather than all pairwise distances for efficiency —
    centroid distance captures the overall shift in representational space
    between conditions.

    Returns a dict with keys:
        centroid_distance       — L2 distance between condition centroids
        mean_pairwise_within_0  — mean pairwise distance within condition 0
        mean_pairwise_within_1  — mean pairwise distance within condition 1
        centroid_0              — centroid of condition 0 activations
        centroid_1              — centroid of condition 1 activations
    """
    acts_0 = condition_data[0]["activations"].astype(np.float32)
    acts_1 = condition_data[1]["activations"].astype(np.float32)

    centroid_0 = acts_0.mean(axis=0)
    centroid_1 = acts_1.mean(axis=0)

    centroid_distance = float(np.linalg.norm(centroid_0 - centroid_1))

    # Sample pairwise within-condition distances (cap at 500 pairs for speed)
    rng    = np.random.default_rng(42)
    n_samp = min(500, len(acts_0))
    sample_0 = acts_0[rng.choice(len(acts_0), size=n_samp, replace=False)]
    pw_0     = cdist(sample_0, sample_0, metric="euclidean")
    mean_pw_0 = float(pw_0[np.triu_indices(n_samp, k=1)].mean())

    n_samp   = min(500, len(acts_1))
    sample_1 = acts_1[rng.choice(len(acts_1), size=n_samp, replace=False)]
    pw_1     = cdist(sample_1, sample_1, metric="euclidean")
    mean_pw_1 = float(pw_1[np.triu_indices(n_samp, k=1)].mean())

    return {
        "centroid_distance":      centroid_distance,
        "mean_pairwise_within_0": mean_pw_0,
        "mean_pairwise_within_1": mean_pw_1,
    }


# ---------------------------------------------------------------------------
# Human RDM correlation  [PLACEHOLDER]
# ---------------------------------------------------------------------------

def compute_rdm_correlation(
    condition_data: dict[int, dict],
    human_rdm_path: Path | None,
    layer:          str,
) -> dict:
    """
    Compute Spearman correlation between the ConvNet RDM (for this layer)
    and the human RDM from psychophysics.

    *** PLACEHOLDER ***
    This function is a stub. When psychophysics RDMs are available:

        1. Load your human RDM from human_rdm_path (.npz expected).
           Expected format:
               np.load(human_rdm_path)['rdm']  — shape (N, N) symmetric matrix
               np.load(human_rdm_path)['image_ids']  — shape (N,) matching row/col order

        2. Align image ordering between human RDM and ConvNet activations
           using the image_ids arrays from both.

        3. Compute the ConvNet RDM from activations using pairwise cosine
           or Euclidean distance.

        4. Vectorise both upper triangles and compute spearmanr.

    Returns a dict with keys:
        spearman_r      — Spearman rho (NaN until RDMs available)
        spearman_p      — p-value (NaN until RDMs available)
        n_shared_images — number of images present in both RDMs
        status          — 'placeholder' or 'computed'
    """
    if human_rdm_path is None or not Path(human_rdm_path).exists():
        return {
            "spearman_r":      float("nan"),
            "spearman_p":      float("nan"),
            "n_shared_images": 0,
            "status":          "placeholder — human RDM not yet available",
        }

    # ------------------------------------------------------------------ #
    # TODO: replace this block when RDMs are ready                        #
    # ------------------------------------------------------------------ #
    try:
        human_data    = np.load(human_rdm_path, allow_pickle=True)
        human_rdm     = human_data["rdm"]          # (N, N)
        human_ids     = list(human_data["image_ids"])

        # Gather ConvNet activations for images present in the human RDM
        all_acts  = np.concatenate(
            [condition_data[0]["activations"], condition_data[1]["activations"]],
            axis=0,
        )
        all_ids   = list(condition_data[0]["image_ids"]) + \
                    list(condition_data[1]["image_ids"])
        id_to_idx = {str(iid): i for i, iid in enumerate(all_ids)}

        shared_ids = [iid for iid in human_ids if str(iid) in id_to_idx]
        if len(shared_ids) < 2:
            return {
                "spearman_r":      float("nan"),
                "spearman_p":      float("nan"),
                "n_shared_images": len(shared_ids),
                "status":          "insufficient shared images between RDMs",
            }

        conv_idx  = [id_to_idx[str(iid)] for iid in shared_ids]
        hum_idx   = [human_ids.index(iid) for iid in shared_ids]

        conv_acts = all_acts[conv_idx]
        hum_rdm_s = human_rdm[np.ix_(hum_idx, hum_idx)]

        # ConvNet RDM via cosine distance
        conv_rdm  = cdist(conv_acts, conv_acts, metric="cosine")

        # Upper triangle (excluding diagonal)
        n         = len(shared_ids)
        triu_idx  = np.triu_indices(n, k=1)
        rho, pval = spearmanr(conv_rdm[triu_idx], hum_rdm_s[triu_idx])

        return {
            "spearman_r":      float(rho),
            "spearman_p":      float(pval),
            "n_shared_images": len(shared_ids),
            "status":          "computed",
        }

    except Exception as e:
        return {
            "spearman_r":      float("nan"),
            "spearman_p":      float("nan"),
            "n_shared_images": 0,
            "status":          f"error: {e}",
        }
    # ------------------------------------------------------------------ #


# ---------------------------------------------------------------------------
# Layer selection summary
# ---------------------------------------------------------------------------

def select_best_layer(report: dict) -> dict:
    """
    Given the full calibration report, rank layers by each metric and
    report the best candidate per metric.

    Criteria:
        - Highest Spearman r with human RDM (primary — when available)
        - Largest centroid distance (most condition-sensitive layer)
        - Smallest ratio of centroid_distance / mean_pairwise_within
          (most discriminative relative to within-condition spread)

    Returns a dict summarising the recommendation.
    """
    layers  = list(report.keys())
    summary = {}

    # Spearman r ranking (skip if all NaN)
    rhos = {l: report[l]["rdm_correlation"]["spearman_r"] for l in layers}
    if not all(np.isnan(v) for v in rhos.values()):
        best_spearman = max(rhos, key=lambda l: rhos[l] if not np.isnan(rhos[l]) else -np.inf)
        summary["best_by_spearman_r"] = {
            "layer": best_spearman,
            "value": rhos[best_spearman],
        }
    else:
        summary["best_by_spearman_r"] = {
            "layer": "unavailable — human RDM not yet provided",
            "value": None,
        }

    # Centroid distance ranking
    centroid_dists = {
        l: report[l]["activation_distances"]["centroid_distance"] for l in layers
    }
    best_centroid = max(centroid_dists, key=lambda l: centroid_dists[l])
    summary["best_by_centroid_distance"] = {
        "layer": best_centroid,
        "value": centroid_dists[best_centroid],
    }

    # Discriminability ratio: centroid_distance / mean_within
    ratios = {}
    for l in layers:
        cd  = report[l]["activation_distances"]["centroid_distance"]
        pw0 = report[l]["activation_distances"]["mean_pairwise_within_0"]
        pw1 = report[l]["activation_distances"]["mean_pairwise_within_1"]
        mean_within = (pw0 + pw1) / 2 if (pw0 and pw1) else None
        ratios[l]   = cd / mean_within if mean_within else None

    valid_ratios = {l: v for l, v in ratios.items() if v is not None}
    if valid_ratios:
        best_ratio = max(valid_ratios, key=lambda l: valid_ratios[l])
        summary["best_by_discriminability_ratio"] = {
            "layer": best_ratio,
            "value": valid_ratios[best_ratio],
            "all_ratios": ratios,
        }

    summary["note"] = (
        "Layer selection is a decision for the researcher. "
        "When human RDMs are available, best_by_spearman_r is the primary criterion. "
        "Until then, best_by_discriminability_ratio provides a principled interim choice."
    )

    return summary


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def run_calibration(
    activations_dir: str | Path,
    annotations_dir: str | Path,
    output_dir:      str | Path,
    layer_names:     list[str],
    human_rdm_path:  str | Path | None = None,
    device:          str = "cpu",
    max_lpips_pairs: int = 500,
) -> dict:
    """
    Run the full noise calibration pipeline across all layers.

    Returns the full calibration report as a dict.
    """
    activations_dir = Path(activations_dir)
    annotations_dir = Path(annotations_dir)
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    for layer in layer_names:
        print(f"\n{'='*50}")
        print(f"Layer: {layer}")
        print(f"{'='*50}")

        # Load activations for this layer
        condition_data = load_activations(activations_dir, layer)
        print(
            f"  Condition 0: {condition_data[0]['activations'].shape} | "
            f"Condition 1: {condition_data[1]['activations'].shape}"
        )

        # 1. LPIPS distances
        print("\n  1. LPIPS perceptual distances")
        lpips_results = compute_lpips_distances(
            condition_data,
            annotations_dir,
            device=device,
            max_pairs=max_lpips_pairs,
        )
        print(f"     Cross-condition mean LPIPS : "
              f"{lpips_results.get('mean_cross_condition', 'N/A'):.4f}"
              if lpips_results.get("mean_cross_condition") else
              f"     LPIPS: {lpips_results}")

        # 2. Activation distances
        print("\n  2. Activation distances")
        act_results = compute_activation_distances(condition_data)
        print(f"     Centroid distance          : {act_results['centroid_distance']:.4f}")
        print(f"     Mean pairwise within cond 0: {act_results['mean_pairwise_within_0']:.4f}")
        print(f"     Mean pairwise within cond 1: {act_results['mean_pairwise_within_1']:.4f}")

        # 3. Human RDM correlation [placeholder]
        print("\n  3. Human RDM correlation")
        rdm_results = compute_rdm_correlation(condition_data, human_rdm_path, layer)
        print(f"     Status    : {rdm_results['status']}")
        if not np.isnan(rdm_results["spearman_r"]):
            print(f"     Spearman r: {rdm_results['spearman_r']:.4f} "
                  f"(p={rdm_results['spearman_p']:.4f})")

        report[layer] = {
            "lpips_distances":     lpips_results,
            "activation_distances": act_results,
            "rdm_correlation":     rdm_results,
        }

    # Layer selection summary
    print(f"\n{'='*50}")
    print("Layer selection summary")
    print(f"{'='*50}")
    selection = select_best_layer(report)
    report["_layer_selection"] = selection

    for metric, result in selection.items():
        if metric == "note":
            print(f"\nNote: {result}")
        elif isinstance(result, dict) and "layer" in result:
            val = f"{result['value']:.4f}" if result["value"] is not None else "N/A"
            print(f"  {metric}: {result['layer']} ({val})")

    # Save JSON report
    json_path = output_dir / "calibration_report.json"
    with open(json_path, "w") as f:
        # Convert any remaining NaN to null for valid JSON
        json.dump(report, f, indent=2, default=lambda x: None if (
            isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nFull report saved to {json_path}")

    # Save CSV summary (one row per layer)
    csv_path = output_dir / "calibration_report.csv"
    csv_rows = []
    for layer in layer_names:
        r = report[layer]
        csv_rows.append({
            "layer":                    layer,
            "lpips_cross_condition":    r["lpips_distances"].get("mean_cross_condition"),
            "lpips_within_0":           r["lpips_distances"].get("mean_within_0"),
            "lpips_within_1":           r["lpips_distances"].get("mean_within_1"),
            "centroid_distance":        r["activation_distances"]["centroid_distance"],
            "mean_pairwise_within_0":   r["activation_distances"]["mean_pairwise_within_0"],
            "mean_pairwise_within_1":   r["activation_distances"]["mean_pairwise_within_1"],
            "spearman_r":               r["rdm_correlation"]["spearman_r"],
            "spearman_p":               r["rdm_correlation"]["spearman_p"],
            "rdm_status":               r["rdm_correlation"]["status"],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV summary saved to {csv_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4: Noise calibration — compute LPIPS distances, "
            "activation distances, and human RDM correlations across layers."
        )
    )
    parser.add_argument(
        "--activations_dir", type=str, required=True,
        help="Stage 3 output directory (results/activations)."
    )
    parser.add_argument(
        "--annotations_dir", type=str, required=True,
        help="Folder containing annotation JSONs and image files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/calibration",
        help="Directory for calibration outputs."
    )
    parser.add_argument(
        "--human_rdm", type=str, default=None,
        help=(
            "Path to human RDM .npz file (optional — placeholder used if omitted). "
            "Expected keys: 'rdm' (N, N) and 'image_ids' (N,)."
        )
    )
    parser.add_argument(
        "--layers", type=str, nargs="+",
        default=["layer1", "layer2", "layer3", "layer4", "avgpool"],
        help="Layers to calibrate (must match Stage 3 output)."
    )
    parser.add_argument(
        "--max_lpips_pairs", type=int, default=500,
        help="Maximum cross-condition image pairs for LPIPS computation."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LPIPS model (cuda / cpu)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Stage 4 — Noise Calibration")
    print("=" * 60)
    print(f"  Activations dir : {args.activations_dir}")
    print(f"  Annotations dir : {args.annotations_dir}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Human RDM       : {args.human_rdm or 'not provided (placeholder)'}")
    print(f"  Layers          : {args.layers}")
    print(f"  Device          : {args.device}")
    print("=" * 60)

    run_calibration(
        activations_dir  = args.activations_dir,
        annotations_dir  = args.annotations_dir,
        output_dir       = args.output_dir,
        layer_names      = args.layers,
        human_rdm_path   = args.human_rdm,
        device           = args.device,
        max_lpips_pairs  = args.max_lpips_pairs,
    )

    print("\nDone.")
