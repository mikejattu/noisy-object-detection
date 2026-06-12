"""
Stage 5 — Geometric Comparison
================================
Compares the geometry of perceptual representations across noise conditions
(difficult: 0 vs 1) using UMAP manifold learning.

Two geometric properties are measured per layer per condition:

    1. Intrinsic dimensionality
       — Participation Ratio (PR) from PCA: effective number of dimensions
         needed to explain the variance in the activation space.
       — Measures whether noise compresses or expands the representational
         manifold.

    2. RDM geometry
       — Pairwise cosine dissimilarity matrix (RDM) per condition.
       — Spearman correlation between condition RDMs: how similar is the
         geometric structure across conditions?
       — Mean dissimilarity and variance per condition: how spread out and
         variable are representations?

UMAP embeddings are computed per layer per condition for visualisation.

Outputs
-------
results/geometry/
    metrics.json                  — all geometric metrics per layer
    metrics.csv                   — summary table
    plots/
        {layer}_embedding.png     — UMAP embeddings, condition 0 vs 1
        {layer}_rdm.png           — RDM heatmaps, condition 0 vs 1
        {layer}_dimensionality.png — participation ratio bar chart
        summary.png               — all layers side by side

Pipeline context
----------------
- Inputs  : results/activations/  (Stage 3 output)
            results/calibration/  (Stage 4 output — used to select best layer)
- Outputs : results/geometry/

Usage
-----
python stage5/geometric_comparison.py \\
    --activations_dir  results/activations \\
    --calibration_dir  results/calibration \\
    --output_dir       results/geometry \\
    --layers           layer1 layer2 layer3 layer4 avgpool \\
    --n_neighbors      15 \\
    --min_dist         0.1 \\
    --seed             42
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn(
        "umap-learn not installed. UMAP embeddings will be skipped.\n"
        "Install with: pip install umap-learn",
        stacklevel=2,
    )

# Colour scheme: condition 0 = teal, condition 1 = coral
CONDITION_COLOURS = {0: "#1D9E75", 1: "#D85A30"}
CONDITION_LABELS  = {0: "Condition 0 (not difficult)", 1: "Condition 1 (difficult)"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(activations_dir: Path, layer: str) -> dict[int, dict]:
    """Load condition_0.npz and condition_1.npz for a given layer."""
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
            "activations": data["activations"].astype(np.float32),
            "image_ids":   data["image_ids"],
        }
    return result


def load_best_layer(calibration_dir: Path) -> str | None:
    """
    Read Stage 4 calibration report and return the recommended best layer
    by discriminability ratio (used when human RDMs are not yet available).
    Returns None if calibration report is missing.
    """
    report_path = calibration_dir / "calibration_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        report = json.load(f)
    selection = report.get("_layer_selection", {})
    best = selection.get("best_by_spearman_r", {})
    if best.get("value") is not None:
        return best["layer"]
    best = selection.get("best_by_discriminability_ratio", {})
    return best.get("layer")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(activations: np.ndarray, max_samples: int = 1000) -> np.ndarray:
    """
    Standardise activations and subsample if needed.

    Subsampling is capped at max_samples per condition to keep UMAP and
    pairwise distance computation tractable without losing geometric structure.
    """
    rng = np.random.default_rng(42)
    if len(activations) > max_samples:
        idx = rng.choice(len(activations), size=max_samples, replace=False)
        activations = activations[idx]
    scaler = StandardScaler()
    return scaler.fit_transform(activations)


# ---------------------------------------------------------------------------
# 1. Intrinsic dimensionality — Participation Ratio
# ---------------------------------------------------------------------------

def participation_ratio(activations: np.ndarray, n_components: int = 50) -> dict:
    """
    Compute the Participation Ratio (PR) as a measure of intrinsic
    dimensionality.

    PR = (sum of eigenvalues)^2 / sum of (eigenvalues^2)

    PR is the effective number of PCA dimensions that carry the variance.
    A higher PR means more dimensions are needed — the manifold is higher
    dimensional. Noise-induced compression shows up as a drop in PR.

    Also records the number of components needed to explain 80% and 95%
    of variance.

    Parameters
    ----------
    activations  : standardised activation matrix (N, D)
    n_components : number of PCA components to fit (capped at min(N, D))
    """
    n = min(n_components, activations.shape[0], activations.shape[1])
    pca = PCA(n_components=n)
    pca.fit(activations)

    eigenvalues = pca.explained_variance_
    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dims_80 = int(np.searchsorted(cumvar, 0.80) + 1)
    dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    return {
        "participation_ratio":  pr,
        "dims_for_80pct_var":   dims_80,
        "dims_for_95pct_var":   dims_95,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components_fitted":  n,
    }


# ---------------------------------------------------------------------------
# 2. RDM geometry
# ---------------------------------------------------------------------------

def compute_rdm(activations: np.ndarray, max_size: int = 300) -> np.ndarray:
    """
    Compute pairwise cosine dissimilarity matrix (RDM).

    Subsampled to max_size images if needed — consistent subsampling seed
    ensures the same images are selected across conditions when possible.
    """
    rng = np.random.default_rng(0)
    if len(activations) > max_size:
        idx = rng.choice(len(activations), size=max_size, replace=False)
        activations = activations[idx]
    return cdist(activations, activations, metric="cosine")


def rdm_geometry(condition_data: dict[int, dict]) -> dict:
    """
    Compute RDM metrics for both conditions and compare them.

    Metrics:
        mean_dissimilarity_{0,1}   — mean off-diagonal RDM value per condition
        std_dissimilarity_{0,1}    — std of off-diagonal RDM values
        rdm_spearman_r             — Spearman correlation between the two RDMs
                                     (requires same number of images; uses
                                     the smaller N for alignment)
        rdm_spearman_p
    """
    rdm_0 = compute_rdm(condition_data[0]["activations"])
    rdm_1 = compute_rdm(condition_data[1]["activations"])

    n0    = rdm_0.shape[0]
    n1    = rdm_1.shape[0]
    triu0 = rdm_0[np.triu_indices(n0, k=1)]
    triu1 = rdm_1[np.triu_indices(n1, k=1)]

    # Cross-condition RDM correlation — align to smaller N
    n_shared = min(n0, n1)
    rng      = np.random.default_rng(0)
    idx0     = rng.choice(n0, size=n_shared, replace=False)
    idx1     = rng.choice(n1, size=n_shared, replace=False)
    rdm_0s   = rdm_0[np.ix_(idx0, idx0)]
    rdm_1s   = rdm_1[np.ix_(idx1, idx1)]
    triu_idx = np.triu_indices(n_shared, k=1)
    rho, pval = spearmanr(rdm_0s[triu_idx], rdm_1s[triu_idx])

    return {
        "mean_dissimilarity_0": float(triu0.mean()),
        "std_dissimilarity_0":  float(triu0.std()),
        "mean_dissimilarity_1": float(triu1.mean()),
        "std_dissimilarity_1":  float(triu1.std()),
        "rdm_spearman_r":       float(rho),
        "rdm_spearman_p":       float(pval),
        "n_images_condition_0": n0,
        "n_images_condition_1": n1,
    }, rdm_0, rdm_1


# ---------------------------------------------------------------------------
# 3. UMAP embedding
# ---------------------------------------------------------------------------

def compute_umap(
    condition_data: dict[int, dict],
    n_neighbors:    int = 15,
    min_dist:       float = 0.1,
    seed:           int = 42,
) -> dict[int, np.ndarray] | None:
    """
    Fit UMAP jointly on both conditions to a shared 2D embedding space.

    Fitting jointly (rather than separately per condition) ensures that
    the two conditions are embedded in the same coordinate system, making
    geometric comparison meaningful.

    Returns a dict {condition: embedding (N, 2)} or None if umap unavailable.
    """
    if not UMAP_AVAILABLE:
        return None

    acts_0 = condition_data[0]["activations"]
    acts_1 = condition_data[1]["activations"]

    # Subsample for tractability
    rng    = np.random.default_rng(seed)
    n_samp = min(800, len(acts_0), len(acts_1))
    idx_0  = rng.choice(len(acts_0), size=n_samp, replace=False)
    idx_1  = rng.choice(len(acts_1), size=n_samp, replace=False)

    combined   = np.vstack([acts_0[idx_0], acts_1[idx_1]])
    labels     = np.array([0] * n_samp + [1] * n_samp)

    reducer    = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
        verbose=False,
    )
    embedding  = reducer.fit_transform(combined)

    return {
        0: embedding[:n_samp],
        1: embedding[n_samp:],
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap(
    embeddings: dict | None,
    layer:      str,
    output_path: Path,
) -> None:
    """Side-by-side UMAP scatter: condition 0 and condition 1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    fig.suptitle(f"UMAP embedding — {layer}", fontsize=13, fontweight="bold")

    if embeddings is None:
        for ax in axes:
            ax.text(0.5, 0.5, "UMAP unavailable\n(install umap-learn)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    for ax, condition in zip(axes, (0, 1)):
        emb = embeddings[condition]
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=CONDITION_COLOURS[condition],
            alpha=0.5, s=8, linewidths=0,
        )
        ax.set_title(CONDITION_LABELS[condition], fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rdm(
    rdm_0:       np.ndarray,
    rdm_1:       np.ndarray,
    layer:       str,
    output_path: Path,
) -> None:
    """Side-by-side RDM heatmaps for both conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Representational dissimilarity matrices — {layer}",
                 fontsize=13, fontweight="bold")

    vmax = max(rdm_0.max(), rdm_1.max())
    for ax, rdm, condition in zip(axes, (rdm_0, rdm_1), (0, 1)):
        im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(CONDITION_LABELS[condition], fontsize=10)
        ax.set_xlabel("Image index", fontsize=9)
        ax.set_ylabel("Image index", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine distance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dimensionality(
    pr_results:  dict[str, dict],
    output_path: Path,
) -> None:
    """
    Bar chart of participation ratio per layer per condition.
    Each layer has two bars (condition 0 and 1) side by side.
    """
    layers     = list(pr_results.keys())
    pr_0       = [pr_results[l][0]["participation_ratio"] for l in layers]
    pr_1       = [pr_results[l][1]["participation_ratio"] for l in layers]
    x          = np.arange(len(layers))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, pr_0, width, label=CONDITION_LABELS[0],
           color=CONDITION_COLOURS[0], alpha=0.85)
    ax.bar(x + width/2, pr_1, width, label=CONDITION_LABELS[1],
           color=CONDITION_COLOURS[1], alpha=0.85)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Participation Ratio", fontsize=10)
    ax.set_title("Intrinsic dimensionality per layer", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=9)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(
    metrics:     dict[str, dict],
    pr_results:  dict[str, dict],
    embeddings:  dict[str, dict | None],
    output_path: Path,
) -> None:
    """
    Summary figure: one column per layer, three rows —
    UMAP embedding, participation ratio, RDM Spearman r.
    """
    layers = list(metrics.keys())
    n_cols = len(layers)
    fig    = plt.figure(figsize=(4 * n_cols, 10))
    gs     = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.5, wspace=0.35)

    fig.suptitle("Geometric comparison summary — all layers",
                 fontsize=14, fontweight="bold", y=1.01)

    for col, layer in enumerate(layers):
        # Row 0: UMAP scatter (both conditions overlaid)
        ax0 = fig.add_subplot(gs[0, col])
        emb = embeddings.get(layer)
        if emb is not None:
            for condition in (0, 1):
                e = emb[condition]
                ax0.scatter(e[:, 0], e[:, 1],
                            c=CONDITION_COLOURS[condition],
                            alpha=0.4, s=4, linewidths=0,
                            label=f"C{condition}")
            ax0.legend(fontsize=7, markerscale=2)
        else:
            ax0.text(0.5, 0.5, "UMAP\nunavailable",
                     ha="center", va="center", transform=ax0.transAxes, fontsize=8)
        ax0.set_title(layer, fontsize=10, fontweight="bold")
        ax0.set_xlabel("UMAP 1", fontsize=7)
        ax0.set_ylabel("UMAP 2", fontsize=7)
        ax0.tick_params(labelsize=6)

        # Row 1: Participation ratio bar
        ax1 = fig.add_subplot(gs[1, col])
        pr0 = pr_results[layer][0]["participation_ratio"]
        pr1 = pr_results[layer][1]["participation_ratio"]
        ax1.bar([0, 1], [pr0, pr1],
                color=[CONDITION_COLOURS[0], CONDITION_COLOURS[1]],
                alpha=0.85)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["C0", "C1"], fontsize=8)
        ax1.set_ylabel("PR", fontsize=8)
        ax1.set_title("Dimensionality", fontsize=9)
        ax1.tick_params(labelsize=7)
        ax1.grid(axis="y", alpha=0.3)

        # Row 2: RDM Spearman r (single value — annotated bar)
        ax2 = fig.add_subplot(gs[2, col])
        rho = metrics[layer]["rdm_geometry"]["rdm_spearman_r"]
        colour = "#1D9E75" if rho > 0.5 else "#D85A30" if rho < 0.3 else "#BA7517"
        ax2.bar([0], [rho], color=colour, alpha=0.85)
        ax2.set_ylim(0, 1)
        ax2.set_xticks([])
        ax2.set_ylabel("Spearman r", fontsize=8)
        ax2.set_title("RDM similarity\n(C0 vs C1)", fontsize=9)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.text(0, rho + 0.02, f"{rho:.2f}", ha="center", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(axis="y", alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary figure saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_geometric_comparison(
    activations_dir: str | Path,
    calibration_dir: str | Path,
    output_dir:      str | Path,
    layer_names:     list[str],
    n_neighbors:     int = 15,
    min_dist:        float = 0.1,
    seed:            int = 42,
) -> dict:
    """
    Run the full geometric comparison pipeline across all layers.
    Returns the full metrics dict.
    """
    activations_dir = Path(activations_dir)
    calibration_dir = Path(calibration_dir)
    output_dir      = Path(output_dir)
    plots_dir       = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load best layer recommendation from Stage 4
    best_layer = load_best_layer(calibration_dir)
    if best_layer:
        print(f"  Stage 4 recommended layer: {best_layer}")
    else:
        print("  No Stage 4 calibration report found — running all layers.")

    all_metrics  = {}
    all_pr       = {}
    all_embeddings = {}

    for layer in layer_names:
        print(f"\n{'='*50}")
        print(f"Layer: {layer}" + (" ← Stage 4 best" if layer == best_layer else ""))
        print(f"{'='*50}")

        condition_data = load_activations(activations_dir, layer)

        # Preprocess
        for cond in (0, 1):
            condition_data[cond]["activations"] = preprocess(
                condition_data[cond]["activations"]
            )

        # 1. Intrinsic dimensionality
        print("\n  1. Intrinsic dimensionality (Participation Ratio)")
        pr = {}
        for cond in (0, 1):
            pr[cond] = participation_ratio(condition_data[cond]["activations"])
            print(
                f"     Condition {cond}: PR = {pr[cond]['participation_ratio']:.2f} | "
                f"dims@80% = {pr[cond]['dims_for_80pct_var']} | "
                f"dims@95% = {pr[cond]['dims_for_95pct_var']}"
            )
        all_pr[layer] = pr

        # 2. RDM geometry
        print("\n  2. RDM geometry")
        rdm_metrics, rdm_0, rdm_1 = rdm_geometry(condition_data)
        print(f"     Mean dissimilarity C0 : {rdm_metrics['mean_dissimilarity_0']:.4f}")
        print(f"     Mean dissimilarity C1 : {rdm_metrics['mean_dissimilarity_1']:.4f}")
        print(f"     RDM Spearman r (C0 vs C1): {rdm_metrics['rdm_spearman_r']:.4f} "
              f"(p={rdm_metrics['rdm_spearman_p']:.4f})")

        # 3. UMAP
        print("\n  3. UMAP embedding")
        embeddings = compute_umap(
            condition_data,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
        )
        all_embeddings[layer] = embeddings
        if embeddings:
            print("     UMAP computed.")
        else:
            print("     UMAP skipped (umap-learn not installed).")

        # Store metrics
        all_metrics[layer] = {
            "dimensionality": {
                "condition_0": pr[0],
                "condition_1": pr[1],
                "pr_difference": pr[0]["participation_ratio"] - pr[1]["participation_ratio"],
            },
            "rdm_geometry": rdm_metrics,
            "is_best_layer": layer == best_layer,
        }

        # Per-layer plots
        print("\n  Saving per-layer plots...")
        plot_umap(
            embeddings, layer,
            plots_dir / f"{layer}_embedding.png",
        )
        plot_rdm(
            rdm_0, rdm_1, layer,
            plots_dir / f"{layer}_rdm.png",
        )
        print(f"     {layer}_embedding.png, {layer}_rdm.png saved.")

    # Dimensionality summary plot (all layers)
    plot_dimensionality(all_pr, plots_dir / "dimensionality_all_layers.png")
    print(f"\n  Dimensionality plot saved.")

    # Summary figure
    plot_summary(
        all_metrics, all_pr, all_embeddings,
        plots_dir / "summary.png",
    )

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2,
                  default=lambda x: None if (
                      isinstance(x, float) and np.isnan(x)) else x)
    print(f"\nMetrics saved to {metrics_path}")

    # Save metrics CSV
    csv_path = output_dir / "metrics.csv"
    csv_rows = []
    for layer in layer_names:
        m = all_metrics[layer]
        csv_rows.append({
            "layer":                  layer,
            "is_best_layer":          m["is_best_layer"],
            "pr_condition_0":         m["dimensionality"]["condition_0"]["participation_ratio"],
            "pr_condition_1":         m["dimensionality"]["condition_1"]["participation_ratio"],
            "pr_difference":          m["dimensionality"]["pr_difference"],
            "dims_80pct_condition_0": m["dimensionality"]["condition_0"]["dims_for_80pct_var"],
            "dims_80pct_condition_1": m["dimensionality"]["condition_1"]["dims_for_80pct_var"],
            "mean_dissim_condition_0": m["rdm_geometry"]["mean_dissimilarity_0"],
            "mean_dissim_condition_1": m["rdm_geometry"]["mean_dissimilarity_1"],
            "rdm_spearman_r":         m["rdm_geometry"]["rdm_spearman_r"],
            "rdm_spearman_p":         m["rdm_geometry"]["rdm_spearman_p"],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV summary saved to {csv_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5: Geometric comparison — UMAP, intrinsic dimensionality, "
            "and RDM geometry across noise conditions."
        )
    )
    parser.add_argument(
        "--activations_dir", type=str, required=True,
        help="Stage 3 output directory (results/activations)."
    )
    parser.add_argument(
        "--calibration_dir", type=str, default="results/calibration",
        help="Stage 4 output directory (results/calibration)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/geometry",
        help="Directory for geometry outputs."
    )
    parser.add_argument(
        "--layers", type=str, nargs="+",
        default=["layer1", "layer2", "layer3", "layer4", "avgpool"],
        help="Layers to analyse (must match Stage 3 output)."
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=15,
        help="UMAP n_neighbors parameter."
    )
    parser.add_argument(
        "--min_dist", type=float, default=0.1,
        help="UMAP min_dist parameter."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for UMAP and subsampling."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Stage 5 — Geometric Comparison")
    print("=" * 60)
    print(f"  Activations dir : {args.activations_dir}")
    print(f"  Calibration dir : {args.calibration_dir}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Layers          : {args.layers}")
    print(f"  UMAP n_neighbors: {args.n_neighbors}")
    print(f"  UMAP min_dist   : {args.min_dist}")
    print("=" * 60)

    run_geometric_comparison(
        activations_dir = args.activations_dir,
        calibration_dir = args.calibration_dir,
        output_dir      = args.output_dir,
        layer_names     = args.layers,
        n_neighbors     = args.n_neighbors,
        min_dist        = args.min_dist,
        seed            = args.seed,
    )

    print("\nDone.")
