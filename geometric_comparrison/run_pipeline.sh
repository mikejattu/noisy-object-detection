#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# Runs the full noise-representations pipeline (Stages 3–5) with sensible
# defaults. Edit the variables below to match your paths and hardware.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# To run with a human RDM (Stage 4):
#   HUMAN_RDM=data/human_rdm.npz ./run_pipeline.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-output/annotations}"
RESULTS_DIR="${RESULTS_DIR:-results}"
HUMAN_RDM="${HUMAN_RDM:-}"          # optional — leave empty until RDMs ready
DEVICE="${DEVICE:-cpu}"             # set to 'cuda' if GPU available
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LAYERS="layer1 layer2 layer3 layer4 avgpool"
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================"
echo " Noise, Representations, and Geometry"
echo " Full pipeline run"
echo "========================================"
echo "  Annotations : $ANNOTATIONS_DIR"
echo "  Results dir : $RESULTS_DIR"
echo "  Device      : $DEVICE"
echo "  Human RDM   : ${HUMAN_RDM:-not provided}"
echo "========================================"

# Stage 3 — ConvNet extraction
echo ""
echo ">>> Stage 3: ConvNet representation extraction"
python stage3/extract_representations.py \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --output_dir      "$RESULTS_DIR/activations" \
    --layers          $LAYERS \
    --batch_size      "$BATCH_SIZE" \
    --device          "$DEVICE" \
    --seed            "$SEED"

# Stage 4 — Noise calibration
echo ""
echo ">>> Stage 4: Noise calibration"
HUMAN_RDM_ARG=""
if [ -n "$HUMAN_RDM" ]; then
    HUMAN_RDM_ARG="--human_rdm $HUMAN_RDM"
fi
python stage4/noise_calibration.py \
    --activations_dir "$RESULTS_DIR/activations" \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --output_dir      "$RESULTS_DIR/calibration" \
    --layers          $LAYERS \
    --device          "$DEVICE" \
    $HUMAN_RDM_ARG

# Stage 5 — Geometric comparison
echo ""
echo ">>> Stage 5: Geometric comparison"
python stage5/geometric_comparison.py \
    --activations_dir "$RESULTS_DIR/activations" \
    --calibration_dir "$RESULTS_DIR/calibration" \
    --output_dir      "$RESULTS_DIR/geometry" \
    --layers          $LAYERS \
    --seed            "$SEED"

echo ""
echo "========================================"
echo " Pipeline complete."
echo " Results saved to: $RESULTS_DIR"
echo "========================================"
