# Noise, Representations, and Geometry

Investigating how noise affects the geometry of perceptual representations in humans and convolutional neural networks. Using manifold learning to characterise low-dimensional representational structure across noise conditions and compare the degradation trajectory between biological and artificial vision systems.

---

## Project overview

Standard comparisons between human and ANN perception focus on behavioural outputs — accuracy, reaction time, error patterns. This project moves the comparison one level deeper: rather than asking *whether* noise hurts performance, we ask *how* noise changes the geometric structure of representations, and whether that change looks the same in humans and ConvNets.

The pipeline has five stages:

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Stimulus design — object images, noise conditions | Planned |
| 2 | Psychophysics — similarity judgements, proxy model fitting | Planned |
| **3** | **ConvNet extraction — ResNet50 activations per layer per condition** | ✅ Complete |
| **4** | **Noise calibration — LPIPS, activation distances, layer selection** | ✅ Complete |
| **5** | **Geometric comparison — UMAP, dimensionality, RDM geometry** | ✅ Complete |

---

## Quickstart

```bash
git clone https://github.com/your-username/noise-representations.git
cd noise-representations
pip install -r requirements.txt

# Run the full pipeline
./run_pipeline.sh

# Or with a GPU and human RDMs
DEVICE=cuda HUMAN_RDM=data/human_rdm.npz ./run_pipeline.sh
```

---

## Dataset format

Annotations and images share the same folder:

```
output/annotations/
    000098.json
    000098.jpg
    001575.json
    001575.jpg
```

Each annotation JSON follows VOC2007 structure with a `difficult` flag per object:

```json
{
  "image_id": "000098",
  "dataset":  "voc2007",
  "objects": [
    { "class": "cat", "difficult": 0 }
  ]
}
```

---

## Stage 3 — ConvNet representation extraction

Extracts ResNet50 activations for each image grouped by difficulty condition. All five layers extracted by default — layer selection is deferred to Stage 4.

```bash
python stage3/extract_representations.py \
    --annotations_dir  output/annotations \
    --output_dir       results/activations
```

**Output:**
```
results/activations/
    {layer}/
        condition_0.npz    # activations (N, D), image_ids (N,)
        condition_1.npz
    metadata.json
```

---

## Stage 4 — Noise calibration

Establishes principled correspondences between conditions across three systems:

1. **LPIPS perceptual distance** — cross-condition image similarity
2. **Activation distance** — centroid separation and discriminability per layer
3. **Human RDM correlation** — Spearman r with psychophysics RDM *(placeholder until data ready)*

```bash
# Without human RDMs
python stage4/noise_calibration.py \
    --activations_dir  results/activations \
    --annotations_dir  output/annotations \
    --output_dir       results/calibration

# With human RDMs
python stage4/noise_calibration.py \
    --activations_dir  results/activations \
    --annotations_dir  output/annotations \
    --output_dir       results/calibration \
    --human_rdm        data/human_rdm.npz
```

**Human RDM format** (when ready):

| Key | Shape | Description |
|-----|-------|-------------|
| `rdm` | `(N, N)` | Symmetric dissimilarity matrix |
| `image_ids` | `(N,)` | Image IDs matching row/col order |

**Output:**
```
results/calibration/
    calibration_report.json
    calibration_report.csv
```

---

## Stage 5 — Geometric comparison

Compares representational geometry across conditions using:

- **Participation Ratio** — effective intrinsic dimensionality per condition
- **RDM geometry** — mean dissimilarity, variance, cross-condition Spearman r
- **UMAP** — joint 2D embedding of both conditions per layer

```bash
python stage5/geometric_comparison.py \
    --activations_dir  results/activations \
    --calibration_dir  results/calibration \
    --output_dir       results/geometry
```

**Output:**
```
results/geometry/
    metrics.json
    metrics.csv
    plots/
        {layer}_embedding.png
        {layer}_rdm.png
        dimensionality_all_layers.png
        summary.png
```

---

## Repository structure

```
noise-representations/
├── README.md
├── requirements.txt
├── run_pipeline.sh
├── .gitignore
├── LICENSE
├── stage3/
│   └── extract_representations.py
├── stage4/
│   └── noise_calibration.py
└── stage5/
    └── geometric_comparison.py
```

---

## Citation

```bibtex
@misc{noise-representations,
  title  = {Noise, Representations, and Geometry},
  year   = {2025},
  url    = {https://github.com/your-username/noise-representations}
}
```

---

## License

MIT
