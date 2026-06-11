"""Model-registry types.

The registry normalises every model to two functions:

  - preprocess:        PIL.Image -> CHW tensor   (runs AFTER image-space corruption)
  - classifier.logits: tensor    -> logits       (over a canonical label space)

so the eval loop is model-agnostic. CLIP's zero-shot head is built inside its
load() and exposed through the same `logits` interface — the loop never
special-cases it. This seam is exactly where silent resolution/normalisation
mismatches hide, so the constants live in metadata and are guarded at load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch
from PIL import Image


@runtime_checkable
class Classifier(Protocol):
    """A loaded, eval-ready model emitting logits over a canonical label space."""

    num_classes: int
    label_space: str  # e.g. "imagenet1k" — canonical index order; asserted at load

    def logits(self, batch: torch.Tensor) -> torch.Tensor:  # (B, num_classes)
        ...


# Model-specific transform: post-corruption PIL image -> CHW float tensor (single image).
# Corruptions are applied in image space upstream; this runs strictly after them.
Preprocess = Callable[[Image.Image], torch.Tensor]

# Lazy factory: build + move to device + eval(). Downloads weights by name; never reads repo weights.
LoadFn = Callable[[torch.device], Classifier]


@dataclass(frozen=True)
class ModelMeta:
    family: str            # resnet50 | vit_b16 | convnext_b | clip_vit_b16
    objective: str         # supervised-1k | stylized-1k | clip-contrastive | augmix-deepaug
    source: str            # exact, resolvable identifier (timm / openclip / robustbench / repo)
    input_size: int        # e.g. 224
    normalization: str     # "imagenet" | "clip" | "inception" — constants that silently corrupt results
    axis: str              # which study axis this model sits on
    pole: str              # texture | shape | control  (poles meaningful for the matched pair)
    role: str              # one-line role in the design
    params_m: float        # parameter count (millions); metadata only
    removable: str         # what the study loses if this key is dropped
    expected_top1: float | None = None  # clean-set sanity target; None until measured/verified


@dataclass(frozen=True)
class ModelSpec:
    key: str               # stable id -> results 'model' column; never rename post-hoc
    load: LoadFn
    preprocess: Preprocess
    meta: ModelMeta
