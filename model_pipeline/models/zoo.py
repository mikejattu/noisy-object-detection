"""The six candidate models (see README 'Models' table).

load/preprocess come from loaders.py, wired from source-verified recipes; the
metadata below records the verified constants (exact checkpoint, normalization,
clean-set expected top-1) so the study design and the runtime guards are explicit.

Verified facts per model live in the `source`/`normalization`/`expected_top1`
fields and in loaders.py docstrings; sources were checked against the official
model cards / library source. Runtime residual risks (network egress, version
pins, the clean-set sanity guard) are summarised in module docstrings and the
project README — the real test is measuring clean top-1 on the GPU box.
"""
from __future__ import annotations

from . import loaders
from .registry import register
from .spec import ModelMeta, ModelSpec

_SPECS = [
    ModelSpec(
        key="resnet50_in1k",
        load=loaders.load_resnet50_in1k,
        preprocess=loaders.preprocess_resnet50_in1k,
        meta=ModelMeta(
            family="resnet50",
            objective="supervised-1k",
            source="timm:resnet50.tv_in1k",  # torchvision IMAGENET1K_V1 weights
            input_size=224,
            normalization="imagenet",
            axis="shape-texture",
            pole="texture",
            role="reference + texture-biased pole",
            params_m=25.6,
            removable="lose the texture pole and the standard-training reference point",
            expected_top1=76.13,
        ),
    ),
    ModelSpec(
        key="resnet50_sin",
        load=loaders.load_resnet50_sin,
        preprocess=loaders.preprocess_resnet50_sin,
        meta=ModelMeta(
            family="resnet50",
            objective="stylized-1k",
            source="geirhos/texture-vs-shape:resnet50_trained_on_SIN (bitbucket)",  # pure SIN, not SIN+IN
            input_size=224,
            normalization="imagenet",
            axis="shape-texture",
            pole="shape",
            role="shape-biased pole, architecture-matched to reference",
            params_m=25.6,
            removable="lose the shape pole; the primary axis collapses to a single point",
            expected_top1=60.18,
        ),
    ),
    ModelSpec(
        key="vit_b16_in1k",
        load=loaders.load_vit_b16_in1k,
        preprocess=loaders.preprocess_vit_b16_in1k,
        meta=ModelMeta(
            family="vit_b16",
            objective="supervised-1k",
            source="timm:vit_base_patch16_224.augreg2_in21k_ft_in1k",  # in21k-pretrained, in1k-finetuned
            input_size=224,
            normalization="inception",  # 0.5/0.5 — NOT ImageNet; verified vs HF config.json
            axis="spatial-integration",
            pole="control",
            role="attention vs. convolution (spatial integration)",
            params_m=86.6,
            removable="lose the attention-vs-conv contrast",
            expected_top1=85.108,
        ),
    ),
    ModelSpec(
        key="convnext_b_in1k",
        load=loaders.load_convnext_b_in1k,
        preprocess=loaders.preprocess_convnext_b_in1k,
        meta=ModelMeta(
            family="convnext_b",
            objective="supervised-1k",
            source="timm:convnext_base.fb_in1k",  # pure in1k (NOT in22k-ft)
            input_size=224,
            normalization="imagenet",
            axis="conv-op-vs-recipe",
            pole="control",
            role="separates conv operation from modern training recipe",
            params_m=88.6,
            removable="lose the ability to attribute ViT effects to op vs. recipe",
            expected_top1=83.82,
        ),
    ),
    ModelSpec(
        key="clip_vit_b16",
        load=loaders.load_clip_vit_b16,
        preprocess=loaders.preprocess_clip_vit_b16,
        meta=ModelMeta(
            family="clip_vit_b16",
            objective="clip-contrastive",
            source="open_clip:ViT-B-16 (pretrained=openai)",
            input_size=224,
            normalization="clip",  # OpenAI CLIP mean/std
            axis="training-objective",
            pole="control",
            role="language-aligned objective (zero-shot head built at load)",
            params_m=86.2,
            removable="lose the training-objective contrast",
            expected_top1=68.3,  # zero-shot
        ),
    ),
    ModelSpec(
        key="resnet50_augmix",
        load=loaders.load_resnet50_augmix,
        preprocess=loaders.preprocess_resnet50_augmix,
        meta=ModelMeta(
            family="resnet50",
            objective="augmix",  # AugMix only; DeepAug+AugMix is the separate Hendrycks2020Many key
            source="robustbench:Hendrycks2020AugMix (imagenet/corruptions)",
            input_size=224,
            normalization="imagenet (baked into model)",  # preprocess must NOT normalize
            axis="explicit-robustness",
            pole="control",
            role="explicit-robustness control",
            params_m=25.6,
            removable="lose the explicit-robustness reference point",
            expected_top1=77.34,
        ),
    ),
]

for _spec in _SPECS:
    register(_spec)
