"""The six candidate models (see README 'Models' table).

Study-design metadata is filled now. The load()/preprocess() recipes — exact
checkpoint names, per-model normalisation constants, and CLIP's zero-shot head —
are filled by the verification pass; until then load()/preprocess() raise, so a
half-wired model can never silently produce numbers.

Sources mirror the README contract exactly:
  ResNet-50            -> timm
  ResNet-50 (SIN)      -> Geirhos texture-vs-shape
  ViT-B/16             -> timm
  ConvNeXt-B           -> timm
  CLIP ViT-B/16        -> OpenCLIP
  ResNet-50 AugMix     -> RobustBench
"""
from __future__ import annotations

from .registry import register
from .spec import ModelMeta, ModelSpec


def _pending(key: str, what: str):
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"{key}: {what} recipe pending verification pass (timm/openclip/robustbench)"
        )

    return _raise


_SPECS = [
    ModelSpec(
        key="resnet50_in1k",
        load=_pending("resnet50_in1k", "load"),
        preprocess=_pending("resnet50_in1k", "preprocess"),
        meta=ModelMeta(
            family="resnet50",
            objective="supervised-1k",
            source="timm:resnet50  # VERIFY weights variant (e.g. .a1_in1k vs .tv_in1k)",
            input_size=224,
            normalization="imagenet",
            axis="shape-texture",
            pole="texture",
            role="reference + texture-biased pole",
            params_m=25.6,
            removable="lose the texture pole and the standard-training reference point",
        ),
    ),
    ModelSpec(
        key="resnet50_sin",
        load=_pending("resnet50_sin", "load"),
        preprocess=_pending("resnet50_sin", "preprocess"),
        meta=ModelMeta(
            family="resnet50",
            objective="stylized-1k",
            source="geirhos/texture-vs-shape:resnet50_trained_on_SIN  # VERIFY (SIN vs SIN+IN)",
            input_size=224,
            normalization="imagenet",
            axis="shape-texture",
            pole="shape",
            role="shape-biased pole, architecture-matched to reference",
            params_m=25.6,
            removable="lose the shape pole; the primary axis collapses to a single point",
        ),
    ),
    ModelSpec(
        key="vit_b16_in1k",
        load=_pending("vit_b16_in1k", "load"),
        preprocess=_pending("vit_b16_in1k", "preprocess"),
        meta=ModelMeta(
            family="vit_b16",
            objective="supervised-1k",
            # VERIFY: in1k vs in21k-finetuned, AND normalisation — many timm ViTs use 0.5/0.5 (inception),
            # not ImageNet mean/std. Getting this wrong is exactly the silent-mismatch failure mode.
            source="timm:vit_base_patch16_224  # VERIFY variant + norm",
            input_size=224,
            normalization="inception",
            axis="spatial-integration",
            pole="control",
            role="attention vs. convolution (spatial integration)",
            params_m=86.6,
            removable="lose the attention-vs-conv contrast",
        ),
    ),
    ModelSpec(
        key="convnext_b_in1k",
        load=_pending("convnext_b_in1k", "load"),
        preprocess=_pending("convnext_b_in1k", "preprocess"),
        meta=ModelMeta(
            family="convnext_b",
            objective="supervised-1k",
            source="timm:convnext_base  # VERIFY weights variant + norm",
            input_size=224,
            normalization="imagenet",
            axis="conv-op-vs-recipe",
            pole="control",
            role="separates conv operation from modern training recipe",
            params_m=88.6,
            removable="lose the ability to attribute ViT effects to op vs. recipe",
        ),
    ),
    ModelSpec(
        key="clip_vit_b16",
        load=_pending("clip_vit_b16", "load (incl. zero-shot head)"),
        preprocess=_pending("clip_vit_b16", "preprocess"),
        meta=ModelMeta(
            family="clip_vit_b16",
            objective="clip-contrastive",
            # Zero-shot head (imagenet1k class prompts + templates) built inside load();
            # frozen in the spec for reproducibility. CLIP normalisation, NOT ImageNet.
            source="open_clip:ViT-B-16 (pretrained=openai)  # VERIFY pretrained tag",
            input_size=224,
            normalization="clip",
            axis="training-objective",
            pole="control",
            role="language-aligned objective (zero-shot head built at load)",
            params_m=86.2,
            removable="lose the training-objective contrast",
        ),
    ),
    ModelSpec(
        key="resnet50_augmix",
        load=_pending("resnet50_augmix", "load"),
        preprocess=_pending("resnet50_augmix", "preprocess"),
        meta=ModelMeta(
            family="resnet50",
            objective="augmix-deepaug",
            source="robustbench:Hendrycks2020AugMix_ResNet50  # VERIFY (AugMix vs DeepAug+AugMix)",
            input_size=224,
            normalization="imagenet",
            axis="explicit-robustness",
            pole="control",
            role="explicit-robustness control",
            params_m=25.6,
            removable="lose the explicit-robustness reference point",
        ),
    ),
]

for _spec in _SPECS:
    register(_spec)
