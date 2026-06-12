"""Per-model load + preprocess implementations, wired from the verified recipes.

Every model is normalised to:
  - a `load_<key>(device) -> Classifier` factory (lazy: downloads weights by name
    on first call, asserts the eval config, returns an eval-mode model on `device`);
  - a `preprocess_<key>` callable (PIL.Image -> CHW tensor), built lazily on first
    use so importing the registry stays cheap and pulls no weights.

Heavy imports (timm / torchvision / open_clip / robustbench) live INSIDE the
functions, so the registry can be imported without every backend installed.

Constants verified against source (see model cards / library source in zoo.py).
A wrong normalization constant silently corrupts results, so timm models also
assert their resolved config at load time.
"""
from __future__ import annotations

from typing import Callable

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_SIN_WEIGHTS_URL = (
    "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/"
    "6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar"
)


# --------------------------------------------------------------------------- #
# Uniform classifier wrappers — every model exposes .logits(batch) -> (B, 1000).
# --------------------------------------------------------------------------- #
class _ModuleClassifier:
    """For models whose module already returns imagenet1k-ordered logits
    (timm models, the SIN torchvision ResNet, RobustBench's normalize+model Sequential)."""

    num_classes = 1000
    label_space = "imagenet1k"

    def __init__(self, module: "torch.nn.Module") -> None:
        self._module = module

    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        return self._module(batch)


class _ClipZeroShotClassifier:
    """CLIP: logits = logit_scale * normalize(encode_image(x)) @ text_classifier.

    `text_classifier` is (embed_dim, num_classes), L2-normalised per column, already
    oriented for image_features @ classifier (do NOT transpose)."""

    num_classes = 1000
    label_space = "imagenet1k"

    def __init__(self, model: "torch.nn.Module", text_classifier: torch.Tensor) -> None:
        self._model = model
        self._clf = text_classifier

    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self._model.logit_scale.exp() * feats @ self._clf


# --------------------------------------------------------------------------- #
# Lazy-preprocess plumbing + transform builders.
# --------------------------------------------------------------------------- #
def _lazy(builder: Callable[[], Callable]) -> Callable:
    """Wrap a transform builder so the (heavy-import) transform is built once, on
    first call. Returns a Preprocess: PIL.Image -> tensor."""
    cache: dict = {}

    def preprocess(img):
        tf = cache.get("tf")
        if tf is None:
            tf = builder()
            cache["tf"] = tf
        return tf(img)

    return preprocess


def _close(a, b, eps: float = 1e-6) -> bool:
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def _timm_data_config(model):
    import timm

    # resolve_model_data_config is the current name; older timm used resolve_data_config.
    if hasattr(timm.data, "resolve_model_data_config"):
        return timm.data.resolve_model_data_config(model)
    return timm.data.resolve_data_config({}, model=model)


def _build_timm_transform(*, input_size, interpolation, mean, std, crop_pct):
    """timm's own eval transform from explicit constants — no model/weights needed.
    Reproduces resize(floor(input/crop_pct)) -> center-crop -> ToTensor -> Normalize."""
    import timm

    return timm.data.create_transform(
        input_size=(3, input_size, input_size),
        is_training=False,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
    )


def _build_imagenet_eval_transform(*, mean, std, resize=256, crop=224, normalize=True):
    """Standard ImageNet eval transform (torchvision). `normalize=False` for models
    that normalise internally (RobustBench)."""
    from torchvision import transforms as T

    steps = [
        T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(crop),
        T.ToTensor(),
    ]
    if normalize:
        steps.append(T.Normalize(mean=mean, std=std))
    return T.Compose(steps)


def _build_clip_transform():
    """open_clip ViT-B-16 eval transform (preprocess_val equivalent) without pulling
    weights: Resize(224, bicubic, shortest-side) -> CenterCrop(224) -> ToTensor -> Normalize(CLIP)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ]
    )


def _assert_timm_cfg(model, *, mean, std, interpolation, key):
    cfg = _timm_data_config(model)
    if not (_close(cfg["mean"], mean) and _close(cfg["std"], std)):
        raise AssertionError(
            f"{key}: resolved norm {cfg['mean']}/{cfg['std']} != expected {mean}/{std} "
            f"(checkpoint variant changed? this silently corrupts results)"
        )
    if cfg["interpolation"] != interpolation:
        raise AssertionError(
            f"{key}: resolved interpolation {cfg['interpolation']!r} != expected {interpolation!r}"
        )
    return cfg


# --------------------------------------------------------------------------- #
# resnet50_in1k — timm resnet50.tv_in1k (texture pole / reference). ImageNet norm, bilinear.
# --------------------------------------------------------------------------- #
def load_resnet50_in1k(device: torch.device):
    import timm

    model = timm.create_model("resnet50.tv_in1k", pretrained=True, num_classes=1000)
    _assert_timm_cfg(model, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bilinear", key="resnet50_in1k")
    return _ModuleClassifier(model.eval().to(device))


preprocess_resnet50_in1k = _lazy(
    lambda: _build_timm_transform(
        input_size=224, interpolation="bilinear", mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct=0.875
    )
)


# --------------------------------------------------------------------------- #
# resnet50_sin — Geirhos pure-SIN ResNet-50 (shape pole). torchvision arch + Bitbucket weights.
# --------------------------------------------------------------------------- #
def load_resnet50_sin(device: torch.device):
    import torchvision

    model = torchvision.models.resnet50(weights=None)
    try:
        ckpt = torch.hub.load_state_dict_from_url(_SIN_WEIGHTS_URL, map_location="cpu")
    except Exception:
        # torch>=2.6 defaults weights_only=True, which can reject this legacy pickle (trusted source).
        ckpt = torch.hub.load_state_dict_from_url(_SIN_WEIGHTS_URL, map_location="cpu", weights_only=False)
    # Checkpoint was saved from a DataParallel wrap -> strip the leading 'module.' prefix.
    state = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"resnet50_sin state_dict mismatch: missing={missing}, unexpected={unexpected}")
    return _ModuleClassifier(model.eval().to(device))


preprocess_resnet50_sin = _lazy(
    lambda: _build_imagenet_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, resize=256, crop=224)
)


# --------------------------------------------------------------------------- #
# vit_b16_in1k — timm augreg2 in21k->in1k. INCEPTION norm (0.5/0.5), bicubic, crop_pct 0.9.
# --------------------------------------------------------------------------- #
def load_vit_b16_in1k(device: torch.device):
    import timm

    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=1000
    )
    _assert_timm_cfg(model, mean=INCEPTION_MEAN, std=INCEPTION_STD, interpolation="bicubic", key="vit_b16_in1k")
    return _ModuleClassifier(model.eval().to(device))


preprocess_vit_b16_in1k = _lazy(
    lambda: _build_timm_transform(
        input_size=224, interpolation="bicubic", mean=INCEPTION_MEAN, std=INCEPTION_STD, crop_pct=0.9
    )
)


# --------------------------------------------------------------------------- #
# convnext_b_in1k — timm convnext_base.fb_in1k. ImageNet norm, bicubic, crop_pct 0.875.
# --------------------------------------------------------------------------- #
def load_convnext_b_in1k(device: torch.device):
    import timm

    model = timm.create_model("convnext_base.fb_in1k", pretrained=True, num_classes=1000)
    _assert_timm_cfg(model, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bicubic", key="convnext_b_in1k")
    return _ModuleClassifier(model.eval().to(device))


preprocess_convnext_b_in1k = _lazy(
    lambda: _build_timm_transform(
        input_size=224, interpolation="bicubic", mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct=0.875
    )
)


# --------------------------------------------------------------------------- #
# clip_vit_b16 — open_clip ViT-B-16 (openai). Zero-shot imagenet head built at load.
# --------------------------------------------------------------------------- #
def load_clip_vit_b16(device: torch.device):
    import open_clip

    # pretrained="openai" auto-applies quick_gelu and pulls weights from HF Hub.
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        text_classifier = open_clip.build_zero_shot_classifier(
            model,
            tokenizer,
            open_clip.IMAGENET_CLASSNAMES,        # 1000 names, standard synset order (idx0='tench')
            open_clip.OPENAI_IMAGENET_TEMPLATES,  # 80-template OpenAI ensemble
            num_classes_per_batch=10,
            device=device,
            use_tqdm=False,
        )  # -> (embed_dim=512, num_classes=1000), L2-normalised columns; do NOT transpose
    return _ClipZeroShotClassifier(model, text_classifier)


preprocess_clip_vit_b16 = _lazy(_build_clip_transform)


# --------------------------------------------------------------------------- #
# resnet50_augmix — RobustBench Hendrycks2020AugMix. Normalization BAKED INTO the model.
# --------------------------------------------------------------------------- #
def load_resnet50_augmix(device: torch.device):
    from robustbench.utils import load_model

    # Returns nn.Sequential(normalize -> resnet50); model(batch) -> (B, 1000) logits.
    model = load_model(model_name="Hendrycks2020AugMix", dataset="imagenet", threat_model="corruptions")
    return _ModuleClassifier(model.eval().to(device))


# NOTE: normalize=False — RobustBench's ImageNormalizer normalises inside the model;
# the transform must output [0, 1] tensors or inputs get double-normalised.
preprocess_resnet50_augmix = _lazy(
    lambda: _build_imagenet_eval_transform(mean=None, std=None, resize=256, crop=224, normalize=False)
)
