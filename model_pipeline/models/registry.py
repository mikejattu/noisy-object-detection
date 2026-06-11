"""Process-global model registry.

Keyed by ModelSpec.key; configs select subsets, which is what makes each model
removable-with-loss. Loads are lazy — importing the registry does not pull weights.
"""
from __future__ import annotations

from .spec import ModelSpec

_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    if spec.key in _REGISTRY:
        raise KeyError(f"duplicate model key: {spec.key!r}")
    _REGISTRY[spec.key] = spec


def get(key: str) -> ModelSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model key {key!r}; registered: {keys()}") from None


def keys() -> list[str]:
    return sorted(_REGISTRY)


def specs() -> list[ModelSpec]:
    return [_REGISTRY[k] for k in keys()]
