"""Model registry. Importing this package registers the six candidate models."""
from . import zoo  # noqa: F401  (registration side effect)
from .registry import get, keys, register, specs
from .spec import Classifier, LoadFn, ModelMeta, ModelSpec, Preprocess

__all__ = [
    "get",
    "keys",
    "register",
    "specs",
    "Classifier",
    "LoadFn",
    "ModelMeta",
    "ModelSpec",
    "Preprocess",
]
