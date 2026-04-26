"""models — NIH ChestX-ray14 model registry."""
from __future__ import annotations

from models.densenet import DenseNet121Classifier, get_densenet
from models.swin     import SwinClassifier,        get_swin
from models.hybrid   import HybridClassifier,      get_hybrid

__all__ = [
    "DenseNet121Classifier", "get_densenet",
    "SwinClassifier",        "get_swin",
    "HybridClassifier",      "get_hybrid",
    "get_model",
]

_FACTORIES = {
    "densenet": get_densenet,
    "swin":     get_swin,
    "hybrid":   get_hybrid,
}


def get_model(model_name: str, config):
    """Instantiate and return a model by name.

    Args:
        model_name: "densenet", "swin", or "hybrid" (case-insensitive).
        config:     Project config module passed through to the factory.

    Returns:
        The model on the appropriate device.

    Raises:
        ValueError: If model_name is not recognised.
    """
    key = model_name.lower()
    if key not in _FACTORIES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {list(_FACTORIES)}"
        )
    return _FACTORIES[key](config)
