"""Model registry."""
from __future__ import annotations

from arachne.models.base import BaseTableClassifier
from arachne.models.classical import ClassicalClassifier

__all__ = ["BaseTableClassifier", "ClassicalClassifier", "get_model"]


def get_model(model_config: dict, features_config: dict) -> BaseTableClassifier:
    """Instantiate a model from config."""
    model_type = model_config.get("type", "")
    feature_type = features_config.get("type", "tfidf")

    if feature_type == "tfidf" or feature_type == "count":
        return ClassicalClassifier(model_config, features_config)
    elif feature_type == "transformer_tokenizer":
        from arachne.models.transformer import TransformerClassifier
        return TransformerClassifier(model_config, features_config)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
