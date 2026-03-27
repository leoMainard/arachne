"""Feature extraction strategies."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
import numpy as np


def build_tfidf_vectorizer(params: dict) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer from config params."""
    ngram_range = params.get("ngram_range", [1, 2])
    return TfidfVectorizer(
        max_features=params.get("max_features", 15000),
        ngram_range=tuple(ngram_range),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 2),
        analyzer=params.get("analyzer", "word"),
        strip_accents=params.get("strip_accents", None),
        lowercase=params.get("lowercase", True),
    )


def get_feature_extractor(features_config: dict):
    """Return the appropriate feature extractor from config."""
    feature_type = features_config.get("type", "tfidf")
    params = features_config.get("params", {})

    if feature_type == "tfidf":
        return build_tfidf_vectorizer(params)
    else:
        raise ValueError(
            f"Unknown feature type '{feature_type}'. "
            "Transformer tokenization is handled inside the transformer model."
        )
