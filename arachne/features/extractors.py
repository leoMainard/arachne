"""Extracteurs de features pour la classification de tableaux."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from arachne.constants import TypeFeatures


def construire_vectoriseur_tfidf(params: dict) -> TfidfVectorizer:
    """Instancie un vectoriseur TF-IDF à partir des paramètres de configuration.

    Args:
        params: Dictionnaire de paramètres (section ``features.params`` du YAML).
                Clés supportées : max_features, ngram_range, sublinear_tf,
                min_df, analyzer, strip_accents, lowercase.

    Retours:
        Vectoriseur TF-IDF configuré (non entraîné).
    """
    plage_ngrams = params.get("ngram_range", [1, 2])
    return TfidfVectorizer(
        max_features=params.get("max_features", 15000),
        ngram_range=tuple(plage_ngrams),
        sublinear_tf=params.get("sublinear_tf", True),
        min_df=params.get("min_df", 2),
        analyzer=params.get("analyzer", "word"),
        strip_accents=params.get("strip_accents", None),
        lowercase=params.get("lowercase", True),
    )


def obtenir_extracteur(config_features: dict) -> TfidfVectorizer:
    """Retourne l'extracteur de features correspondant à la configuration.

    Args:
        config_features: Section ``features`` du fichier YAML de l'expérience.

    Retours:
        Extracteur de features instancié (non entraîné).

    Raises:
        ValueError: Si le type de features est inconnu ou géré ailleurs (transformer).
    """
    type_features = config_features.get("type", TypeFeatures.TFIDF)

    if type_features == TypeFeatures.TFIDF:
        return construire_vectoriseur_tfidf(config_features.get("params", {}))
    else:
        raise ValueError(
            f"Type de features '{type_features}' non géré ici. "
            "La tokenisation transformer est intégrée au modèle TransformerClassifieur."
        )
