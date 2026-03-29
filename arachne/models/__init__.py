"""Registre des modèles de classification."""
from __future__ import annotations

from arachne.constants import TypeFeatures
from arachne.models.base import ClassifieurBase
from arachne.models.classical import ClassifieurClassique

__all__ = ["ClassifieurBase", "ClassifieurClassique", "obtenir_modele"]


def obtenir_modele(config_modele: dict, config_features: dict) -> ClassifieurBase:
    """Instancie le modèle correspondant à la configuration.

    Args:
        config_modele: Section ``model`` du fichier YAML.
        config_features: Section ``features`` du fichier YAML.

    Retours:
        Instance du classifieur correspondant (non entraîné).

    Raises:
        ValueError: Si le type de features est inconnu.
    """
    type_features = config_features.get("type", TypeFeatures.TFIDF)

    if type_features in (TypeFeatures.TOKENIZER_TRANSFORMER, "transformer_tokenizer"):
        from arachne.models.transformer import ClassifieurTransformer
        return ClassifieurTransformer(config_modele, config_features)
    elif type_features in (
        TypeFeatures.TFIDF, "tfidf",
        TypeFeatures.TFIDF_SEPARE, "tfidf_separe",
        TypeFeatures.FEATURES_EXPLICITES, "features_explicites",
        TypeFeatures.TFIDF_LEMMATISE, "tfidf_lemmatise",
    ):
        return ClassifieurClassique(config_modele, config_features)
    else:
        raise ValueError(
            f"Type de features inconnu : '{type_features}'. "
            f"Types disponibles : {[t.value for t in TypeFeatures]}"
        )
