"""Constantes et énumérations du projet Arachne."""
from __future__ import annotations

from enum import Enum


class Label(str, Enum):
    """Classes de classification des tableaux."""
    BATIMENT = "batiment"
    VEHICULE = "vehicule"
    SINISTRE = "sinistre"
    AUTRE = "autre"


class SourceDonnees(str, Enum):
    """Sources de données supportées."""
    POSTGRESQL = "postgresql"
    LOCAL = "local"


class TypeModele(str, Enum):
    """Types de modèles de classification disponibles."""
    REGRESSION_LOGISTIQUE = "logistic_regression"
    SVM_LINEAIRE = "linear_svc"
    FORET_ALEATOIRE = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    CAMEMBERT = "camembert"
    COMPLEMENT_NB = "complement_nb"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    ENSEMBLE_VOTE = "ensemble_vote"


class TypeFeatures(str, Enum):
    """Stratégies d'extraction de features disponibles."""
    TFIDF = "tfidf"
    TOKENIZER_TRANSFORMER = "transformer_tokenizer"
    TFIDF_SEPARE = "tfidf_separe"
    FEATURES_EXPLICITES = "features_explicites"
    TFIDF_LEMMATISE = "tfidf_lemmatise"


SEPARATEUR_CONTENU = "__CONTENU__"

# Listes dérivées des enums, utiles pour sklearn et les métriques
LABELS: list[str] = [label.value for label in Label]
LABEL_VERS_ID: dict[str, int] = {label.value: i for i, label in enumerate(Label)}
ID_VERS_LABEL: dict[int, str] = {i: label.value for i, label in enumerate(Label)}

# Colonnes obligatoires dans le jeu de données
COLONNES_REQUISES: frozenset[str] = frozenset({"table_data", "label"})
