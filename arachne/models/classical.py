"""Classifieurs ML classiques basés sur des pipelines scikit-learn."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from arachne.constants import TypeModele
from arachne.features.extractors import obtenir_extracteur
from arachne.models.base import ClassifieurBase


FICHIER_MODELE = "pipeline.joblib"


def _construire_classifieur_sklearn(config_modele: dict):
    """Instancie le classifieur sklearn correspondant à la configuration.

    Args:
        config_modele: Section ``model`` du fichier YAML (type + params).

    Retours:
        Estimateur sklearn non entraîné.

    Raises:
        ValueError: Si le type de modèle est inconnu.
    """
    type_modele = config_modele["type"]
    params = config_modele.get("params", {})

    if type_modele == TypeModele.REGRESSION_LOGISTIQUE:
        return LogisticRegression(**params)
    elif type_modele == TypeModele.SVM_LINEAIRE:
        # LinearSVC n'implémente pas predict_proba nativement ;
        # on l'encapsule dans CalibratedClassifierCV pour obtenir des probabilités.
        svc = LinearSVC(**params)
        return CalibratedClassifierCV(svc, cv=3)
    elif type_modele == TypeModele.FORET_ALEATOIRE:
        return RandomForestClassifier(**params)
    elif type_modele == TypeModele.GRADIENT_BOOSTING:
        return HistGradientBoostingClassifier(**params)
    else:
        types_disponibles = [t.value for t in TypeModele if t != TypeModele.CAMEMBERT]
        raise ValueError(
            f"Type de modèle classique inconnu : '{type_modele}'. "
            f"Types disponibles : {types_disponibles}"
        )


class ClassifieurClassique(ClassifieurBase):
    """Classifieur basé sur un pipeline sklearn : vectoriseur TF-IDF + classifieur.

    Args:
        config_modele: Section ``model`` du fichier YAML.
        config_features: Section ``features`` du fichier YAML.

    Exemple:
        >>> clf = ClassifieurClassique(config_modele, config_features)
        >>> clf.entrainer(textes_train, labels_train)
        >>> predictions = clf.predire(textes_test)
    """

    def __init__(self, config_modele: dict, config_features: dict) -> None:
        self._config_modele = config_modele
        self._config_features = config_features
        self._pipeline: Pipeline | None = None
        self._classes: list[str] = []

    def entrainer(
        self,
        textes_train: list[str],
        labels_train: list[str],
        textes_val: list[str] | None = None,
        labels_val: list[str] | None = None,
    ) -> None:
        """Entraîne le pipeline TF-IDF + classifieur.

        Args:
            textes_train: Textes d'entraînement.
            labels_train: Labels d'entraînement.
            textes_val: Non utilisé pour les modèles classiques.
            labels_val: Non utilisé pour les modèles classiques.
        """
        vectoriseur = obtenir_extracteur(self._config_features)
        classifieur = _construire_classifieur_sklearn(self._config_modele)

        self._pipeline = Pipeline([
            ("vectoriseur", vectoriseur),
            ("classifieur", classifieur),
        ])
        self._pipeline.fit(textes_train, labels_train)
        self._classes = list(self._pipeline.classes_)

    def predire(self, textes: list[str]) -> list[str]:
        """Prédit les classes pour une liste de textes.

        Args:
            textes: Textes à classifier.

        Retours:
            Liste des labels prédits.

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.
        """
        if self._pipeline is None:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelez entrainer() d'abord.")
        return list(self._pipeline.predict(textes))

    def predire_probabilites(self, textes: list[str]) -> np.ndarray:
        """Calcule les probabilités d'appartenance à chaque classe.

        Args:
            textes: Textes à classifier.

        Retours:
            Tableau numpy (n_échantillons, n_classes).

        Raises:
            RuntimeError: Si le modèle n'a pas encore été entraîné.
        """
        if self._pipeline is None:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelez entrainer() d'abord.")
        return self._pipeline.predict_proba(textes)

    def obtenir_classes(self) -> list[str]:
        """Retourne les classes dans l'ordre du pipeline.

        Retours:
            Liste de noms de classes.
        """
        return self._classes

    def sauvegarder(self, repertoire: Path) -> None:
        """Sauvegarde le pipeline sklearn via joblib.

        Args:
            repertoire: Répertoire de destination.
        """
        repertoire.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, repertoire / FICHIER_MODELE)

    @classmethod
    def charger(cls, repertoire: Path) -> "ClassifieurClassique":
        """Charge un pipeline sklearn depuis un répertoire.

        Args:
            repertoire: Répertoire contenant le fichier pipeline.joblib.

        Retours:
            Instance de ClassifieurClassique restaurée.
        """
        pipeline = joblib.load(repertoire / FICHIER_MODELE)
        instance = cls.__new__(cls)
        instance._pipeline = pipeline
        instance._classes = list(pipeline.classes_)
        instance._config_modele = {}
        instance._config_features = {}
        return instance
