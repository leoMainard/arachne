"""Classifieurs ML classiques basés sur des pipelines scikit-learn."""
from __future__ import annotations

from pathlib import Path

from io import BytesIO
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:
    from arachne.data.s3 import ConnecteurS3
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from arachne.constants import TypeModele
from arachne.features.extractors import obtenir_extracteur
from arachne.models.base import ClassifieurBase


FICHIER_MODELE = "pipeline.joblib"


class _XGBAvecEncodage(BaseEstimator, ClassifierMixin):
    """Wrapper XGBClassifier avec encodage automatique des labels string→int.

    Paramètres stockés à plat pour compatibilité avec sklearn.clone().
    """

    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params or {}

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._xgb = XGBClassifier(eval_metric="mlogloss", verbosity=0, **self.xgb_params)
        self._xgb.fit(X, y_enc)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._xgb.predict(X))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)


def _sparse_vers_dense(X):
    """Convertit une matrice creuse en tableau dense. Picklable par joblib."""
    return X.toarray() if hasattr(X, "toarray") else X


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
        # HistGradientBoostingClassifier ne supporte pas les matrices creuses (sparse)
        # produites par TF-IDF. On encapsule dans un sous-pipeline avec conversion dense.
        from sklearn.preprocessing import FunctionTransformer
        return Pipeline([
            ("to_dense", FunctionTransformer(_sparse_vers_dense, accept_sparse=True)),
            ("gbm", HistGradientBoostingClassifier(**params)),
        ])

    elif type_modele in (TypeModele.COMPLEMENT_NB, "complement_nb"):
        return ComplementNB(**params)

    elif type_modele in (TypeModele.LIGHTGBM, "lightgbm"):
        # Import optionnel : lightgbm n'est requis que pour ce modèle.
        try:
            from lightgbm import LGBMClassifier
        except ImportError as erreur:
            raise ImportError(
                "lightgbm est requis pour ce modèle. "
                "Installez avec : pip install lightgbm"
            ) from erreur
        return LGBMClassifier(verbose=-1, **params)

    elif type_modele in (TypeModele.XGBOOST, "xgboost"):
        try:
            import xgboost  # noqa: F401
        except ImportError as erreur:
            raise ImportError(
                "xgboost est requis pour ce modèle. "
                "Installez avec : pip install xgboost"
            ) from erreur
        # XGBoost >= 2.0 n'accepte pas les labels string directement.
        return _XGBAvecEncodage(xgb_params=params)

    elif type_modele in (TypeModele.ENSEMBLE_VOTE, "ensemble_vote"):
        from sklearn.ensemble import VotingClassifier
        # Chaque votant encapsule son propre TF-IDF pour éviter les fuites.
        params_tfidf = {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 2}
        vote = params.get("vote", "soft")
        voters = [
            ("lr", Pipeline([
                ("tfidf", TfidfVectorizer(**params_tfidf)),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
            ])),
            ("svm", Pipeline([
                ("tfidf", TfidfVectorizer(**params_tfidf)),
                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", max_iter=2000), cv=3)),
            ])),
            ("rf", Pipeline([
                ("tfidf", TfidfVectorizer(**params_tfidf)),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1)),
            ])),
        ]
        return VotingClassifier(estimators=voters, voting=vote)

    else:
        types_disponibles = [t.value for t in TypeModele if t != TypeModele.CAMEMBERT]
        raise ValueError(
            f"Type de modèle classique inconnu : '{type_modele}'. "
            f"Types disponibles : {types_disponibles}"
        )


def _est_autonome(config_modele: dict) -> bool:
    """Retourne True si le modèle gère ses propres features (ex: ensemble_vote)."""
    return config_modele.get("type") in (TypeModele.ENSEMBLE_VOTE, "ensemble_vote")


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

    @classmethod
    def construire_pipeline(cls, config_modele: dict, config_features: dict) -> Pipeline:
        """Construit un pipeline sklearn non entraîné.

        Utilisé notamment pour la validation croisée dans le trainer,
        afin d'éviter de dépendre de fonctions privées depuis l'extérieur.

        Args:
            config_modele: Section ``model`` du fichier YAML.
            config_features: Section ``features`` du fichier YAML.

        Retours:
            Pipeline sklearn (vectoriseur + classifieur), non entraîné.
        """
        classifieur = _construire_classifieur_sklearn(config_modele)
        if _est_autonome(config_modele):
            return Pipeline([("classifieur", classifieur)])
        return Pipeline([
            ("vectoriseur", obtenir_extracteur(config_features)),
            ("classifieur", classifieur),
        ])

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
        classifieur = _construire_classifieur_sklearn(self._config_modele)

        if _est_autonome(self._config_modele):
            # L'ensemble gère son propre TF-IDF : pas de vectoriseur externe.
            self._pipeline = Pipeline([("classifieur", classifieur)])
        else:
            vectoriseur = obtenir_extracteur(self._config_features)
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

    def sauvegarder(
        self,
        repertoire: Path | None,
        connecteur_s3: "ConnecteurS3 | None" = None,
        prefixe_s3: str = "",
    ) -> None:
        """Sauvegarde le pipeline sklearn via joblib (local et/ou S3).

        Args:
            repertoire: Répertoire local de destination.
                        Passer ``None`` pour ne pas écrire sur le disque.
            connecteur_s3: Si fourni, upload le pipeline vers S3 via ``envoyer_objet``
                           (sérialisation en mémoire, pas d'écriture disque intermédiaire).
            prefixe_s3: Préfixe de la clé S3, ex : ``"arachne/exp_123/model"``.
        """
        if repertoire is not None:
            repertoire.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, repertoire / FICHIER_MODELE)

        if connecteur_s3 is not None:
            buf = BytesIO()
            joblib.dump(self._pipeline, buf)
            buf.seek(0)
            cle = f"{prefixe_s3.rstrip('/')}/{FICHIER_MODELE}"
            connecteur_s3.envoyer_objet(buf.read(), cle)

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
