"""Classical ML classifiers (sklearn pipelines)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from arachne.features.extractors import get_feature_extractor
from arachne.models.base import BaseTableClassifier


MODEL_FILE = "pipeline.joblib"


def _build_sklearn_classifier(model_config: dict):
    """Instantiate a sklearn classifier from config."""
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "linear_svc":
        # LinearSVC doesn't have predict_proba; wrap with calibration
        svc = LinearSVC(**params)
        return CalibratedClassifierCV(svc, cv=3)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(
            f"Unknown classical model type: '{model_type}'. "
            "Available: logistic_regression, linear_svc, random_forest, gradient_boosting"
        )


class ClassicalClassifier(BaseTableClassifier):
    """sklearn Pipeline: TF-IDF vectorizer + classifier."""

    def __init__(self, model_config: dict, features_config: dict):
        self._model_config = model_config
        self._features_config = features_config
        self._pipeline: Pipeline | None = None
        self._classes: list[str] = []

    def fit(
        self,
        texts_train: list[str],
        labels_train: list[str],
        texts_val: list[str] | None = None,
        labels_val: list[str] | None = None,
    ) -> None:
        vectorizer = get_feature_extractor(self._features_config)
        classifier = _build_sklearn_classifier(self._model_config)

        self._pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ])
        self._pipeline.fit(texts_train, labels_train)
        self._classes = list(self._pipeline.classes_)

    def predict(self, texts: list[str]) -> list[str]:
        if self._pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return list(self._pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._pipeline.predict_proba(texts)

    def get_classes(self) -> list[str]:
        return self._classes

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, directory / MODEL_FILE)

    @classmethod
    def load(cls, directory: Path) -> "ClassicalClassifier":
        pipeline = joblib.load(directory / MODEL_FILE)
        instance = cls.__new__(cls)
        instance._pipeline = pipeline
        instance._classes = list(pipeline.classes_)
        instance._model_config = {}
        instance._features_config = {}
        return instance
