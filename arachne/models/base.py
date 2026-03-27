"""Abstract base class for all table classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseTableClassifier(ABC):
    """Common interface for all classifiers."""

    @abstractmethod
    def fit(self, texts_train: list[str], labels_train: list[str],
            texts_val: list[str] | None = None, labels_val: list[str] | None = None) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, texts: list[str]) -> list[str]:
        """Predict class labels."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities. Shape: (n_samples, n_classes)."""

    @abstractmethod
    def get_classes(self) -> list[str]:
        """Return ordered list of class names."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Save model artifacts to directory."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "BaseTableClassifier":
        """Load model from directory."""
