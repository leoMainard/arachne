"""Classe de base abstraite pour tous les classifieurs de tableaux."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arachne.data.s3 import ConnecteurS3


class ClassifieurBase(ABC):
    """Interface commune à tous les classifieurs.

    Tout nouveau modèle doit hériter de cette classe et implémenter
    l'ensemble des méthodes abstraites.
    """

    @abstractmethod
    def entrainer(
        self,
        textes_train: list[str],
        labels_train: list[str],
        textes_val: list[str] | None = None,
        labels_val: list[str] | None = None,
    ) -> None:
        """Entraîne le modèle sur les données fournies.

        Args:
            textes_train: Représentations textuelles des tableaux d'entraînement.
            labels_train: Labels correspondants aux tableaux d'entraînement.
            textes_val: Textes de validation (optionnel, utilisé par les transformers).
            labels_val: Labels de validation (optionnel).
        """

    @abstractmethod
    def predire(self, textes: list[str]) -> list[str]:
        """Prédit les classes pour une liste de textes.

        Args:
            textes: Représentations textuelles des tableaux à classifier.

        Retours:
            Liste des labels prédits.
        """

    @abstractmethod
    def predire_probabilites(self, textes: list[str]) -> np.ndarray:
        """Calcule les probabilités d'appartenance à chaque classe.

        Args:
            textes: Représentations textuelles des tableaux.

        Retours:
            Tableau numpy de forme (n_échantillons, n_classes).
        """

    @abstractmethod
    def obtenir_classes(self) -> list[str]:
        """Retourne la liste ordonnée des classes connues du modèle.

        Retours:
            Liste de noms de classes dans l'ordre utilisé par predire_probabilites.
        """

    @abstractmethod
    def sauvegarder(
        self,
        repertoire: Path | None,
        connecteur_s3: "ConnecteurS3 | None" = None,
        prefixe_s3: str = "",
    ) -> None:
        """Sauvegarde les artefacts du modèle.

        Args:
            repertoire: Répertoire local de destination (créé si absent).
                        Passer ``None`` pour désactiver la sauvegarde locale.
            connecteur_s3: Connecteur S3 optionnel. Si fourni, les artefacts
                           sont également (ou exclusivement) uploadés vers S3.
            prefixe_s3: Préfixe de la clé S3, ex : ``"arachne/exp_123/model"``.
        """

    @classmethod
    @abstractmethod
    def charger(cls, repertoire: Path) -> "ClassifieurBase":
        """Charge un modèle depuis un répertoire sauvegardé.

        Args:
            repertoire: Chemin du répertoire contenant les artefacts du modèle.

        Retours:
            Instance du classifieur restaurée.
        """
