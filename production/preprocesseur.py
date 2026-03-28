"""Préprocesseur de tableaux pour l'inférence en production.

Ce module est autonome : aucune dépendance envers le package arachne.
Dépendances requises : pyyaml uniquement.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml


def _charger_config_experience(repertoire_experience: Path) -> dict:
    """Lit la configuration sauvegardée lors de l'entraînement.

    Args:
        repertoire_experience: Chemin du répertoire d'expérience Arachne.

    Retours:
        Dictionnaire de configuration de l'expérience.

    Raises:
        FileNotFoundError: Si config.yaml est absent du répertoire.
    """
    chemin_config = repertoire_experience / "config.yaml"
    if not chemin_config.exists():
        raise FileNotFoundError(
            f"config.yaml introuvable dans {repertoire_experience}. "
            "Vérifiez que le chemin pointe vers un répertoire d'expérience valide."
        )
    with open(chemin_config, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class PreprocesseurProduction:
    """Préprocesseur de tableaux pour l'inférence en production.

    Reproduit exactement la logique de prétraitement appliquée lors de
    l'entraînement, en lisant la configuration depuis le répertoire d'expérience.

    Ce module est autonome : il n'importe pas le package arachne.

    Args:
        config: Dictionnaire de configuration de production.
                Doit contenir ``modele.repertoire_experience``.
                Peut contenir une section ``preprocessing`` pour surcharger
                les paramètres de l'expérience.

    Exemple:
        >>> config = {"modele": {"repertoire_experience": "models/tfidf_logistic_..."}}
        >>> prep = PreprocesseurProduction(config)
        >>> texte = prep.transformer([["Col1", "Col2"], ["Val1", "Val2"]])
    """

    def __init__(self, config: dict) -> None:
        repertoire = Path(config["modele"]["repertoire_experience"])
        config_experience = _charger_config_experience(repertoire)

        # Paramètres de l'expérience surchargés par ceux de la config de production
        config_preprocessing = {
            **config_experience.get("preprocessing", {}),
            **config.get("preprocessing", {}),
        }

        self._lignes_entetes: int = config_preprocessing.get("header_rows", 1)
        self._poids_entetes: int = config_preprocessing.get("header_weight", 3)
        self._max_cellules: int = config_preprocessing.get("max_content_cells", 200)
        self._longueur_max: int | None = config_preprocessing.get("max_length", None)

    def transformer(self, tableau: list[list]) -> str:
        """Convertit un tableau en représentation textuelle pondérée.

        Les en-têtes sont répétés ``poids_entetes`` fois pour amplifier leur
        influence lors de la vectorisation.

        Args:
            tableau: Matrice 2D du tableau (list[list[str]]).

        Retours:
            Représentation textuelle du tableau, prête pour la prédiction.
        """
        if not tableau:
            return ""

        nb_lignes_entetes = min(self._lignes_entetes, len(tableau))
        entetes = tableau[:nb_lignes_entetes]
        contenu = tableau[nb_lignes_entetes:]

        cellules_entetes = [
            self._nettoyer(cellule)
            for ligne in entetes
            for cellule in ligne
            if self._nettoyer(cellule)
        ]
        texte_entetes = " | ".join(cellules_entetes)

        cellules_contenu = [
            self._nettoyer(cellule)
            for ligne in contenu
            for cellule in ligne
            if self._nettoyer(cellule)
        ][:self._max_cellules]
        texte_contenu = " | ".join(cellules_contenu)

        parties = [texte_entetes] * self._poids_entetes
        if texte_contenu:
            parties.append(texte_contenu)

        texte = " ".join(p for p in parties if p)

        if self._longueur_max:
            texte = texte[:self._longueur_max]

        return texte

    def transformer_lot(self, tableaux: list[list[list]]) -> list[str]:
        """Convertit une liste de tableaux en représentations textuelles.

        Args:
            tableaux: Liste de matrices 2D.

        Retours:
            Liste de représentations textuelles, une par tableau.
        """
        return [self.transformer(tableau) for tableau in tableaux]

    @staticmethod
    def _nettoyer(valeur) -> str:
        """Normalise une cellule en chaîne de caractères propre."""
        if valeur is None:
            return ""
        return re.sub(r"\s+", " ", str(valeur).strip())
