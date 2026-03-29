"""Utilitaires partagés du module production."""
from __future__ import annotations

from pathlib import Path

import yaml


def charger_config_experience(repertoire_experience: Path) -> dict:
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
