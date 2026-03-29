"""Chargement et fusion des configurations d'expériences."""
from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusionne deux dicts en profondeur ; les valeurs de override ont priorité."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def charger_config(config_path: str | Path) -> dict:
    """Charge la configuration d'une expérience fusionnée avec les valeurs par défaut.

    Args:
        config_path: Chemin vers le fichier YAML de l'expérience.

    Retours:
        Dictionnaire de configuration fusionné (base.yaml + expérience).
    """
    config_path = Path(config_path)

    if config_path.parent.name == "experiments":
        base_path = config_path.parent.parent / "base.yaml"
    else:
        base_path = config_path.parent / "base.yaml"

    base_config: dict = {}
    if base_path.exists():
        with open(base_path, encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

    with open(config_path, encoding="utf-8") as f:
        experiment_config = yaml.safe_load(f) or {}

    return _deep_merge(base_config, experiment_config)
