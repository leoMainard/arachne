"""Configuration loading and merging."""
from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts; override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict:
    """Load experiment config merged with base defaults."""
    config_path = Path(config_path)

    # Resolve base.yaml relative to project root (2 levels up from configs/experiments/)
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
