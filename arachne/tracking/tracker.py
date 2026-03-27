"""Experiment tracking: saves config, metrics, and artifacts as JSON."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """Manages saving of experiment artifacts and metrics."""

    def __init__(self, experiment_name: str, output_dir: Path = Path("models")):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.experiment_dir = output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._metrics: dict = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": None,
            "data": {},
            "cv_results": {},
            "test_metrics": {},
            "status": "running",
        }

    def log_config(self, config: dict) -> None:
        import yaml
        with open(self.experiment_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def log_data_info(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        class_distribution: dict,
    ) -> None:
        self._metrics["data"] = {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_total": n_train + n_val + n_test,
            "class_distribution": {k: int(v) for k, v in class_distribution.items()},
        }

    def log_cv_results(self, cv_results: dict) -> None:
        self._metrics["cv_results"] = cv_results

    def log_test_metrics(self, metrics: dict) -> None:
        self._metrics["test_metrics"] = metrics

    def log_duration(self, seconds: float) -> None:
        self._metrics["duration_seconds"] = round(seconds, 2)

    def finalize(self) -> None:
        self._metrics["status"] = "completed"
        self._save_metrics()

    def get_summary(self) -> dict:
        return self._metrics.copy()

    def _save_metrics(self) -> None:
        with open(self.experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, ensure_ascii=False)


def load_all_experiments(models_dir: Path = Path("models")) -> list[dict]:
    """Load all experiment metrics from the models directory."""
    experiments = []
    for metrics_file in sorted(models_dir.glob("*/metrics.json"), reverse=True):
        try:
            with open(metrics_file, encoding="utf-8") as f:
                data = json.load(f)
                data["_path"] = str(metrics_file.parent)
                experiments.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return experiments
