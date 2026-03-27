"""Metrics computation and plot generation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> dict:
    """Compute full classification metrics."""
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    per_class = {}
    for label in classes:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "per_class": per_class,
    }


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[0],
    )
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=axes[1],
        vmin=0, vmax=1,
    )
    axes[1].set_title(f"{title} — Normalized")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_plot(metrics: dict, output_path: Path, title: str = "Metrics") -> None:
    """Save per-class metrics as a bar chart."""
    per_class = metrics.get("per_class", {})
    if not per_class:
        return

    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    ax.bar(x, recall, width, label="Recall", color="#DD8452")
    ax.bar(x + width, f1, width, label="F1", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=metrics.get("accuracy", 0), color="red", linestyle="--", alpha=0.5, label=f"Accuracy ({metrics['accuracy']:.3f})")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
