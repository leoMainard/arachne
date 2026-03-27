"""Training pipeline with optional cross-validation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from arachne.data.loader import load_data
from arachne.data.preprocessing import split_dataset, tables_to_texts
from arachne.models import get_model
from arachne.training.evaluator import compute_metrics, save_confusion_matrix, save_metrics_plot
from arachne.tracking.tracker import ExperimentTracker

console = Console()


def run_experiment(config: dict) -> dict:
    """Run a full training experiment from config.

    Returns the experiment metrics dict.
    """
    exp_name = config.get("experiment", {}).get("name", "experiment")
    exp_description = config.get("experiment", {}).get("description", "")
    data_config = config["data"]
    preprocessing_config = config.get("preprocessing", {})
    training_config = config.get("training", {})

    tracker = ExperimentTracker(
        experiment_name=exp_name,
        output_dir=Path(config.get("tracking", {}).get("output_dir", "models")),
    )
    tracker.log_config(config)

    console.rule(f"[bold blue]{exp_name}")
    if exp_description:
        console.print(f"[dim]{exp_description}[/dim]")

    # --- Load data ---
    console.print("\n[bold]1. Loading data...[/bold]")
    df = load_data(config)
    console.print(f"   Loaded {len(df)} samples.")
    console.print(f"   Class distribution:\n{df['label'].value_counts().to_string()}")

    # --- Split ---
    df_train, df_val, df_test = split_dataset(
        df,
        test_size=data_config.get("test_size", 0.2),
        val_size=data_config.get("val_size", 0.1),
        stratify=data_config.get("stratify", True),
        random_seed=data_config.get("random_seed", 42),
    )
    console.print(f"\n   Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    tracker.log_data_info(
        n_train=len(df_train),
        n_val=len(df_val),
        n_test=len(df_test),
        class_distribution=df["label"].value_counts().to_dict(),
    )

    # --- Preprocess to text ---
    console.print("\n[bold]2. Preprocessing tables to text...[/bold]")
    texts_train = tables_to_texts(df_train["table_data"].tolist(), preprocessing_config)
    texts_val = tables_to_texts(df_val["table_data"].tolist(), preprocessing_config)
    texts_test = tables_to_texts(df_test["table_data"].tolist(), preprocessing_config)

    labels_train = df_train["label"].tolist()
    labels_val = df_val["label"].tolist()
    labels_test = df_test["label"].tolist()

    classes = sorted(data_config.get("labels", ["batiment", "vehicule", "sinistre", "autre"]))

    # --- Build model ---
    console.print("\n[bold]3. Building model...[/bold]")
    model = get_model(config.get("model", {}), config.get("features", {}))

    # --- Cross-validation (classical models only) ---
    cv_folds = training_config.get("cv_folds", 5)
    cv_results: dict = {}

    if cv_folds and cv_folds > 1:
        console.print(f"\n[bold]4. Cross-validation ({cv_folds} folds)...[/bold]")

        # For CV we use a fresh pipeline fitted on train+val texts
        texts_tv = texts_train + texts_val
        labels_tv = labels_train + labels_val

        from arachne.models.classical import ClassicalClassifier
        if isinstance(model, ClassicalClassifier):
            from arachne.features.extractors import get_feature_extractor
            from arachne.models.classical import _build_sklearn_classifier
            cv_pipeline = Pipeline([
                ("vectorizer", get_feature_extractor(config.get("features", {}))),
                ("classifier", _build_sklearn_classifier(config.get("model", {}))),
            ])
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=data_config.get("random_seed", 42))
            fold_scores = cross_val_score(
                cv_pipeline, texts_tv, labels_tv,
                cv=skf, scoring=training_config.get("scoring", "accuracy"), n_jobs=-1,
            )
            cv_results = {
                "mean_accuracy": round(float(np.mean(fold_scores)), 4),
                "std_accuracy": round(float(np.std(fold_scores)), 4),
                "fold_scores": [round(float(s), 4) for s in fold_scores],
            }
            console.print(
                f"   CV accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
            )
            tracker.log_cv_results(cv_results)
        else:
            console.print("   [yellow]CV skipped for transformer models.[/yellow]")

    # --- Train final model ---
    console.print("\n[bold]5. Training final model...[/bold]")
    t0 = time.time()
    model.fit(texts_train, labels_train, texts_val, labels_val)
    duration = time.time() - t0
    console.print(f"   Training completed in {duration:.1f}s")

    # --- Evaluate ---
    console.print("\n[bold]6. Evaluating on test set...[/bold]")
    y_pred = model.predict(texts_test)
    test_metrics = compute_metrics(labels_test, y_pred, classes)

    console.print(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
    console.print(f"   Macro F1:    {test_metrics['macro_f1']:.4f}")
    console.print(f"   Weighted F1: {test_metrics['weighted_f1']:.4f}")

    _print_per_class_table(test_metrics["per_class"])

    tracker.log_test_metrics(test_metrics)
    tracker.log_duration(duration)

    # --- Save plots ---
    plots_dir = tracker.experiment_dir / "plots"
    save_confusion_matrix(
        labels_test, y_pred, classes,
        output_path=plots_dir / "confusion_matrix.png",
        title=exp_name,
    )
    save_metrics_plot(
        test_metrics,
        output_path=plots_dir / "per_class_metrics.png",
        title=f"{exp_name} — Per-class metrics",
    )

    # --- Save model ---
    if config.get("tracking", {}).get("save_model", True):
        console.print("\n[bold]7. Saving model...[/bold]")
        model.save(tracker.experiment_dir / "model")
        console.print(f"   Saved to: {tracker.experiment_dir}")

    tracker.finalize()
    console.print(f"\n[bold green]Done! Results saved to: {tracker.experiment_dir}[/bold green]")

    return tracker.get_summary()


def _print_per_class_table(per_class: dict) -> None:
    table = Table(title="Per-class metrics", show_header=True)
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")

    for label, metrics in per_class.items():
        table.add_row(
            label,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            str(metrics['support']),
        )
    console.print(table)
