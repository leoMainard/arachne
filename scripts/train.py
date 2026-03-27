"""CLI entry point for training experiments.

Usage:
    python scripts/train.py --config configs/experiments/tfidf_logistic.yaml
    python scripts/train.py --config configs/experiments/tfidf_logistic.yaml --data-source local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from arachne.config import load_config
from arachne.training.trainer import run_experiment

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a table classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config configs/experiments/tfidf_logistic.yaml
  python scripts/train.py --config configs/experiments/camembert.yaml --data-source local
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--data-source",
        choices=["postgresql", "local"],
        help="Override the data source from config.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the model (only metrics and plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    config = load_config(args.config)

    # CLI overrides
    if args.data_source:
        config["data"]["source"] = args.data_source
    if args.no_save:
        config.setdefault("tracking", {})["save_model"] = False

    try:
        summary = run_experiment(config)
        acc = summary.get("test_metrics", {}).get("accuracy", "N/A")
        console.print(f"\n[bold]Test accuracy: {acc}[/bold]")
    except Exception as e:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
