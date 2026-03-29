"""Point d'entrée CLI pour l'entraînement des expériences.

Utilisation :
    python scripts/train.py --config configs/experiments/tfidf_logistic.yaml
    python scripts/train.py --config configs/experiments/camembert.yaml --source-donnees local
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from arachne.config import charger_config
from arachne.training.trainer import executer_experience

console = Console()


def _analyser_arguments() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Retours:
        Namespace avec les arguments parsés.
    """
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de classification de tableaux.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/train.py --config configs/experiments/tfidf_logistic.yaml
  python scripts/train.py --config configs/experiments/camembert.yaml --source-donnees local
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Chemin vers le fichier YAML de configuration de l'expérience.",
    )
    parser.add_argument(
        "--source-donnees",
        choices=["postgresql", "local"],
        help="Surcharge la source de données définie dans la configuration.",
    )
    parser.add_argument(
        "--sans-sauvegarde",
        action="store_true",
        help="Ne sauvegarde pas le modèle (métriques et graphiques uniquement).",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du script d'entraînement."""
    args = _analyser_arguments()

    if not args.config.exists():
        console.print(f"[red]Fichier de configuration introuvable : {args.config}[/red]")
        sys.exit(1)

    config = charger_config(args.config)

    if args.source_donnees:
        config["data"]["source"] = args.source_donnees
    if args.sans_sauvegarde:
        config.setdefault("tracking", {})["save_model"] = False

    try:
        resume = executer_experience(config)
        accuracy = resume.get("test_metrics", {}).get("accuracy", "N/A")
        console.print(f"\n[bold]Accuracy test : {accuracy}[/bold]")
    except Exception:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
