"""Export des données labellisées depuis PostgreSQL vers un fichier parquet local.

Utilisation :
    python scripts/export_data.py --dbname ma_base --user mon_utilisateur --output data/tables.parquet
    python scripts/export_data.py --dbname ma_base --user mon_utilisateur --requete "SELECT id, table_data, label FROM ma_table"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from arachne.data.loader import ChargeurDonnees

console = Console()


def _analyser_arguments() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Retours:
        Namespace avec les arguments parsés.
    """
    parser = argparse.ArgumentParser(
        description="Exporte les données depuis PostgreSQL vers un fichier parquet."
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("data/tables.parquet"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--requete", default=None,
        help="Requête SQL personnalisée (doit retourner les colonnes table_data et label)."
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du script d'export."""
    args = _analyser_arguments()

    config_db = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
    }

    console.print(f"Connexion à PostgreSQL ({args.host}:{args.port}/{args.dbname})...")
    try:
        df = ChargeurDonnees.depuis_postgresql(config_db, requete=args.requete)
    except Exception as erreur:
        console.print(f"[red]Erreur lors du chargement depuis PostgreSQL : {erreur}[/red]")
        sys.exit(1)

    console.print(f"{len(df)} lignes chargées.")
    console.print(f"Distribution des classes :\n{df['label'].value_counts().to_string()}")

    ChargeurDonnees.exporter_parquet(df, args.output)
    console.print(f"[green]Fichier sauvegardé : {args.output}[/green]")


if __name__ == "__main__":
    main()
