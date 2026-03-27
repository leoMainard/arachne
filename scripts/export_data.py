"""Export labeled table data from PostgreSQL to a local parquet file.

Usage:
    python scripts/export_data.py --output data/tables.parquet
    python scripts/export_data.py --output data/tables.parquet --query "SELECT id, table_data, label FROM my_table"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from arachne.data.loader import load_from_postgresql, export_to_parquet

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export data from PostgreSQL to parquet.")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/tables.parquet"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--query", default=None, help="Custom SQL query.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_config = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password,
    }

    console.print(f"Connecting to PostgreSQL ({args.host}:{args.port}/{args.dbname})...")
    try:
        df = load_from_postgresql(db_config, query=args.query)
    except Exception as e:
        console.print(f"[red]Failed to load from PostgreSQL: {e}[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(df)} rows.")
    console.print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    export_to_parquet(df, args.output)
    console.print(f"[green]Saved to: {args.output}[/green]")


if __name__ == "__main__":
    main()
