"""Data loading from PostgreSQL or local files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"table_data", "label"}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize column names."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def _parse_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure table_data column contains parsed list[list[str]] objects."""
    def parse(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    df = df.copy()
    df["table_data"] = df["table_data"].apply(parse)
    return df


def load_from_postgresql(db_config: dict, query: str | None = None) -> pd.DataFrame:
    """Load data from PostgreSQL.

    Args:
        db_config: dict with keys: host, port, dbname, user, password
        query: optional custom SQL query (must return table_data, label columns)
    """
    try:
        import psycopg2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "psycopg2 is required for PostgreSQL. Install with: pip install psycopg2-binary"
        ) from e

    if query is None:
        query = "SELECT id, table_data, label FROM tables"

    conn = psycopg2.connect(**db_config)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    df = _ensure_columns(df)
    df = _parse_table_data(df)
    return df


def load_from_local(path: str | Path) -> pd.DataFrame:
    """Load data from a local parquet, CSV, or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use parquet, csv, or json.")

    df = _ensure_columns(df)
    df = _parse_table_data(df)
    return df


def load_data(config: dict) -> pd.DataFrame:
    """Load data according to config."""
    source = config["data"]["source"]

    if source == "postgresql":
        db_config = config["data"].get("postgresql", {})
        query = config["data"].get("query", None)
        return load_from_postgresql(db_config, query)
    elif source == "local":
        path = config["data"]["local_path"]
        return load_from_local(path)
    else:
        raise ValueError(f"Unknown data source '{source}'. Use 'postgresql' or 'local'.")


def export_to_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save DataFrame to parquet for local caching."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize table_data as JSON strings for parquet compatibility
    df_save = df.copy()
    df_save["table_data"] = df_save["table_data"].apply(json.dumps)
    df_save.to_parquet(output_path, index=False)
