"""Table preprocessing: matrix → text representation."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = ["batiment", "vehicule", "sinistre", "autre"]
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {i: label for i, label in enumerate(LABELS)}


def clean_cell(value) -> str:
    """Normalize a table cell to a clean string."""
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def table_to_text(
    table: list[list],
    header_rows: int = 1,
    header_weight: int = 3,
    max_content_cells: int = 200,
    max_length: Optional[int] = None,
) -> str:
    """Convert a table matrix to a weighted text representation.

    Headers are repeated `header_weight` times to increase their influence
    during vectorization.

    Args:
        table: 2D list of cell values.
        header_rows: number of rows treated as headers.
        header_weight: how many times to repeat header text.
        max_content_cells: max number of content cells to include.
        max_length: optional character limit for the output text.

    Returns:
        Single string representing the table.
    """
    if not table:
        return ""

    header_rows = min(header_rows, len(table))
    headers = table[:header_rows]
    content = table[header_rows:]

    header_cells = [
        clean_cell(cell)
        for row in headers
        for cell in row
        if clean_cell(cell)
    ]
    header_text = " | ".join(header_cells)

    content_cells = [
        clean_cell(cell)
        for row in content
        for cell in row
        if clean_cell(cell)
    ][:max_content_cells]
    content_text = " | ".join(content_cells)

    parts = [header_text] * header_weight
    if content_text:
        parts.append(content_text)

    text = " ".join(p for p in parts if p)

    if max_length:
        text = text[:max_length]

    return text


def tables_to_texts(tables: list[list[list]], preprocessing_config: dict) -> list[str]:
    """Convert a list of table matrices to text representations."""
    return [
        table_to_text(
            table,
            header_rows=preprocessing_config.get("header_rows", 1),
            header_weight=preprocessing_config.get("header_weight", 3),
            max_content_cells=preprocessing_config.get("max_content_cells", 200),
            max_length=preprocessing_config.get("max_length", None),
        )
        for table in tables
    ]


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, validation, and test sets."""
    stratify_col = df["label"] if stratify else None

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_seed,
    )

    # Val size relative to train_val
    relative_val_size = val_size / (1.0 - test_size)
    stratify_col_tv = df_train_val["label"] if stratify else None

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        stratify=stratify_col_tv,
        random_state=random_seed,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
