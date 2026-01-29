# Column resolution helpers for the TF-IDF classification pipeline.
"""Identify text, target, identifier, and Law columns with sensible fallbacks."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

TEXT_CANDIDATES = [
    "text",
    "Text",
    "article_text",
    "Article",
    "content",
    "body",
    "texto",
]

LAW_CANDIDATES = ["Law", "law", "ley", "statute_name"]
ID_CANDIDATES = [
    "article_id",
    "articleId",
    "article",
    "id",
    "identifier",
    "articulo",
    "section_id",
]


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_column(
    dataframe: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    normalized = {_normalize(column): column for column in dataframe.columns}
    for candidate in candidates:
        key = _normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None


def select_text_column(dataframe: pd.DataFrame) -> str:
    """Return the best text column based on predefined candidates or heuristics."""

    column = _find_column(dataframe, TEXT_CANDIDATES)
    if column:
        return column

    string_columns = [
        column for column in dataframe.columns if dataframe[column].dtype == object
    ]
    if not string_columns:
        raise ValueError("No text-like column found in dataset.")

    avg_lengths: Dict[str, float] = {}
    for column in string_columns:
        lengths = dataframe[column].dropna().astype(str).str.len()
        avg_lengths[column] = lengths.mean() if not lengths.empty else 0.0
    return max(avg_lengths, key=avg_lengths.get)


def select_target_column(
    dataframe: pd.DataFrame, aliases: List[str], target_name: str
) -> str:
    """Find the column matching a target's alias list."""

    column = _find_column(dataframe, aliases)
    if column:
        return column

    raise ValueError(
        f"Target '{target_name}' not found. Available columns: {list(dataframe.columns)}"
    )


def select_identifier_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Return an identifier column (article id) if present."""

    return _find_column(dataframe, ID_CANDIDATES)


def select_law_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Return the Law column if present."""

    return _find_column(dataframe, LAW_CANDIDATES)
