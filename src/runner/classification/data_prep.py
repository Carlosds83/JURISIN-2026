# Data preparation utilities for TF-IDF classification.
"""Clean labels, attach metadata columns, and create stratified splits."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def build_base_frame(
    dataframe: pd.DataFrame,
    text_column: str,
    target_column: str,
    identifier_column: str | None,
    law_column: str | None,
) -> pd.DataFrame:
    """Return a sanitized dataframe with the required columns."""

    frame = pd.DataFrame(
        {
            "text": dataframe[text_column].astype(str),
            "label": dataframe[target_column],
        }
    )
    if identifier_column and identifier_column in dataframe.columns:
        frame["row_id"] = dataframe[identifier_column].astype(str)
    else:
        frame["row_id"] = dataframe.index.astype(str)

    if law_column and law_column in dataframe.columns:
        frame["law"] = dataframe[law_column].astype(str)
    else:
        frame["law"] = None

    frame = frame.replace({"": None}).dropna(subset=["text", "label"])
    frame["text"] = frame["text"].str.strip()
    frame = frame[frame["text"] != ""]
    frame["label"] = frame["label"].apply(lambda value: str(value).strip())
    frame = frame[frame["label"] != ""]
    return frame


def stratified_splits(
    frame: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    """Split the dataframe into train/val/test partitions."""

    if frame.empty:
        raise ValueError("Cannot split empty dataframe for classification.")

    stratify = frame["label"]
    train_df, temp_df = train_test_split(
        frame,
        test_size=1 - train_ratio,
        stratify=stratify,
        random_state=seed,
    )

    temp_ratio = test_ratio / (test_ratio + val_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=temp_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return {"train": train_df, "val": val_df, "test": test_df}
