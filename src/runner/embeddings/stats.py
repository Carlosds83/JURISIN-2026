# Projection statistics for embeddings.
"""Compute centroid distances per rule so the Viewer can surface summary metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CentroidStats:
    """Summary describing how separated two groups are in projection space."""

    centroid_distance: float
    group_sizes: Dict[str, int]


def compute_centroid_stats(
    projection_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> Optional[CentroidStats]:
    """Compute Euclidean distance between group centroids when both exist.

    Args:
        projection_df (pandas.DataFrame): Must contain columns row_id, x, y.
        labels_df (pandas.DataFrame): Contains row_id, group_label, group_value.

    Returns:
        CentroidStats | None: None when a group lacks data or rows missing.

    Viewer:
        Numbers feed tooltips/indicators within the Embeddings tab.
    """

    merged = projection_df.merge(labels_df, on="row_id", how="inner")
    valid = merged.dropna(subset=["x", "y", "group_value"])
    if valid.empty:
        return None

    groups = valid["group_value"].unique()
    if len(groups) < 2:
        return None

    centroids = {}
    sizes = {}
    for value in groups:
        subset = valid[valid["group_value"] == value]
        if subset.empty:
            continue
        centroids[value] = np.array([subset["x"].mean(), subset["y"].mean()])
        sizes[str(value)] = int(len(subset))

    if len(centroids) < 2:
        return None

    group_keys = sorted(centroids.keys())
    distance = float(np.linalg.norm(centroids[group_keys[0]] - centroids[group_keys[1]]))
    return CentroidStats(centroid_distance=distance, group_sizes=sizes)
