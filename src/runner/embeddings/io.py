# Disk IO helpers for embeddings artifacts.
"""Centralize file naming conventions for embeddings, projections, groups, and stats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .rules import RuleResult
from .stats import CentroidStats


@dataclass
class EmbeddingPaths:
    """Convenience bundle containing every folder used by the embeddings pipeline."""

    root: Path
    vectors: Path
    projections: Path
    groups: Path
    stats: Path
    manifest: Path = field(init=False)
    rules_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.manifest = self.root / "embeddings_manifest.json"
        self.rules_path = self.groups / "rules.json"


def prepare_directories(run_dir: Path) -> EmbeddingPaths:
    """Create the embeddings folder structure under the run directory.

    Args:
        run_dir (Path): Base path for the active run.

    Returns:
        EmbeddingPaths: Convenience object with every resolved folder.

    Viewer:
        Embeddings tab expects artifacts under runs/<run>/embeddings/.
    """

    embeddings_root = run_dir / "embeddings"
    vectors_dir = embeddings_root / "embeddings"
    projections_dir = embeddings_root / "projections"
    groups_dir = embeddings_root / "groups"
    stats_dir = embeddings_root / "stats"
    for path in (vectors_dir, projections_dir, groups_dir, stats_dir):
        path.mkdir(parents=True, exist_ok=True)
    return EmbeddingPaths(
        root=embeddings_root,
        vectors=vectors_dir,
        projections=projections_dir,
        groups=groups_dir,
        stats=stats_dir,
    )


def save_embeddings_npz(
    output_path: Path,
    embeddings: np.ndarray,
    row_ids: Iterable[str],
) -> Path:
    """Persist embeddings and aligned row identifiers to a compressed NPZ file.

    Args:
        output_path (Path): Destination NPZ file path.
        embeddings (np.ndarray): Embedding matrix with shape (n_rows, n_dim).
        row_ids (Iterable[str]): Identifiers aligned with the embedding rows.

    Returns:
        Path: Path to the saved NPZ artifact.

    Viewer:
        Vectors feed the Embeddings inspector and future offline analysis.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        row_ids=np.array([str(value) for value in row_ids]),
    )
    return output_path


def save_projection_parquet(
    output_path: Path,
    row_ids: Iterable[str],
    projection_df: pd.DataFrame,
) -> pd.DataFrame:
    """Persist a projection DataFrame with row identifiers and return it.

    Args:
        output_path (Path): Destination Parquet file path.
        row_ids (Iterable[str]): Row identifiers preserving dataset order.
        projection_df (pandas.DataFrame): DataFrame with columns x and y.

    Returns:
        pandas.DataFrame: Projection frame augmented with row_id column.

    Viewer:
        Embeddings tab loads these Parquet files to render scatter/KDE plots.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = projection_df.copy()
    data.insert(0, "row_id", [str(value) for value in row_ids])
    data.to_parquet(output_path, index=False)
    return data


def save_group_labels(paths: EmbeddingPaths, results: List[RuleResult]) -> None:
    """Write per-rule labels and the aggregate description file.

    Args:
        paths (EmbeddingPaths): Folder bundle returned by `prepare_directories`.
        results (list[RuleResult]): Rule payloads including per-row labels.

    Returns:
        None

    Viewer:
        Embeddings tab reads rules.json plus each labels_<rule>.parquet to color points.
    """

    rule_metadata: List[Dict[str, object]] = []
    for result in results:
        labels_path = paths.groups / f"labels_{result.definition.rule_id}.parquet"
        result.labels.to_parquet(labels_path, index=False)
        rule_metadata.append(
            {
                "rule_id": result.definition.rule_id,
                "description": result.definition.description,
                "column": result.definition.column,
                "positive_label": result.definition.positive_label,
                "negative_label": result.definition.negative_label,
                "labels_path": str(labels_path.relative_to(paths.root)),
            }
        )

    paths.rules_path.write_text(
        json.dumps(rule_metadata, indent=2),
        encoding="utf-8",
    )


def save_stats_json(
    paths: EmbeddingPaths,
    embedding_id: str,
    projection_name: str,
    rule_id: str,
    stats: CentroidStats,
) -> Path:
    """Write centroid statistics to a JSON payload.

    Args:
        paths (EmbeddingPaths): Folder bundle produced by `prepare_directories`.
        embedding_id (str): Identifier derived from the model name.
        projection_name (str): Projection key ("umap" or "tsne").
        rule_id (str): Identifier describing the comparison rule.
        stats (CentroidStats): Distance plus group size metadata.

    Returns:
        Path: JSON artifact path for downstream processing.

    Viewer:
        Enables lightweight summaries/tooltips for embeddings separation.
    """

    payload = {
        "embedding_id": embedding_id,
        "projection": projection_name,
        "rule_id": rule_id,
        "centroid_distance": stats.centroid_distance,
        "group_sizes": stats.group_sizes,
    }
    output_path = paths.stats / f"stats_{embedding_id}_{projection_name}_{rule_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_manifest(
    paths: EmbeddingPaths,
    run_dir: Path,
    *,  # enforce keyword usage
    text_column: str,
    row_id_column: str,
    models: List[Dict[str, object]],
    projection_names: List[str],
    rule_ids: List[str],
) -> None:
    """Persist the embeddings manifest consumed by the Viewer.

    Args:
        paths (EmbeddingPaths): Folder bundle returned by `prepare_directories`.
        run_dir (Path): Base directory for the current run.
        text_column (str): Original column used to compute embeddings.
        row_id_column (str): Column (or __index__) representing row identifiers.
        models (list[dict]): Metadata for each embedding model/projection set.
        projection_names (list[str]): Names of projection types produced (e.g., UMAP).
        rule_ids (list[str]): Rule identifiers available for coloring/filtering.

    Returns:
        None

    Viewer:
        Embeddings tab parses this file to discover available models/projections.
    """

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text_column": text_column,
        "row_id_column": row_id_column,
        "embedding_models": [record["embedding_id"] for record in models],
        "projections_available": projection_names,
        "rules": rule_ids,
        "folders": {
            "embeddings": str(paths.vectors.relative_to(run_dir)),
            "projections": str(paths.projections.relative_to(run_dir)),
            "groups": str(paths.groups.relative_to(run_dir)),
            "stats": str(paths.stats.relative_to(run_dir)),
        },
        "models": models,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
