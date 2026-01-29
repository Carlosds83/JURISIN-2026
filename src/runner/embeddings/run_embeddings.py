# Embeddings pipeline orchestrator.
"""Load text, compute embeddings, build projections, and persist Viewer-ready artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .embed import EmbeddingResult, generate_embeddings
from .io import (
    EmbeddingPaths,
    prepare_directories,
    save_embeddings_npz,
    save_group_labels,
    save_projection_parquet,
    save_stats_json,
    write_manifest,
)
from .project import ProjectionName, project_vectors
from .rules import RuleResult, generate_rules
from .stats import compute_centroid_stats

TEXT_CANDIDATES = [
    "Spanish text",
    "spanish_text",
    "text",
    "Text",
    "article_text",
    "content",
]
ROW_ID_CANDIDATES = [
    "article_id",
    "articleId",
    "articulo",
    "id",
    "identifier",
    "row_id",
]

MODEL_CATALOG = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
PROJECTIONS: List[ProjectionName] = ["umap", "tsne"]


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _select_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize(column): column for column in dataframe.columns}
    for candidate in candidates:
        token = _normalize(candidate)
        if token in normalized:
            return normalized[token]
    return None


def _select_text_column(dataframe: pd.DataFrame) -> str:
    column = _select_column(dataframe, TEXT_CANDIDATES)
    if column:
        return column

    object_columns = [
        column for column in dataframe.columns if dataframe[column].dtype == object
    ]
    if not object_columns:
        raise RuntimeError("Embeddings pipeline requires at least one text column.")

    avg_lengths = {}
    for column in object_columns:
        lengths = dataframe[column].dropna().astype(str).str.len()
        avg_lengths[column] = lengths.mean() if not lengths.empty else 0.0
    return max(avg_lengths, key=avg_lengths.get)


def _select_row_ids(dataframe: pd.DataFrame) -> Tuple[pd.Series, str]:
    column = _select_column(dataframe, ROW_ID_CANDIDATES)
    if column:
        return dataframe[column].astype(str), column

    return dataframe.index.to_series().astype(str), "__index__"


def _slugify_model(model_name: str) -> str:
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:10]
    return f"emb_{digest}"


def run_embeddings_pipeline(dataset: pd.DataFrame, run_dir: Path) -> None:
    """Entry point invoked from run_all.py after the dataset is persisted.

    Args:
        dataset (pandas.DataFrame): Base data from the Runner.
        run_dir (Path): Run-specific directory (runs/run_<timestamp>_<id>).

    Viewer:
        Embeddings tab consumes the artifacts created here.
    """

    print("[Runner][Embeddings] Starting embeddings pipeline...")
    paths = prepare_directories(run_dir)

    try:
        text_column = _select_text_column(dataset)
    except RuntimeError as selection_error:
        raise RuntimeError(f"Embeddings pipeline aborted: {selection_error}") from selection_error
    row_ids, row_id_column = _select_row_ids(dataset)
    text_series = dataset[text_column].fillna("").astype(str)

    rules: List[RuleResult] = generate_rules(dataset, row_ids)
    if not rules:
        print("[Runner][Embeddings] No compatible triggers found; skipping embeddings module.")
        return

    save_group_labels(paths, rules)

    model_records: List[Dict[str, object]] = []
    for model_name in MODEL_CATALOG:
        print(f"[Runner][Embeddings] Encoding texts with {model_name} ...")
        embeddings: EmbeddingResult = generate_embeddings(text_series, model_name)
        embedding_id = _slugify_model(model_name)
        vector_path = paths.vectors / f"{embedding_id}.npz"
        save_embeddings_npz(vector_path, embeddings.vectors, row_ids)

        projection_paths: Dict[str, Path] = {}
        for projection in PROJECTIONS:
            print(f"[Runner][Embeddings] Computing {projection} projection...")
            coords = project_vectors(embeddings.vectors, projection)
            output_path = paths.projections / f"{embedding_id}_{projection}.parquet"
            projection_with_ids = save_projection_parquet(
                output_path, row_ids, coords
            )
            projection_paths[projection] = output_path

            for rule in rules:
                stats = compute_centroid_stats(projection_with_ids, rule.labels)
                if stats:
                    save_stats_json(paths, embedding_id, projection, rule.definition.rule_id, stats)

        model_records.append(
            {
                "embedding_id": embedding_id,
                "model_name": model_name,
                "vectors_path": str(vector_path.relative_to(paths.root)),
                "projections": {
                    name: str(path.relative_to(paths.root))
                    for name, path in projection_paths.items()
                },
            }
        )

    write_manifest(
        paths,
        run_dir=run_dir,
        text_column=text_column,
        row_id_column=row_id_column,
        models=model_records,
        projection_names=PROJECTIONS,
        rule_ids=[rule.definition.rule_id for rule in rules],
    )
    print(f"[Runner] Embeddings manifest written to {paths.manifest}")
    print("[Runner][Embeddings] Finished successfully.")
