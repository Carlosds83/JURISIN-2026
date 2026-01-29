# Classification orchestrator using TF-IDF + Logistic Regression.
"""Runs per-target experiments and saves Viewer-compatible artifacts."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.exceptions import NotFittedError

from . import column_map
from .data_prep import build_base_frame, stratified_splits
from .model import build_pipeline, fit_pipeline
from .eval import (
    build_confusion_matrix,
    build_prediction_frame,
    class_labels_mapping,
    compute_metrics,
    predict_with_probabilities,
)
from .io import ensure_classification_dirs, save_experiment_outputs, save_results_table

RANDOM_SEED = 42
MODEL_NAME = "tfidf_logreg"


@dataclass
class TargetSpec:
    name: str
    aliases: List[str]
    task_type: str  # "binary" or "multiclass"


TARGET_SPECS = [
    TargetSpec("Relevance", ["relevance", "Relevance"], "binary"),
    TargetSpec("Completeness", ["completeness", "Completeness"], "binary"),
    TargetSpec("Differential Regime", ["differential_regime", "Differential Regime", "regimen diferencial"], "binary"),
    TargetSpec("Discretionality", ["discretionality", "Discretionality"], "binary"),
    TargetSpec("Interpretability", ["interpretability", "Interpretability"], "multiclass"),
]


def run_classification_pipeline(dataset: pd.DataFrame, run_dir: Path) -> None:
    """Execute the TF-IDF classification experiments."""

    try:
        import sklearn  # noqa: F401
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("scikit-learn is required for the classification pipeline.") from error

    text_column = column_map.select_text_column(dataset)
    identifier_column = column_map.select_identifier_column(dataset)
    law_column = column_map.select_law_column(dataset)
    classification_dir = ensure_classification_dirs(run_dir)
    summary_rows: List[Dict[str, object]] = []
    generated_artifacts: List[Path] = []

    for spec in TARGET_SPECS:
        try:
            target_column = column_map.select_target_column(
                dataset, spec.aliases, spec.name
            )
        except ValueError as error:
            print(f"[Runner] Classification skipped for {spec.name}: {error}")
            continue

        base_frame = build_base_frame(
            dataframe=dataset,
            text_column=text_column,
            target_column=target_column,
            identifier_column=identifier_column,
            law_column=law_column,
        )

        if base_frame["label"].nunique() < 2:
            print(f"[Runner] Classification skipped for {spec.name}: not enough classes.")
            continue

        try:
            splits = stratified_splits(base_frame, seed=RANDOM_SEED)
        except ValueError as error:
            print(f"[Runner] Classification skipped for {spec.name}: {error}")
            continue

        pipeline = build_pipeline(spec.task_type)
        fit_pipeline(pipeline, splits["train"])

        try:
            outputs = predict_with_probabilities(pipeline, splits["test"])
        except NotFittedError as error:
            print(f"[Runner] Classification failed for {spec.name}: {error}")
            continue

        predictions_df = build_prediction_frame(outputs, splits["test"], spec.task_type)
        metrics = compute_metrics(outputs, spec.task_type)
        confusion = build_confusion_matrix(outputs)
        class_labels = class_labels_mapping(splits["train"])

        experiment_id = build_experiment_id(spec.name)
        experiment_dir = classification_dir / "experiments" / experiment_id
        save_experiment_outputs(
            experiment_dir,
            predictions_df,
            metrics,
            confusion,
            class_labels,
        )
        generated_artifacts.extend(
            [
                experiment_dir / "predictions.parquet",
                experiment_dir / "metrics.json",
                experiment_dir / "confusion_matrix.csv",
                experiment_dir / "class_labels.json",
            ]
        )

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": MODEL_NAME,
                "target_name": spec.name,
                "task_type": spec.task_type,
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "seed": RANDOM_SEED,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "timestamp": int(time.time()),
                "predictions_path": str(
                    Path("classification") / "experiments" / experiment_id / "predictions.parquet"
                ),
                "metrics_path": str(
                    Path("classification") / "experiments" / experiment_id / "metrics.json"
                ),
                "confusion_path": str(
                    Path("classification") / "experiments" / experiment_id / "confusion_matrix.csv"
                ),
            }
        )

        print(f"[Runner] Classification completed for {spec.name} -> {experiment_id}")

    if summary_rows:
        save_results_table(classification_dir, summary_rows)
        print("[Runner] Classification summary saved to results_table.parquet")
        manifest_path = classification_dir / "classification_manifest.json"
        manifest_payload = {
            "experiment_ids": [row["experiment_id"] for row in summary_rows],
            "targets": sorted({row["target_name"] for row in summary_rows}),
            "model_name": MODEL_NAME,
            "created_at": int(time.time()),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    else:
        print("[Runner] No classification experiments were executed.")

    if generated_artifacts:
        print("[Runner] Generated artifacts:")
        for artifact in generated_artifacts:
            print(f" - {artifact}")


def build_experiment_id(target_name: str) -> str:
    """Return a unique identifier using the target name and timestamp."""

    short_id = uuid.uuid4().hex[:6]
    timestamp = int(time.time())
    slug = target_name.lower().replace(" ", "")[:8]
    return f"{slug}_{RANDOM_SEED}_{short_id}_{timestamp % 100000}"
