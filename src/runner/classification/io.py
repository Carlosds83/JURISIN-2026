# IO helpers for saving classification artifacts.
"""Persist experiment outputs under runs/<run>/classification/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def ensure_classification_dirs(run_dir: Path) -> Path:
    """Create the classification folder and return its path."""

    classification_dir = run_dir / "classification"
    (classification_dir / "experiments").mkdir(parents=True, exist_ok=True)
    return classification_dir


def save_experiment_outputs(
    experiment_dir: Path,
    predictions: pd.DataFrame,
    metrics: Dict[str, object],
    confusion_matrix: pd.DataFrame,
    class_labels: Dict[str, str],
) -> None:
    """Write the per-experiment files needed by the Viewer."""

    experiment_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(experiment_dir / "predictions.parquet", index=False)
    confusion_matrix.to_csv(experiment_dir / "confusion_matrix.csv")
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (experiment_dir / "class_labels.json").write_text(
        json.dumps(class_labels, indent=2), encoding="utf-8"
    )


def save_results_table(
    classification_dir: Path,
    rows: List[Dict[str, object]],
) -> None:
    """Aggregate experiment metadata."""

    table = pd.DataFrame(rows)
    table.to_parquet(classification_dir / "results_table.parquet", index=False)
