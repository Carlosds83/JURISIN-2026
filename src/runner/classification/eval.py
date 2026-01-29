# Evaluation helpers for scikit-learn classification outputs.
"""Generate predictions, metrics, and confusion matrices for persistence."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def predict_with_probabilities(pipeline, frame: pd.DataFrame) -> Dict[str, object]:
    """Run inference and return raw outputs required downstream."""

    y_true = frame["label"].tolist()
    y_pred = pipeline.predict(frame["text"])
    probabilities = None
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(frame["text"])
    classes = classifier.classes_
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "classes": classes,
    }


def build_prediction_frame(
    outputs: Dict[str, object],
    frame: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """Create a dataframe including y_true, y_pred, row_id, law, and scores."""

    df = pd.DataFrame(
        {
            "row_id": frame["row_id"].tolist(),
            "law": frame["law"].tolist(),
            "y_true": outputs["y_true"],
            "y_pred": outputs["y_pred"],
        }
    )

    probabilities = outputs["probabilities"]
    classes = outputs["classes"]
    if probabilities is not None:
        if task_type == "binary":
            positive_index = 1 if len(classes) > 1 else 0
            df["y_score"] = probabilities[:, positive_index]
        else:
            for idx, label in enumerate(classes):
                df[f"score_{label}"] = probabilities[:, idx]
    return df


def compute_metrics(
    outputs: Dict[str, object],
    task_type: str,
) -> Dict[str, object]:
    """Return per-class metrics and aggregate stats."""

    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    labels = outputs["classes"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for idx, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(np.average(f1, weights=support)),
        "per_class": per_class,
    }

    probabilities = outputs["probabilities"]
    if probabilities is not None and task_type == "binary":
        try:
            positive_index = 1 if len(labels) > 1 else 0
            y_score = probabilities[:, positive_index]
            # Attempt numeric conversion for ROC/PR AUC
            numeric_true = pd.Series(y_true).astype(float)
            metrics["roc_auc"] = float(roc_auc_score(numeric_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(numeric_true, y_score))
        except Exception:
            metrics.setdefault(
                "notes", []
            ).append("Could not compute ROC/PR AUC due to non-numeric labels.")

    return metrics


def build_confusion_matrix(outputs: Dict[str, object]) -> pd.DataFrame:
    """Return confusion matrix as a pandas dataframe."""

    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    labels = outputs["classes"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def class_labels_mapping(frame: pd.DataFrame) -> Dict[str, str]:
    """Return a raw-label -> display-label mapping."""

    labels = sorted(frame["label"].unique().tolist())
    return {str(label): str(label) for label in labels}
