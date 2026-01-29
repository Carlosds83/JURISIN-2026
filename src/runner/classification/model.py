# Modeling utilities using TF-IDF + Logistic Regression.
"""Create scikit-learn pipelines for binary and multiclass targets."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline(task_type: str) -> Pipeline:
    """Return a TF-IDF + LogisticRegression pipeline."""

    if task_type not in {"binary", "multiclass"}:
        raise ValueError(f"Unsupported task type: {task_type}")

    logistic_kwargs: Dict[str, object] = {"max_iter": 1000, "solver": "lbfgs"}

    classifier = LogisticRegression(**logistic_kwargs)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def fit_pipeline(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    """Train the pipeline on the training split."""

    pipeline.fit(train_df["text"], train_df["label"])
    return pipeline
