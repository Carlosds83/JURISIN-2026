# Projection helpers for embeddings visualizations.
"""Compute UMAP and t-SNE projections so the Viewer can plot 2D coordinates."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd

try:
    import umap  # type: ignore
except ImportError as error:  # pragma: no cover - optional dependency warning
    umap = None
    UMAP_IMPORT_ERROR = error
else:  # pragma: no cover - confirm success
    UMAP_IMPORT_ERROR = None

try:
    from sklearn.manifold import TSNE  # type: ignore
except ImportError as error:  # pragma: no cover
    TSNE = None
    TSNE_IMPORT_ERROR = error
else:  # pragma: no cover
    TSNE_IMPORT_ERROR = None

ProjectionName = Literal["umap", "tsne"]


def project_vectors(
    embeddings: np.ndarray,
    projection: ProjectionName,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project embeddings down to two dimensions with the requested algorithm.

    Args:
        embeddings (np.ndarray): Embedding matrix with shape (n_samples, n_dim).
        projection (Literal["umap","tsne"]): Projection algorithm to use.
        random_state (int): Seed for reproducibility.

    Returns:
        pandas.DataFrame: Columns `x` and `y` representing 2D coordinates.

    Viewer:
        Data is rendered inside the Embeddings tab (scatter/KDE/centroids).
    """

    if projection == "umap":
        if umap is None:
            raise RuntimeError(
                "umap-learn is required to compute UMAP projections. "
                f"Original error: {UMAP_IMPORT_ERROR}"
            )
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        )
        coords = reducer.fit_transform(embeddings)
        return pd.DataFrame(coords, columns=["x", "y"])

    if projection == "tsne":
        if TSNE is None:
            raise RuntimeError(
                "scikit-learn is required for t-SNE projections. "
                f"Original error: {TSNE_IMPORT_ERROR}"
            )
        max_perplexity = max(1, embeddings.shape[0] - 1)
        perplexity = min(30, max_perplexity)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = tsne.fit_transform(embeddings)
        return pd.DataFrame(coords, columns=["x", "y"])

    raise ValueError(f"Unsupported projection type: {projection}")
