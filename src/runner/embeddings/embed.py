# Sentence-level embedding utilities for the Runner.
"""Wrap sentence-transformers so the Runner generates text embeddings once per model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as error:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None  # type: ignore[assignment]
    SENTENCE_TRANSFORMER_IMPORT_ERROR = error
else:  # pragma: no cover - document success
    SENTENCE_TRANSFORMER_IMPORT_ERROR = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Container returned by `generate_embeddings` with vectors and metadata."""

    model_name: str
    vectors: np.ndarray


def _ensure_model(model_name: str) -> SentenceTransformer:
    """Instantiate the requested sentence-transformer model or raise a clear error.

    Viewer: not applicable (Runner only).
    """

    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for the embeddings pipeline. "
            f"Original error: {SENTENCE_TRANSFORMER_IMPORT_ERROR}"
        )
    try:
        return SentenceTransformer(model_name)
    except Exception as model_error:  # pragma: no cover - network/cache issues
        raise RuntimeError(
            f"Unable to load embedding model '{model_name}'. "
            "Verify the model name or install the required weights."
        ) from model_error


def _prepare_texts(texts: Sequence[str]) -> list[str]:
    """Sanitize text entries so the encoder receives clean unicode strings."""

    prepared: list[str] = []
    for entry in texts:
        if entry is None:
            prepared.append("")
        else:
            prepared.append(str(entry))
    return prepared


def generate_embeddings(
    texts: Iterable[str],
    model_name: str,
    batch_size: int = 32,
) -> EmbeddingResult:
    """Encode every document using the requested sentence-transformer model.

    Args:
        texts (Iterable[str]): Raw text corpus aligned with the dataset rows.
        model_name (str): HuggingFace identifier for the embedding model.
        batch_size (int): Mini-batch size to feed into SentenceTransformer.encode.

    Returns:
        EmbeddingResult: Contains the model name and resulting numpy array (N x D).

    Viewer:
        Outputs feed the Embeddings tab after Runner projection steps materialize.
    """

    text_list = _prepare_texts(list(texts))
    model = _ensure_model(model_name)
    vectors = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return EmbeddingResult(model_name=model_name, vectors=vectors.astype("float32"))
