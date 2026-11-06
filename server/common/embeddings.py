"""CPU text embeddings shared by API routes (design §6.3.7)."""

from __future__ import annotations

from typing import Any

import numpy as np

TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
EMBEDDING_DIMENSIONS = 512


class ClipTextEncoder:
    """CPU-only multilingual CLIP text encoder for API queries."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME, model: Any | None = None) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu")
        self._model = model

    def encode(self, text: str) -> np.ndarray:
        embedding = np.asarray(self._model.encode(text, convert_to_numpy=True), dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIMENSIONS,):
            raise ValueError(f"CLIP text embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        length = np.linalg.norm(embedding)
        if length == 0:
            raise ValueError("CLIP text embedding must not be zero")
        return embedding / length
