"""CLIP image embeddings loaded only by the ingest thread (design §6.3.7)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from server.common.embeddings import EMBEDDING_DIMENSIONS

IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


class ClipImageEncoder:
    """Encode already blurred and enhanced BGR frames with the ingest GPU model."""

    def __init__(self, model: Any | None = None) -> None:
        if model is None:
            # This constructor is called from the ingest thread, never a web route (§5.3).
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(IMAGE_MODEL_NAME, device="cuda")
        self._model = model

    def encode(self, frame: np.ndarray, timings_ms: dict[str, float]) -> np.ndarray:
        """Return a unit-length 512-d vector and record the CLIP encode time."""
        started_at = perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embedding = np.asarray(self._model.encode(image, convert_to_numpy=True), dtype=np.float32)
        timings_ms["clip"] = (perf_counter() - started_at) * 1000
        if embedding.shape != (EMBEDDING_DIMENSIONS,):
            raise ValueError(f"CLIP image embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        length = np.linalg.norm(embedding)
        if length == 0:
            raise ValueError("CLIP image embedding must not be zero")
        return embedding / length
