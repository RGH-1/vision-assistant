import numpy as np
import pytest

from server.common.embeddings import EMBEDDING_DIMENSIONS, ClipTextEncoder
from server.ingest.embeddings import ClipImageEncoder


class FakeModel:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding

    def encode(self, *_: object, **__: object) -> np.ndarray:
        return self.embedding


def test_image_encoder_normalizes_vector_and_records_timing() -> None:
    encoder = ClipImageEncoder(FakeModel(np.full(EMBEDDING_DIMENSIONS, 2, dtype=np.float32)))
    timings_ms: dict[str, float] = {}

    embedding = encoder.encode(np.zeros((32, 32, 3), dtype=np.uint8), timings_ms)

    assert embedding.shape == (EMBEDDING_DIMENSIONS,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert timings_ms["clip"] >= 0


def test_text_encoder_normalizes_vector() -> None:
    encoder = ClipTextEncoder(model=FakeModel(np.full(EMBEDDING_DIMENSIONS, 3, dtype=np.float32)))

    embedding = encoder.encode("stairs")

    assert embedding.shape == (EMBEDDING_DIMENSIONS,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


@pytest.mark.parametrize("encoder", [ClipImageEncoder, ClipTextEncoder])
def test_encoders_reject_non_512_vectors(
    encoder: type[ClipImageEncoder] | type[ClipTextEncoder],
) -> None:
    model = FakeModel(np.ones(EMBEDDING_DIMENSIONS - 1, dtype=np.float32))
    if encoder is ClipImageEncoder:
        with pytest.raises(ValueError, match="512"):
            encoder(model).encode(np.zeros((32, 32, 3), dtype=np.uint8), {})
    else:
        with pytest.raises(ValueError, match="512"):
            encoder(model=model).encode("stairs")
