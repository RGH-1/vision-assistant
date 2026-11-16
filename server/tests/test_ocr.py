import numpy as np
import pytest

from server.ingest.ocr import FrameOcr, OcrConfig, _prepare_sign_crop
from server.ingest.store import StoredDetection


class FakeReader:
    def __init__(self, results: list[list[tuple[object, str, float]]]) -> None:
        self.results = results
        self.images: list[np.ndarray] = []

    def readtext(self, image: np.ndarray, **kwargs: object) -> list[tuple[object, str, float]]:
        assert kwargs == {"detail": 1, "paragraph": False}
        self.images.append(image)
        return self.results.pop(0)


def config() -> OcrConfig:
    return OcrConfig(
        full_frame_max_side=1280,
        crop_scale=2,
        clahe_clip_limit=2,
        clahe_tile_size=8,
        denoise_h=7,
        confidence_min=0.4,
        min_text_length=2,
        merge_iou=0.5,
    )


def box(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def test_ocr_reads_full_frame_and_sign_crop_then_keeps_the_best_overlap() -> None:
    reader = FakeReader(
        [
            [(box(20, 20, 100, 80), "Room 204", 0.6), (box(0, 0, 10, 10), "X", 0.99)],
            [(box(0, 0, 160, 120), "Room 204", 0.9), (box(0, 0, 20, 10), "no", 0.2)],
        ]
    )
    ocr = FrameOcr(config(), reader_factory=lambda *_, **__: reader)
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    sign = StoredDetection("sign", 0.8, (0.1, 0.2, 0.5, 0.8), 0.24, "left")
    timings: dict[str, float] = {}

    findings = ocr.read(frame, (sign,), timings)

    assert len(findings) == 1
    assert findings[0].text == "Room 204"
    assert findings[0].confidence == 0.9
    assert findings[0].from_crop is True
    assert findings[0].bbox == pytest.approx((0.1, 0.2, 0.5, 0.8))
    assert [image.shape[:2] for image in reader.images] == [(100, 200), (120, 160)]
    assert timings["ocr"] >= 0


def test_ocr_limits_the_full_frame_to_1280_pixels() -> None:
    reader = FakeReader([[]])
    ocr = FrameOcr(config(), reader_factory=lambda *_, **__: reader)

    assert ocr.read(np.zeros((1000, 2000, 3), dtype=np.uint8), (), {}) == ()
    assert reader.images[0].shape[:2] == (640, 1280)


def test_ocr_loads_one_gpu_reader_for_arabic_and_english() -> None:
    reader = FakeReader([])
    calls: list[tuple[list[str], bool]] = []

    def factory(languages: list[str], *, gpu: bool) -> FakeReader:
        calls.append((languages, gpu))
        return reader

    FrameOcr(config(), reader_factory=factory)

    assert calls == [(["ar", "en"], True)]


def test_sign_crop_uses_the_specified_upscale_clahe_and_denoise() -> None:
    crop = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)

    processed = _prepare_sign_crop(crop, config())

    assert processed.shape == (40, 60, 3)
    assert processed.dtype == np.uint8


def test_ocr_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="confidence"):
        FrameOcr(
            OcrConfig(1280, 2, 2, 8, 7, 1.1, 2, 0.5),
            reader_factory=lambda *_, **__: FakeReader([]),
        )
