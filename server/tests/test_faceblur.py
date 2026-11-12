from pathlib import Path

import cv2
import numpy as np
import pytest

from server.ingest.faceblur import FaceBlurConfig, FaceBlurrer, _expanded_box


class FakeYuNet:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        self.input_sizes.append(input_size)

    def detect(self, _: np.ndarray) -> tuple[None, np.ndarray | None]:
        return None, self.faces


def config(tmp_path: Path) -> FaceBlurConfig:
    return FaceBlurConfig(
        model_path=tmp_path / "yunet.onnx",
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000,
        box_margin=0.1,
        blur_kernel_ratio=0.5,
        blur_kernel_min=31,
    )


def test_blur_applies_a_strong_blur_to_each_detected_face_and_records_timing(
    tmp_path: Path,
) -> None:
    detector = FakeYuNet(
        np.array(
            [[20, 20, 30, 30, 0.99], [65, 40, 25, 35, 0.98]],
            dtype=np.float32,
        )
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    noise = np.random.default_rng(0).integers(0, 256, (35, 30, 3), dtype=np.uint8)
    frame[20:50, 20:50] = noise[:30]
    frame[40:75, 65:90] = np.flip(noise[:, :25], axis=0)
    timings: dict[str, float] = {}

    blurrer = FaceBlurrer(config(tmp_path), detector_factory=lambda *_: detector)
    result = blurrer.blur(frame, timings)

    assert result.faces_blurred == 2
    assert detector.input_sizes == [(120, 100)]
    assert timings["face_blur"] >= 0
    assert (
        cv2.Laplacian(result.image[20:50, 20:50], cv2.CV_64F).var()
        < cv2.Laplacian(frame[20:50, 20:50], cv2.CV_64F).var()
    )
    assert not np.shares_memory(result.image, frame)


def test_blur_leaves_a_face_free_frame_visually_unchanged(tmp_path: Path) -> None:
    detector = FakeYuNet(None)
    frame = np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)

    result = FaceBlurrer(config(tmp_path), detector_factory=lambda *_: detector).blur(frame)

    assert result.faces_blurred == 0
    assert np.array_equal(result.image, frame)


def test_blur_rejects_non_bgr_input(tmp_path: Path) -> None:
    blurrer = FaceBlurrer(config(tmp_path), detector_factory=lambda *_: FakeYuNet(None))

    with pytest.raises(ValueError, match="BGR"):
        blurrer.blur(np.zeros((10, 10), dtype=np.uint8))


def test_expanded_box_clips_to_the_image_boundary() -> None:
    assert _expanded_box(0, 0, 20, 30, 100, 100, 0.1) == (0, 0, 22, 33)
