"""Blur detected faces before a frame leaves the ingest thread (§6.3.6, §9)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from server.settings import CONFIG_PATH, ROOT

DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"


class YuNetDetector(Protocol):
    """The small OpenCV detector surface used by the blurrer."""

    def setInputSize(self, input_size: tuple[int, int]) -> None: ...

    def detect(self, image: np.ndarray) -> tuple[Any, np.ndarray | None]: ...


YuNetFactory = Callable[[str, str, tuple[int, int], float, float, int], YuNetDetector]


@dataclass(frozen=True)
class FaceBlurConfig:
    model_path: Path
    score_threshold: float
    nms_threshold: float
    top_k: int
    box_margin: float
    blur_kernel_ratio: float
    blur_kernel_min: int


@dataclass(frozen=True)
class FaceBlurResult:
    image: np.ndarray
    faces_blurred: int


def load_face_blur_config(path: Path = CONFIG_PATH) -> FaceBlurConfig:
    """Load the committed YuNet configuration without downloading a model."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["face_blur"]
    return FaceBlurConfig(
        **{**values, "model_path": ROOT / values["model_path"]},
    )


class FaceBlurrer:
    """Use YuNet detection only; it never identifies or classifies people."""

    def __init__(
        self,
        config: FaceBlurConfig,
        *,
        detector_factory: YuNetFactory | None = None,
    ) -> None:
        if detector_factory is None and not config.model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model is missing: {config.model_path}. "
                "Download it into data/models before starting ingest."
            )
        if not 0 < config.score_threshold <= 1 or not 0 < config.nms_threshold <= 1:
            raise ValueError("YuNet thresholds must be in (0, 1]")
        if config.top_k <= 0 or config.box_margin < 0:
            raise ValueError("YuNet top_k must be positive and box_margin cannot be negative")
        if not 0 < config.blur_kernel_ratio <= 1 or config.blur_kernel_min < 3:
            raise ValueError("invalid face blur kernel configuration")

        factory = detector_factory or cv2.FaceDetectorYN.create
        self._config = config
        self._detector = factory(
            str(config.model_path),
            "",
            (320, 320),
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )

    def blur(self, frame: np.ndarray, timings_ms: dict[str, float] | None = None) -> FaceBlurResult:
        """Return a copied BGR image with every YuNet face box strongly blurred."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("YuNet requires a BGR image with three channels")

        started_at = perf_counter()
        image = frame.copy()
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _unused, faces = self._detector.detect(image)
        boxes = () if faces is None else tuple(faces)
        for face in boxes:
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            left, top, right, bottom = _expanded_box(
                x, y, box_width, box_height, width, height, self._config.box_margin
            )
            if right > left and bottom > top:
                region = image[top:bottom, left:right]
                kernel = _blur_kernel(region.shape[0], region.shape[1], self._config)
                image[top:bottom, left:right] = cv2.GaussianBlur(region, (kernel, kernel), 0)
        if timings_ms is not None:
            timings_ms["face_blur"] = (perf_counter() - started_at) * 1000
        return FaceBlurResult(image, len(boxes))


def _expanded_box(
    x: int,
    y: int,
    box_width: int,
    box_height: int,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    horizontal_margin = round(box_width * margin)
    vertical_margin = round(box_height * margin)
    return (
        max(0, x - horizontal_margin),
        max(0, y - vertical_margin),
        min(frame_width, x + box_width + horizontal_margin),
        min(frame_height, y + box_height + vertical_margin),
    )


def _blur_kernel(height: int, width: int, config: FaceBlurConfig) -> int:
    size = max(config.blur_kernel_min, round(min(height, width) * config.blur_kernel_ratio))
    return size if size % 2 else size + 1


def main() -> None:
    """Blur a supplied image for the documented manual privacy check."""
    parser = argparse.ArgumentParser(description="Blur all YuNet-detected faces in an image")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(args.input)
    result = FaceBlurrer(load_face_blur_config()).blur(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result.image):
        raise RuntimeError(f"could not write {args.output}")
    print(f"Blurred {result.faces_blurred} face(s) into {args.output}")


if __name__ == "__main__":
    main()
