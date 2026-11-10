"""YOLOE object detection for accepted ingest frames (design §6.3.4)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import yaml

from server.ingest.store import StoredDetection
from server.settings import CONFIG_PATH


class YoloeModel(Protocol):
    """Small surface used from Ultralytics, making detection unit-testable without a GPU."""

    def set_classes(self, classes: list[str]) -> None: ...

    def predict(self, source: np.ndarray, **kwargs: Any) -> list[Any]: ...


YoloeFactory = Callable[[str], YoloeModel]


@dataclass(frozen=True)
class ObjectDetectionConfig:
    model: str
    fallback_model: str
    confidence_min: float
    image_size: int
    classes: tuple[str, ...]


@dataclass(frozen=True)
class Detection:
    """One model result in the database and live-buffer formats required by §6.3.4."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    area_ratio: float
    position: str

    def as_live_buffer_value(self) -> dict[str, float | str | list[float]]:
        """Return the JSON-compatible detection kept with a live frame."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "area_ratio": self.area_ratio,
            "position": self.position,
        }

    def as_stored_detection(self) -> StoredDetection:
        """Return the frame-store value without making the detector own persistence."""
        return StoredDetection(
            self.label,
            self.confidence,
            self.bbox,
            self.area_ratio,
            self.position,
        )


def load_object_detection_config(path: Path = CONFIG_PATH) -> ObjectDetectionConfig:
    """Read the committed YOLOE settings and class vocabulary."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["object_detection"]
    return ObjectDetectionConfig(**{**values, "classes": tuple(values["classes"])})


class YoloeDetector:
    """Load YOLOE once inside the ingest thread and detect objects in accepted frames."""

    def __init__(
        self,
        config: ObjectDetectionConfig,
        *,
        model_factory: YoloeFactory | None = None,
    ) -> None:
        if not 0 < config.confidence_min <= 1:
            raise ValueError("confidence_min must be in (0, 1]")
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not config.classes or len(config.classes) > 40:
            raise ValueError("YOLOE needs between 1 and 40 classes")
        if len(set(config.classes)) != len(config.classes):
            raise ValueError("YOLOE classes must be unique")

        factory = model_factory or _ultralytics_yoloe
        self._config = config
        self._model = _load_model(factory, config)
        # YOLOE embeds these phrases.  Set them exactly once at ingest startup,
        # while the laptop still has network access, so demo inference is offline.
        self._model.set_classes(list(config.classes))

    def detect(self, frame: np.ndarray, timings_ms: dict[str, float]) -> tuple[Detection, ...]:
        """Detect only configured classes and record this frame's model time."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("YOLOE requires a BGR image with three channels")

        started_at = perf_counter()
        results = self._model.predict(
            frame,
            conf=self._config.confidence_min,
            imgsz=self._config.image_size,
            device=0,
            half=True,
            verbose=False,
        )
        timings_ms["yoloe"] = (perf_counter() - started_at) * 1000
        if not results:
            return ()
        return _detections_from_result(
            results[0], frame.shape[1], frame.shape[0], self._config.confidence_min
        )


def _ultralytics_yoloe(model_path: str) -> YoloeModel:
    """Delay the heavy Ultralytics import until the ingest thread constructs the detector."""
    from ultralytics import YOLOE

    return YOLOE(model_path)


def _load_model(factory: YoloeFactory, config: ObjectDetectionConfig) -> YoloeModel:
    try:
        return factory(config.model)
    except Exception:
        try:
            return factory(config.fallback_model)
        except Exception as fallback_error:
            raise RuntimeError("could not load either configured YOLOE model") from fallback_error


def _detections_from_result(
    result: Any,
    width: int,
    height: int,
    confidence_min: float,
) -> tuple[Detection, ...]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()

    coordinates = _as_rows(boxes.xyxy)
    confidences = _as_values(boxes.conf)
    classes = _as_values(boxes.cls)
    detections: list[Detection] = []
    for bbox, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
        if confidence < confidence_min:
            continue
        x1, y1, x2, y2 = _normalized_bbox(bbox, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(result.names[int(class_id)])
        area_ratio = (x2 - x1) * (y2 - y1)
        position = _position((x1 + x2) / 2)
        detections.append(
            Detection(label, float(confidence), (x1, y1, x2, y2), area_ratio, position)
        )
    return tuple(detections)


def _as_rows(value: Any) -> list[list[float]]:
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value).tolist()


def _as_values(value: Any) -> list[float]:
    return np.asarray(value.cpu() if hasattr(value, "cpu") else value).reshape(-1).tolist()


def _normalized_bbox(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        float(np.clip(x1 / width, 0, 1)),
        float(np.clip(y1 / height, 0, 1)),
        float(np.clip(x2 / width, 0, 1)),
        float(np.clip(y2 / height, 0, 1)),
    )


def _position(center_x: float) -> str:
    if center_x < 1 / 3:
        return "left"
    if center_x > 2 / 3:
        return "right"
    return "center"


def main() -> None:
    """Download YOLOE's text encoder and prepare the configured offline vocabulary."""
    parser = argparse.ArgumentParser(description="Warm the YOLOE model for an offline demo")
    parser.add_argument(
        "--warmup", action="store_true", help="download and encode the configured classes"
    )
    parser.parse_args()
    config = load_object_detection_config()
    YoloeDetector(config)
    print(f"YOLOE is ready with {len(config.classes)} classes.")


if __name__ == "__main__":
    main()
