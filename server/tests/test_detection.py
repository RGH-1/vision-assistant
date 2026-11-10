from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from server.ingest.detection import (
    ObjectDetectionConfig,
    YoloeDetector,
    load_object_detection_config,
)


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [
                [0.0, 0.0, 20.0, 20.0],
                [30.0, 10.0, 70.0, 50.0],
                [80.0, 0.0, 100.0, 100.0],
                [0.0, 0.0, 0.0, 10.0],
            ]
        )
        self.conf = np.array([0.90, 0.35, 0.80, 0.99])
        self.cls = np.array([0, 1, 2, 1])

    def __len__(self) -> int:
        return len(self.conf)


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "door", 1: "stairs", 2: "sign"}


class FakeModel:
    def __init__(self) -> None:
        self.classes: list[str] | None = None
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes

    def predict(self, source: np.ndarray, **kwargs: Any) -> list[FakeResult]:
        self.calls.append((source, kwargs))
        return [FakeResult()]


@pytest.fixture
def config() -> ObjectDetectionConfig:
    return ObjectDetectionConfig(
        model="primary.pt",
        fallback_model="fallback.pt",
        confidence_min=0.35,
        image_size=640,
        classes=("door", "stairs", "sign"),
    )


def test_detector_sets_classes_once_and_returns_normalized_detections(
    config: ObjectDetectionConfig,
) -> None:
    model = FakeModel()
    detector = YoloeDetector(config, model_factory=lambda _: model)
    timings_ms: dict[str, float] = {}

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), timings_ms)

    assert model.classes == ["door", "stairs", "sign"]
    assert len(model.calls) == 1
    assert model.calls[0][1] == {
        "conf": 0.35,
        "imgsz": 640,
        "device": 0,
        "half": True,
        "verbose": False,
    }
    assert [(item.label, item.position) for item in detections] == [
        ("door", "left"),
        ("stairs", "center"),
        ("sign", "right"),
    ]
    assert detections[1].bbox == (0.3, 0.1, 0.7, 0.5)
    assert detections[1].area_ratio == pytest.approx(0.16)
    assert timings_ms["yoloe"] >= 0
    live_value = detections[0].as_live_buffer_value()
    assert live_value["label"] == "door"
    assert live_value["position"] == "left"
    assert live_value["bbox"] == [0.0, 0.0, 0.2, 0.2]
    assert live_value["confidence"] == pytest.approx(0.9)
    assert live_value["area_ratio"] == pytest.approx(0.04)
    assert detections[0].as_stored_detection().label == "door"


def test_detector_uses_the_fallback_model(config: ObjectDetectionConfig) -> None:
    model = FakeModel()
    loaded: list[str] = []

    def factory(model_path: str) -> FakeModel:
        loaded.append(model_path)
        if model_path == config.model:
            raise OSError("primary model unavailable")
        return model

    YoloeDetector(config, model_factory=factory)

    assert loaded == ["primary.pt", "fallback.pt"]
    assert model.classes == ["door", "stairs", "sign"]


def test_rejects_invalid_configuration(config: ObjectDetectionConfig) -> None:
    with pytest.raises(ValueError, match="confidence"):
        YoloeDetector(replace(config, confidence_min=0), model_factory=lambda _: FakeModel())
    with pytest.raises(ValueError, match="40"):
        YoloeDetector(
            replace(config, classes=tuple(str(index) for index in range(41))),
            model_factory=lambda _: FakeModel(),
        )


def test_config_matches_the_design_class_list() -> None:
    config = load_object_detection_config()

    assert config.model == "yoloe-26s-seg.pt"
    assert config.fallback_model == "yoloe-11s-seg.pt"
    assert config.confidence_min == 0.35
    assert config.image_size == 640
    assert len(config.classes) == 38
    assert {"door", "wooden door", "stairs", "wooden stairs"} <= set(config.classes)
    assert {"door", "stairs", "sign", "exit sign", "room number sign"} <= set(config.classes)


def test_crop_detections_use_full_frame_coordinates(
    config: ObjectDetectionConfig,
) -> None:
    model = FakeModel()
    detector = YoloeDetector(config, model_factory=lambda _: model)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    detections = detector.detect_region(
        frame,
        region=(0.25, 0.25, 0.75, 0.75),
        timings_ms={},
    )

    assert len(detections) == 3

    door = detections[0]

    assert door.label == "door"
    assert door.bbox == pytest.approx((0.25, 0.25, 0.35, 0.35))
    assert door.area_ratio == pytest.approx(0.01)
    assert door.position == "left"
def test_detection_labels_are_normalized():
    from types import SimpleNamespace
    from server.ingest.detection import _detections_from_result

    class FakeBoxes:
        xyxy = [
            [10, 10, 50, 80],
            [60, 10, 100, 80],
            [110, 10, 150, 80],
        ]
        conf = [0.75, 0.46, 0.80]
        cls = [0, 1, 2]

        def __len__(self):
            return len(self.conf)

    result = SimpleNamespace(
        boxes=FakeBoxes(),
        names={
            0: "wooden door",
            1: "wooden stairs",
            2: "person",
        },
    )

    detections = _detections_from_result(
        result,
        width=200,
        height=100,
        confidence_min=0.35,
    )

    assert [d.label for d in detections] == [
        "door",
        "stairs",
        "person",
    ]