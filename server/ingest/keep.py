"""Informative-frame and best-of-window selection for ingest (§6.3.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import yaml

from server.settings import CONFIG_PATH

KeepReason = Literal["novel", "new_objects", "text", "heartbeat"]
SIGN_LABELS = frozenset(("sign", "exit sign", "room number sign"))


class DetectionLike(Protocol):
    label: str
    confidence: float


@dataclass(frozen=True)
class FrameKeepConfig:
    detection_confidence_min: float
    edge_density_min: float
    canny_low: int
    canny_high: int
    window_seconds: float


@dataclass(frozen=True)
class FrameKeepCandidate:
    """A quality- and novelty-passing frame considered for one window."""

    image: np.ndarray
    captured_at: float
    sharpness: float
    detections: Sequence[DetectionLike | Mapping[str, Any]]
    keep_reason: KeepReason


@dataclass(frozen=True)
class KeptFrame:
    """The sharpest informative candidate selected for storage from a time window."""

    candidate: FrameKeepCandidate
    edge_density: float


@dataclass(frozen=True)
class KeepStats:
    sampled_frames: int
    kept_frames: int

    @property
    def kept_ratio(self) -> float:
        return self.kept_frames / self.sampled_frames if self.sampled_frames else 0.0


def load_frame_keep_config(path: Path = CONFIG_PATH) -> FrameKeepConfig:
    """Read the committed informative-frame and one-second-window thresholds."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["frame_keep"]
    return FrameKeepConfig(**values)


class FrameKeepDecider:
    """Keep the sharpest informative candidate in each one-second window."""

    def __init__(self, config: FrameKeepConfig) -> None:
        if not 0 < config.detection_confidence_min <= 1:
            raise ValueError("detection_confidence_min must be in (0, 1]")
        if not 0 <= config.edge_density_min <= 1:
            raise ValueError("edge_density_min must be in [0, 1]")
        if config.canny_low < 0 or config.canny_high <= config.canny_low:
            raise ValueError("Canny thresholds must be ordered positive values")
        if config.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._config = config
        self._window: int | None = None
        self._best: KeptFrame | None = None
        self._last_sample_at: float | None = None
        self._sampled_frames = 0
        self._kept_frames = 0

    @property
    def stats(self) -> KeepStats:
        """Counters for the ingest status event: sampled, kept, and their ratio."""
        return KeepStats(self._sampled_frames, self._kept_frames)

    def consider(
        self,
        candidate: FrameKeepCandidate | None,
        captured_at: float,
    ) -> tuple[KeptFrame, ...]:
        """Count a sample and emit the previous window's winner when its window closes."""
        if self._last_sample_at is not None and captured_at < self._last_sample_at:
            raise ValueError("sample timestamps must be ordered")
        if candidate is not None and candidate.captured_at != captured_at:
            raise ValueError("candidate timestamp must match its sampled timestamp")

        self._sampled_frames += 1
        self._last_sample_at = captured_at
        window = floor(captured_at / self._config.window_seconds)
        emitted: tuple[KeptFrame, ...] = ()
        if self._window is None:
            self._window = window
        elif window != self._window:
            emitted = self.flush()
            self._window = window

        if candidate is not None:
            self._consider_in_window(candidate)
        return emitted

    def flush(self) -> tuple[KeptFrame, ...]:
        """Emit the current window's winner at stream shutdown or a window boundary."""
        if self._best is None:
            return ()
        kept = self._best
        self._best = None
        self._kept_frames += 1
        return (kept,)

    def _consider_in_window(self, candidate: FrameKeepCandidate) -> None:
        edge_density = _edge_density(candidate.image, self._config)
        if not _is_informative(candidate.detections, edge_density, self._config):
            return
        result = KeptFrame(candidate, edge_density)
        if self._best is None or result.candidate.sharpness > self._best.candidate.sharpness:
            self._best = result


def _edge_density(frame: np.ndarray, config: FrameKeepConfig) -> float:
    if frame.ndim == 2:
        grayscale = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Canny edge density requires a grayscale or BGR image")
    edges = cv2.Canny(grayscale, config.canny_low, config.canny_high)
    return float((edges > 0).mean())


def _is_informative(
    detections: Sequence[DetectionLike | Mapping[str, Any]],
    edge_density: float,
    config: FrameKeepConfig,
) -> bool:
    for detection in detections:
        label, confidence = _detection_label_and_confidence(detection)
        if label in SIGN_LABELS or confidence >= config.detection_confidence_min:
            return True
    return edge_density >= config.edge_density_min


def _detection_label_and_confidence(
    detection: DetectionLike | Mapping[str, Any],
) -> tuple[str, float]:
    if isinstance(detection, Mapping):
        return str(detection["label"]), float(detection["confidence"])
    return detection.label, detection.confidence
