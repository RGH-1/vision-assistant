"""Sharpness and exposure selection before the expensive ingest stages (§6.3.2)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from server.settings import CONFIG_PATH


@dataclass(frozen=True)
class FrameSelectionConfig:
    analysis_width: int
    sharpness_min: float
    sharpness_median_factor: float
    sharpness_soft_ratio: float
    sharpness_history_size: int
    brightness_dark: float
    brightness_bright: float
    contrast_min: float
    unsharp_sigma: float
    unsharp_amount: float


@dataclass(frozen=True)
class FrameMetrics:
    sharpness: float
    brightness: float
    contrast: float
    adaptive_sharpness_threshold: float


@dataclass(frozen=True)
class FrameSelectionResult:
    """A selected frame and the values to store with it later in the pipeline."""

    accepted: bool
    image: np.ndarray
    metrics: FrameMetrics
    needs_enhancement: bool
    enhancement_reasons: tuple[str, ...]
    unsharp_applied: bool
    drop_reason: str | None


def load_frame_selection_config(path: Path = CONFIG_PATH) -> FrameSelectionConfig:
    """Read the tunable quality thresholds from the committed YAML configuration."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["frame_selection"]
    return FrameSelectionConfig(**values)


class FrameSelector:
    """Keep usable frames using the adaptive sharpness/exposure rules from §6.3.2."""

    def __init__(self, config: FrameSelectionConfig) -> None:
        if config.analysis_width <= 0 or config.sharpness_history_size <= 0:
            raise ValueError("analysis_width and sharpness_history_size must be positive")
        if not 0 < config.sharpness_median_factor <= 1:
            raise ValueError("sharpness_median_factor must be in (0, 1]")
        if not 0 < config.sharpness_soft_ratio <= 1:
            raise ValueError("sharpness_soft_ratio must be in (0, 1]")
        self._config = config
        self._sharpness_history: deque[float] = deque(maxlen=config.sharpness_history_size)

    def select(self, frame: np.ndarray) -> FrameSelectionResult:
        """Measure one BGR frame, dropping only unusable bright or strongly blurred frames."""
        grayscale = _grayscale_at_width(frame, self._config.analysis_width)
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        brightness = float(grayscale.mean())
        contrast = float(grayscale.std())
        threshold = self._adaptive_threshold()
        metrics = FrameMetrics(sharpness, brightness, contrast, threshold)
        self._sharpness_history.append(sharpness)

        if brightness > self._config.brightness_bright:
            return FrameSelectionResult(False, frame, metrics, False, (), False, "overexposed")
        reasons = []
        if brightness < self._config.brightness_dark:
            reasons.append("dark")
        if contrast < self._config.contrast_min:
            reasons.append("low_contrast")
        if reasons:
            return FrameSelectionResult(True, frame, metrics, True, tuple(reasons), False, None)
        if sharpness < threshold * self._config.sharpness_soft_ratio:
            return FrameSelectionResult(False, frame, metrics, False, (), False, "blurred")

        unsharp_applied = sharpness < threshold
        selected = _unsharp_mask(frame, self._config) if unsharp_applied else frame
        return FrameSelectionResult(
            True,
            selected,
            metrics,
            bool(reasons),
            tuple(reasons),
            unsharp_applied,
            None,
        )

    def _adaptive_threshold(self) -> float:
        median = 0.0 if not self._sharpness_history else float(np.median(self._sharpness_history))
        return max(self._config.sharpness_min, self._config.sharpness_median_factor * median)


def _grayscale_at_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, frame_width = frame.shape[:2]
    resized = cv2.resize(frame, (width, round(height * width / frame_width)), cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _unsharp_mask(frame: np.ndarray, config: FrameSelectionConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), config.unsharp_sigma)
    return cv2.addWeighted(frame, 1 + config.unsharp_amount, blurred, -config.unsharp_amount, 0)
