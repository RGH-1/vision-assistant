"""Enhance dark, flat or slightly soft frames without attempting deblurring (§6.3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from server.settings import CONFIG_PATH


@dataclass(frozen=True)
class FrameEnhancementConfig:
    dark_brightness: float
    gamma_min: float
    gamma_max: float
    clahe_clip_limit: float
    clahe_tile_size: int
    unsharp_sigma: float
    unsharp_amount: float


@dataclass(frozen=True)
class EnhancementResult:
    """Processed-frame fields ready to pass directly to ``FrameStore``."""

    image: np.ndarray
    enhanced: bool
    enhance_ops: tuple[str, ...]
    original: np.ndarray | None
    dropped: bool = False


def load_frame_enhancement_config(path: Path = CONFIG_PATH) -> FrameEnhancementConfig:
    """Read the committed tunable enhancement parameters."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["frame_enhancement"]
    return FrameEnhancementConfig(**values)


class FrameEnhancer:
    """Apply only deterministic, reportable image operations from the project design."""

    def __init__(self, config: FrameEnhancementConfig) -> None:
        if not 0 < config.gamma_min <= config.gamma_max <= 1:
            raise ValueError("gamma range must be within (0, 1]")
        if config.clahe_clip_limit <= 0 or config.clahe_tile_size <= 0:
            raise ValueError("CLAHE parameters must be positive")
        if config.unsharp_sigma <= 0 or config.unsharp_amount <= 0:
            raise ValueError("unsharp parameters must be positive")
        self._config = config

    def enhance(
        self,
        frame: np.ndarray,
        *,
        brightness: float,
        low_contrast: bool = False,
        slightly_soft: bool = False,
        strong_motion_blur: bool = False,
    ) -> EnhancementResult:
        """Return a corrected copy, preserving the original only when a correction was used."""
        if strong_motion_blur:
            return EnhancementResult(frame, False, (), None, dropped=True)

        image = frame.copy()
        operations: list[str] = []
        if brightness < self._config.dark_brightness:
            image = _gamma_correct(image, self._gamma_for(brightness))
            image = _clahe_l_channel(image, self._config)
            operations.extend(("gamma", "clahe"))
        elif low_contrast:
            image = _clahe_l_channel(image, self._config)
            operations.append("clahe")
        if slightly_soft:
            image = _unsharp_mask(image, self._config)
            operations.append("unsharp")

        return EnhancementResult(
            image=image,
            enhanced=bool(operations),
            enhance_ops=tuple(operations),
            original=frame.copy() if operations else None,
        )

    def _gamma_for(self, brightness: float) -> float:
        bounded = min(max(brightness, 0.0), self._config.dark_brightness)
        span = self._config.gamma_max - self._config.gamma_min
        return self._config.gamma_min + span * (bounded / self._config.dark_brightness)


def _gamma_correct(frame: np.ndarray, gamma: float) -> np.ndarray:
    lookup = np.array([(value / 255) ** gamma * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lookup)


def _clahe_l_channel(frame: np.ndarray, config: FrameEnhancementConfig) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_size, config.clahe_tile_size),
    )
    corrected = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((corrected, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _unsharp_mask(frame: np.ndarray, config: FrameEnhancementConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), config.unsharp_sigma)
    return cv2.addWeighted(
        frame,
        1 + config.unsharp_amount,
        blurred,
        -config.unsharp_amount,
        0,
    )
