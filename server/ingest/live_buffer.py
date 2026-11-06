"""Thread-safe, short-lived frames for questions about the current scene."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class LiveFrame:
    captured_at: float
    jpeg_bytes: bytes
    sharpness: float
    detections: tuple[dict[str, Any], ...]


class LiveBuffer:
    """The last ten sampled frames, guarded for ingest/web-route access (§5.3)."""

    def __init__(self, maxlen: int = 10) -> None:
        self._frames: deque[LiveFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(
        self, frame: np.ndarray, captured_at: float, detections: tuple[dict[str, Any], ...] = ()
    ) -> None:
        resized = self._resize(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        entry = LiveFrame(captured_at, encoded.tobytes(), sharpness, detections)
        with self._lock:
            self._frames.append(entry)

    def snapshot(self) -> list[LiveFrame]:
        with self._lock:
            return list(self._frames)

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= 960:
            return frame
        return cv2.resize(frame, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
