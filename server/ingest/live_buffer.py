"""Short-lived, thread-safe frame storage for questions about the current scene."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

type Detection = dict[str, Any]
type LiveFrame = tuple[float, bytes, float, tuple[Detection, ...]]


class LiveBuffer:
    """Keep the latest ten JPEG frames shared between ingest and API threads (§5.3)."""

    def __init__(self) -> None:
        self._frames: deque[LiveFrame] = deque(maxlen=10)
        self._lock = threading.Lock()

    def add(
        self,
        timestamp: float,
        frame: np.ndarray,
        sharpness: float,
        detections: Sequence[Detection],
    ) -> None:
        """Resize and encode a sampled frame before adding it to the live window."""
        height, width = frame.shape[:2]
        resized = cv2.resize(
            frame, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA
        )
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("could not encode live frame as JPEG")
        entry: LiveFrame = (
            timestamp,
            encoded.tobytes(),
            sharpness,
            tuple(dict(detection) for detection in detections),
        )
        with self._lock:
            self._frames.append(entry)

    def get_sharpest(self, n: int = 5) -> LiveFrame | None:
        """Return the sharpest frame among the latest ``n`` frames for a now-question."""
        if n <= 0:
            raise ValueError("n must be positive")
        with self._lock:
            candidates = tuple(self._frames)[-n:]
        if not candidates:
            return None
        # A tie favours the newest capture, which is better for a live answer.
        return max(reversed(candidates), key=lambda entry: entry[2])

    def clear(self) -> None:
        """Remove all in-memory frames when a session ends or is deleted (§7.5)."""
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
