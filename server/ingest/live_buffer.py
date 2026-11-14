"""Thread-safe, short-lived frames for questions about the current scene."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from server.ingest.faceblur import FaceBlurrer

type Detection = dict[str, Any]
type LiveFrame = tuple[float, bytes, float, tuple[Detection, ...]]


class LiveBuffer:
    """The last ten sampled frames, guarded for ingest/web-route access (§5.3)."""

    def __init__(self) -> None:
        self._frames: deque[LiveFrame] = deque(maxlen=10)
        self._lock = threading.Lock()

    def add(
        self,
        frame: np.ndarray,
        captured_at: float,
        detections: Sequence[Detection] = (),
        *,
        face_blurrer: FaceBlurrer,
    ) -> None:
        """Blur before encoding one sampled frame for a live question (§6.3.6)."""
        resized = self._resize(face_blurrer.blur(frame).image)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("could not encode live frame as JPEG")
        entry: LiveFrame = (
            captured_at,
            encoded.tobytes(),
            sharpness,
            tuple(dict(detection) for detection in detections),
        )
        with self._lock:
            self._frames.append(entry)

    def get_sharpest(self, n: int = 5) -> LiveFrame | None:
        """Return the sharpest of the latest ``n`` frames for a now-question."""
        if n <= 0:
            raise ValueError("n must be positive")
        with self._lock:
            candidates = tuple(self._frames)[-n:]
        if not candidates:
            return None
        # A tie favours the newest capture, which is better for a live answer.
        return max(reversed(candidates), key=lambda entry: entry[2])

    def snapshot(self) -> list[LiveFrame]:
        """Return a stable copy for API use without exposing the locked deque."""
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        """Remove all in-memory frames when a session ends or is deleted (§7.5)."""
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= 960:
            return frame
        return cv2.resize(frame, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
