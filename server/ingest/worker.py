"""Latest-frame RTSP reader and fixed-rate sampler (design §6.3.1)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FrameHandler = Callable[[np.ndarray, float], None]
ConnectionEventFactory = Callable[[bool], dict[str, Any]]


@dataclass(frozen=True)
class IngestSource:
    """Video input selected by the server startup configuration."""

    url: str
    realtime: bool = False


def source_from_args(args: Sequence[str] | None = None) -> IngestSource:
    """Parse the test-stream options documented for the ingest worker."""
    parser = argparse.ArgumentParser(description="Read a Vision Assistant video source")
    parser.add_argument("--source", required=True, help="RTSP URL or file:<path>")
    parser.add_argument(
        "--realtime", action="store_true", help="play a file source at its recorded frame rate"
    )
    parsed = parser.parse_args(args)
    source = parsed.source
    if source.startswith("file:"):
        source = source.removeprefix("file:")
    return IngestSource(url=source, realtime=parsed.realtime)


class IngestWorker:
    """Read continuously, retaining one newest frame for the 5 Hz pipeline sampler.

    The reader owns its ``VideoCapture``.  This prevents a slow downstream image
    pipeline from making OpenCV return queued, stale RTSP frames.
    """

    def __init__(
        self,
        source: IngestSource,
        frame_handler: FrameHandler,
        hub: Any | None = None,
        *,
        connection_event_factory: ConnectionEventFactory | None = None,
        sample_hz: float = 5.0,
        no_frame_timeout_s: float = 3.0,
        reconnect_delay_s: float = 0.25,
        cv2_module: Any = cv2,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        if no_frame_timeout_s <= 0:
            raise ValueError("no_frame_timeout_s must be positive")
        self._source = source
        self._frame_handler = frame_handler
        self._hub = hub
        self._connection_event_factory = connection_event_factory
        self._sample_period_s = 1 / sample_hz
        self._no_frame_timeout_s = no_frame_timeout_s
        self._reconnect_delay_s = reconnect_delay_s
        self._cv2 = cv2_module
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_at = 0.0
        self._reader: threading.Thread | None = None
        self._sampler: threading.Thread | None = None

    def start(self) -> None:
        """Start the capture and sampler threads.  Safe to call once."""
        if self._reader is not None or self._sampler is not None:
            raise RuntimeError("ingest worker is already running")
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_forever, name="rtsp-reader", daemon=True)
        self._sampler = threading.Thread(
            target=self._sample_forever, name="frame-sampler", daemon=True
        )
        self._reader.start()
        self._sampler.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Request shutdown and wait for both threads to finish."""
        self._stop.set()
        for thread in (self._reader, self._sampler):
            if thread is not None:
                thread.join(timeout=timeout_s)
        self._reader = None
        self._sampler = None

    def _publish_connection(self, connected: bool) -> None:
        if self._hub is not None:
            # The event hub transfers this to the asyncio loop (§5.3).
            if self._connection_event_factory is not None:
                event = self._connection_event_factory(connected)
            else:
                event = {"type": "status", "stream_connected": connected}
            self._hub.publish(event)

    def _open_capture(self) -> Any:
        # OpenCV reads this option when the capture is constructed.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        return self._cv2.VideoCapture(self._source.url, self._cv2.CAP_FFMPEG)

    def _read_forever(self) -> None:
        capture: Any | None = None
        last_frame_at = self._monotonic()
        connected = False
        while not self._stop.is_set():
            if capture is None:
                capture = self._open_capture()
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    self._publish_connection(False)
                    self._stop.wait(self._reconnect_delay_s)
                    continue

            grabbed = capture.grab()
            now = self._monotonic()
            if grabbed:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._latest_at = now
                    last_frame_at = now
                    if not connected:
                        connected = True
                        self._publish_connection(True)
                    self._wait_for_file_frame(capture)
                    continue

            if now - last_frame_at >= self._no_frame_timeout_s:
                logger.warning("No video frame for %.1f seconds; reconnecting", now - last_frame_at)
                capture.release()
                capture = None
                self._clear_latest_frame()
                last_frame_at = now
                if connected:
                    connected = False
                self._publish_connection(False)
            else:
                self._stop.wait(0.01)

        if capture is not None:
            capture.release()

    def _clear_latest_frame(self) -> None:
        """Do not keep sending a disconnected stream's stale image to live questions."""
        with self._frame_lock:
            self._latest_frame = None
            self._latest_at = 0.0

    def _wait_for_file_frame(self, capture: Any) -> None:
        if not self._source.realtime:
            return
        fps = capture.get(self._cv2.CAP_PROP_FPS)
        if fps > 0:
            self._stop.wait(1 / fps)

    def _sample_forever(self) -> None:
        next_sample = self._monotonic()
        while not self._stop.is_set():
            now = self._monotonic()
            if now < next_sample:
                self._stop.wait(next_sample - now)
                continue
            with self._frame_lock:
                frame = self._latest_frame
                captured_at = self._latest_at
            if frame is not None:
                self._frame_handler(frame, captured_at)
            next_sample += self._sample_period_s
            # Do not create a burst after a stalled downstream handler.
            if next_sample < self._monotonic():
                next_sample = self._monotonic() + self._sample_period_s


def main() -> None:
    """Small manual runner for the RTSP/file reader before startup integration."""
    source = source_from_args()
    worker = IngestWorker(source, lambda _frame, _captured_at: None)
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    main()
