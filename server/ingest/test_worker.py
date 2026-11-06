import os
import time
from typing import Any

import numpy as np

from server.ingest.worker import IngestSource, IngestWorker, source_from_args


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        if self.index >= len(self.frames):
            return False
        self.index += 1
        return True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return True, self.frames[self.index - 1]

    def get(self, _: int) -> float:
        return 30.0

    def release(self) -> None:
        self.released = True


class FakeCV2:
    CAP_FFMPEG = 1900
    CAP_PROP_FPS = 5

    def __init__(self) -> None:
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, _: str, backend: int) -> FakeCapture:
        assert backend == self.CAP_FFMPEG
        capture = FakeCapture([np.array([[1]]), np.array([[2]]), np.array([[3]])])
        self.captures.append(capture)
        return capture


class Hub:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def test_source_args_accept_file_and_realtime() -> None:
    assert source_from_args(["--source", "file:walk.mp4", "--realtime"]) == IngestSource(
        "walk.mp4", realtime=True
    )


def test_worker_uses_ffmpeg_tcp_and_hands_only_the_newest_frame_to_sampler() -> None:
    fake_cv2 = FakeCV2()
    hub = Hub()
    received: list[int] = []
    worker = IngestWorker(
        IngestSource("rtsp://localhost:8554/phone"),
        lambda frame, _: received.append(int(frame[0, 0])),
        hub,
        sample_hz=30,
        no_frame_timeout_s=0.03,
        reconnect_delay_s=0.005,
        cv2_module=fake_cv2,
    )
    worker.start()
    time.sleep(0.08)
    worker.stop()

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"
    assert received and set(received) == {3}
    assert {event["stream_connected"] for event in hub.events} == {True, False}
    assert all(capture.released for capture in fake_cv2.captures)


def test_clearing_connection_frame_prevents_stale_live_samples() -> None:
    worker = IngestWorker(IngestSource("rtsp://localhost:8554/phone"), lambda *_: None)
    worker._latest_frame = np.array([[1]])
    worker._latest_at = 1.0

    worker._clear_latest_frame()

    assert worker._latest_frame is None
    assert worker._latest_at == 0.0
