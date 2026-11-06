from datetime import UTC
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from server.common.embeddings import EMBEDDING_DIMENSIONS
from server.ingest.fixture import process_video
from server.ingest.store import ProcessedFrame


class FakeEncoder:
    def encode(self, _: np.ndarray, timings_ms: dict[str, float]) -> np.ndarray:
        timings_ms["clip"] = 1.0
        return np.eye(1, EMBEDDING_DIMENSIONS, dtype=np.float32)[0]


class FakeStore:
    def __init__(self) -> None:
        self.frames: list[ProcessedFrame] = []

    def store(self, frame: ProcessedFrame) -> int:
        self.frames.append(frame)
        return len(self.frames)


def test_process_video_samples_each_second(tmp_path: Path) -> None:
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2, (32, 24))
    assert writer.isOpened()
    for value in range(6):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()

    store = FakeStore()
    count = process_video(video, uuid4(), store, FakeEncoder(), sample_interval_s=1)

    assert count == len(store.frames) == 3
    assert all(frame.captured_at.tzinfo is UTC for frame in store.frames)
    assert all(frame.timings_ms["clip"] == 1.0 for frame in store.frames)
    assert all(frame.keep_reason == "heartbeat" for frame in store.frames)
