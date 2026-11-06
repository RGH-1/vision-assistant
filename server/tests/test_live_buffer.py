import threading

import cv2
import numpy as np
import pytest

from server.ingest.live_buffer import LiveBuffer


def test_buffer_encodes_960_pixel_jpegs_and_returns_sharpest_of_five() -> None:
    buffer = LiveBuffer()
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    for timestamp, sharpness in enumerate((10.0, 20.0, 30.0, 40.0, 5.0, 80.0)):
        buffer.add(float(timestamp), frame, sharpness, [{"label": "door"}])

    selected = buffer.get_sharpest()

    assert selected is not None
    timestamp, jpeg_bytes, sharpness, detections = selected
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert timestamp == 5.0
    assert sharpness == 80.0
    assert decoded.shape[1] == 960
    assert detections == ({"label": "door"},)


def test_concurrent_writer_and_reader_do_not_error() -> None:
    buffer = LiveBuffer()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    errors: list[Exception] = []
    finished = threading.Event()

    def writer() -> None:
        try:
            for timestamp in range(100):
                buffer.add(float(timestamp), frame, float(timestamp), [])
        except Exception as error:  # pragma: no cover - asserted by the main thread
            errors.append(error)
        finally:
            finished.set()

    def reader() -> None:
        try:
            while not finished.is_set():
                buffer.get_sharpest()
        except Exception as error:  # pragma: no cover - asserted by the main thread
            errors.append(error)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    reader_thread.join()

    assert errors == []
    assert len(buffer) == 10
    assert buffer.get_sharpest() is not None


def test_clear_removes_all_live_frames() -> None:
    buffer = LiveBuffer()
    buffer.add(1.0, np.zeros((540, 960, 3), dtype=np.uint8), 1.0, [])

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.get_sharpest() is None


def test_get_sharpest_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        LiveBuffer().get_sharpest(0)
