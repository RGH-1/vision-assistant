import threading

import cv2
import numpy as np
import pytest

from server.ingest.live_buffer import LiveBuffer


def sharp_frame(value: int) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (value, value, value), thickness=-1)
    return frame


def test_buffer_encodes_jpegs_and_returns_sharpest_of_five() -> None:
    buffer = LiveBuffer()
    for captured_at, value in enumerate((5, 15, 30, 60, 10, 90)):
        buffer.add(sharp_frame(value), float(captured_at), [{"label": "door"}])

    selected = buffer.get_sharpest()

    assert selected is not None
    captured_at, jpeg_bytes, sharpness, detections = selected
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert captured_at == 5.0
    assert sharpness > 0
    assert decoded.shape[1] == 960
    assert detections == ({"label": "door"},)


def test_concurrent_writer_and_reader_do_not_error() -> None:
    buffer = LiveBuffer()
    errors: list[Exception] = []
    finished = threading.Event()

    def writer() -> None:
        try:
            for captured_at in range(100):
                buffer.add(sharp_frame(captured_at), float(captured_at))
        except Exception as error:  # pragma: no cover - asserted by the main thread
            errors.append(error)
        finally:
            finished.set()

    def reader() -> None:
        try:
            while not finished.is_set():
                buffer.get_sharpest()
                buffer.snapshot()
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


def test_clear_removes_all_live_frames() -> None:
    buffer = LiveBuffer()
    buffer.add(sharp_frame(1), 1.0)

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.get_sharpest() is None


def test_get_sharpest_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        LiveBuffer().get_sharpest(0)



def test_get_sharpest_selects_from_last_five() -> None:
    buffer = LiveBuffer()

    # Create a plain black image with very low sharpness.
    flat = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create a checkerboard image with high sharpness.
    pattern = np.indices((100, 100)).sum(axis=0) % 2

    sharp = np.repeat(
        (pattern * 255).astype(np.uint8)[:, :, None],
        3,
        axis=2,
    )

    # Add two old sharp frames.
    buffer.add(sharp, 0.0)
    buffer.add(sharp, 1.0)

    # Add the five most recent frames.
    buffer.add(flat, 2.0)
    buffer.add(flat, 3.0)
    buffer.add(sharp, 4.0)
    buffer.add(flat, 5.0)
    buffer.add(flat, 6.0)

    # Only the last five frames should be considered.
    result = buffer.get_sharpest(n=5)

    assert result is not None
    assert result[0] == 4.0
    assert result[2] > 0
