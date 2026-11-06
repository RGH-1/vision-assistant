import numpy as np

from server.ingest.live_buffer import LiveBuffer


def test_live_buffer_keeps_ten_jpegs_and_returns_a_snapshot() -> None:
    buffer = LiveBuffer()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    for captured_at in range(12):
        buffer.add(frame, float(captured_at))

    snapshot = buffer.snapshot()

    assert [entry.captured_at for entry in snapshot] == list(map(float, range(2, 12)))
    assert all(entry.jpeg_bytes.startswith(b"\xff\xd8") for entry in snapshot)
    assert all(entry.detections == () for entry in snapshot)
