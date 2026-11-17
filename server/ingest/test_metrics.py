"""Rolling measurements reported by the ingest status event (§8.2, §11)."""

from server.ingest.metrics import IngestMetrics


def test_snapshot_reports_rates_and_gpu_memory() -> None:
    now = [10.0]
    metrics = IngestMetrics(rate_window_s=2, saved_window_s=60, monotonic=lambda: now[0])

    for _ in range(4):
        metrics.record_input()
    for _ in range(2):
        metrics.record_analysed()
    metrics.record_saved()
    metrics.record_gpu_mem(3 * 1024 * 1024)

    snapshot = metrics.snapshot()

    assert snapshot.input_fps == 2
    assert snapshot.analysed_fps == 1
    assert snapshot.saved_per_min == 1
    assert snapshot.gpu_mem_mb == 3


def test_rates_expire_after_their_rolling_windows() -> None:
    now = [0.0]
    metrics = IngestMetrics(rate_window_s=2, saved_window_s=60, monotonic=lambda: now[0])
    metrics.record_input()
    metrics.record_analysed()
    metrics.record_saved()

    now[0] = 2.1
    first = metrics.snapshot()
    now[0] = 60.1
    second = metrics.snapshot()

    assert (first.input_fps, first.analysed_fps, first.saved_per_min) == (0, 0, 1)
    assert (second.input_fps, second.analysed_fps, second.saved_per_min) == (0, 0, 0)
