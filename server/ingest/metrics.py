"""Thread-safe rolling ingest measurements for dashboard status (§8.2, §11)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestMetricsSnapshot:
    input_fps: float
    analysed_fps: float
    saved_per_min: float
    gpu_mem_mb: float | None


class IngestMetrics:
    """Keep recent event timestamps; workers and the web loop may read concurrently."""

    def __init__(
        self,
        *,
        rate_window_s: float = 2.0,
        saved_window_s: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_window_s <= 0 or saved_window_s <= 0:
            raise ValueError("metric windows must be positive")
        self._rate_window_s = rate_window_s
        self._saved_window_s = saved_window_s
        self._monotonic = monotonic
        self._input: deque[float] = deque()
        self._analysed: deque[float] = deque()
        self._saved: deque[float] = deque()
        self._gpu_mem_mb: float | None = None
        self._lock = threading.Lock()

    def record_input(self) -> None:
        self._record(self._input)

    def record_analysed(self) -> None:
        self._record(self._analysed)

    def record_saved(self) -> None:
        self._record(self._saved)

    def record_gpu_mem(self, allocated_bytes: int | None) -> None:
        with self._lock:
            self._gpu_mem_mb = (
                allocated_bytes / (1024 * 1024) if allocated_bytes is not None else None
            )

    def snapshot(self) -> IngestMetricsSnapshot:
        now = self._monotonic()
        with self._lock:
            self._trim(self._input, now - self._rate_window_s)
            self._trim(self._analysed, now - self._rate_window_s)
            self._trim(self._saved, now - self._saved_window_s)
            return IngestMetricsSnapshot(
                input_fps=len(self._input) / self._rate_window_s,
                analysed_fps=len(self._analysed) / self._rate_window_s,
                saved_per_min=len(self._saved) * 60 / self._saved_window_s,
                gpu_mem_mb=self._gpu_mem_mb,
            )

    def _record(self, events: deque[float]) -> None:
        with self._lock:
            events.append(self._monotonic())

    @staticmethod
    def _trim(events: deque[float], before: float) -> None:
        while events and events[0] < before:
            events.popleft()
