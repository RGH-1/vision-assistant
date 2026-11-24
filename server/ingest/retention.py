"""Bounded retention for stored frames and question history (design §7.5)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg


@dataclass(frozen=True)
class RetentionConfig:
    interval_seconds: float
    max_frame_count: int
    max_frame_bytes: int
    question_retention_hours: float


@dataclass(frozen=True)
class RetentionResult:
    frames_deleted: int
    questions_deleted: int


def load_retention_config() -> RetentionConfig:
    """Read the versioned safety limits from config.yaml."""
    # shortcut: PyYAML is already an allowed transitive dependency; use a local import so
    # importing the job during a test does not load every server setting.
    import yaml

    from server.settings import CONFIG_PATH

    values = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["retention"]
    return RetentionConfig(
        interval_seconds=float(values["interval_seconds"]),
        max_frame_count=int(values["max_frame_count"]),
        max_frame_bytes=int(values["max_frame_bytes"]),
        question_retention_hours=float(values["question_retention_hours"]),
    )


def delete_frame_files(data_dir: Path, rows: list[Mapping[str, object]]) -> None:
    """Remove saved media, rejecting a corrupt database path outside ``data_dir``."""
    root = data_dir.resolve()
    for row in rows:
        for key in ("image_path", "thumb_path", "original_path"):
            stored_path = row[key]
            if stored_path is None:
                continue
            path = (data_dir / str(stored_path)).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"stored media path escapes data directory: {stored_path}")
            # A previous partial deletion is harmless: the rows can now be cleaned up on retry.
            path.unlink(missing_ok=True)


def _frame_size(data_dir: Path, row: Mapping[str, object]) -> int:
    root = data_dir.resolve()
    total = 0
    for key in ("image_path", "thumb_path", "original_path"):
        stored_path = row[key]
        if stored_path is None:
            continue
        path = (data_dir / str(stored_path)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"stored media path escapes data directory: {stored_path}")
        if path.is_file():
            total += path.stat().st_size
    return total


class RetentionJob:
    """Delete expired data and enforce frame safety caps from the ingest thread only."""

    def __init__(
        self,
        data_dir: Path,
        connection: psycopg.Connection,
        *,
        retention_minutes: int,
        config: RetentionConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if retention_minutes < 1:
            raise ValueError("retention_minutes must be at least 1")
        self._data_dir = data_dir
        self._connection = connection
        self._retention = timedelta(minutes=retention_minutes)
        self._config = config or load_retention_config()
        self._monotonic = monotonic
        self._now = now
        self._last_run: float | None = None

    def run_if_due(self) -> RetentionResult | None:
        """Run at most once per configured interval; run immediately after startup."""
        current = self._monotonic()
        if self._last_run is not None and current - self._last_run < self._config.interval_seconds:
            return None
        result = self.purge()
        self._last_run = current
        return result

    def purge(self) -> RetentionResult:
        """Delete media before its frame rows, then prune old questions."""
        now = self._now()
        with self._connection.transaction():
            rows = self._connection.execute(
                """
                SELECT id, captured_at, image_path, thumb_path, original_path
                FROM frames ORDER BY captured_at ASC, id ASC
                """
            ).fetchall()
        expired_before = now - self._retention
        selected = {row["id"] for row in rows if row["captured_at"] < expired_before}
        frame_sizes = {row["id"]: _frame_size(self._data_dir, row) for row in rows}
        remaining_count = len(rows) - len(selected)
        remaining_bytes = sum(
            size for frame_id, size in frame_sizes.items() if frame_id not in selected
        )

        # Rows are oldest first, so every cap-driven deletion preserves newer memories.
        for row in rows:
            if (
                remaining_count <= self._config.max_frame_count
                and remaining_bytes <= self._config.max_frame_bytes
            ):
                break
            frame_id = row["id"]
            if frame_id not in selected:
                selected.add(frame_id)
                remaining_count -= 1
                remaining_bytes -= frame_sizes[frame_id]

        frames_deleted = len(selected)
        if selected:
            selected_rows = [row for row in rows if row["id"] in selected]
            delete_frame_files(self._data_dir, selected_rows)
            with self._connection.transaction():
                self._connection.execute("DELETE FROM frames WHERE id = ANY(%s)", (list(selected),))

        with self._connection.transaction():
            cursor = self._connection.execute(
                "DELETE FROM questions WHERE asked_at < %s",
                (now - timedelta(hours=self._config.question_retention_hours),),
            )
            questions_deleted = cursor.rowcount
        return RetentionResult(frames_deleted=frames_deleted, questions_deleted=questions_deleted)
