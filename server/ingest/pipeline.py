"""Live ingest: turn each sampled camera frame into a stored memory frame (design §6.3.1)."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import psycopg

from server.common.db import open_ingest_connection
from server.ingest.detection import YoloeDetector, load_object_detection_config
from server.ingest.embeddings import ClipImageEncoder
from server.ingest.enhance import EnhancementResult, FrameEnhancer, load_frame_enhancement_config
from server.ingest.faceblur import FaceBlurrer
from server.ingest.keep import FrameKeepCandidate, FrameKeepDecider, load_frame_keep_config
from server.ingest.novelty import FrameNoveltySelector, load_frame_novelty_config
from server.ingest.retention import RetentionJob
from server.ingest.selection import FrameMetrics, FrameSelector, load_frame_selection_config
from server.ingest.store import EventPublisher, FrameStore, ProcessedFrame


@dataclass(frozen=True)
class LiveCandidate(FrameKeepCandidate):
    """A keep candidate, plus what FrameStore needs if it wins its 1-second window."""

    metrics: FrameMetrics
    enhancement: EnhancementResult
    dhash: int
    timings_ms: dict[str, float]


def load_models() -> tuple[Any, Any]:
    return YoloeDetector(load_object_detection_config()), ClipImageEncoder()


def wall_clock(captured_at: float) -> datetime:
    """The worker stamps frames with time.monotonic(); the database needs a real time."""
    return datetime.now(UTC) - timedelta(seconds=time.monotonic() - captured_at)


class FramePipeline:
    """Called only from the ingest thread: it owns the GPU models and a sync connection (§5.3)."""

    def __init__(
        self,
        data_dir: Path,
        face_blurrer: FaceBlurrer,
        hub: EventPublisher | None = None,
        *,
        models: Callable[[], tuple[Any, Any]] = load_models,
        connect: Callable[[], psycopg.Connection] = open_ingest_connection,
    ) -> None:
        self._data_dir = data_dir
        self._face_blurrer = face_blurrer
        self._hub = hub
        self._models = models
        self._connect = connect
        self._selector = FrameSelector(load_frame_selection_config())
        self._enhancer = FrameEnhancer(load_frame_enhancement_config())
        self._novelty = FrameNoveltySelector(load_frame_novelty_config())
        keep_config = load_frame_keep_config()
        self._window_s = keep_config.window_seconds
        self._keep = FrameKeepDecider(keep_config)
        self._window: int | None = None
        self._detector: Any = None
        self._clip: Any = None
        self._connection: psycopg.Connection | None = None
        self._store: FrameStore | None = None
        self._retention: RetentionJob | None = None

    def process(self, frame: np.ndarray, captured_at: float) -> None:
        if self._detector is None:
            # Loaded on the first frame, so the models load in the ingest thread (§5.3).
            self._detector, self._clip = self._models()
            self._connection = self._connect()
            self._store = FrameStore(
                self._data_dir, self._connection, self._face_blurrer, self._hub
            )
            # Settings has loaded RETENTION_MINUTES before the ingest thread starts.
            self._retention = RetentionJob(
                self._data_dir,
                self._connection,
                retention_minutes=int(os.environ.get("RETENTION_MINUTES", "30")),
            )

        self._retention.run_if_due()

        window = math.floor(captured_at / self._window_s)
        if window != self._window:
            # Save the last window's winner first, so this frame's dHash is compared with it.
            for kept in self._keep.flush():
                self._save(kept.candidate, kept.edge_density)
            self._window = window
        # Its window was just flushed, so consider() never has a winner to hand back here.
        self._keep.consider(self._candidate(frame, captured_at), captured_at)

    def _candidate(self, frame: np.ndarray, captured_at: float) -> LiveCandidate | None:
        selection = self._selector.select(frame)
        if not selection.accepted:
            return None
        enhancement = self._enhancer.enhance(
            frame,
            brightness=selection.metrics.brightness,
            low_contrast="low_contrast" in selection.enhancement_reasons,
            slightly_soft=selection.unsharp_applied,
        )
        timings_ms: dict[str, float] = {}
        detections = self._detector.detect(enhancement.image, timings_ms)
        # YOLOE runs before the dHash check (§6.3.1 draws it after), because the "new objects"
        # rule in §6.3.2 needs this frame's labels.
        labels = [detection.label for detection in detections]
        novelty = self._novelty.select(enhancement.image, labels, captured_at)
        if not novelty.keep:
            return None
        return LiveCandidate(
            image=enhancement.image,
            captured_at=captured_at,
            sharpness=selection.metrics.sharpness,
            detections=detections,
            keep_reason=novelty.keep_reason,
            metrics=selection.metrics,
            enhancement=enhancement,
            dhash=novelty.dhash,
            timings_ms=timings_ms,
        )

    def _save(self, candidate: LiveCandidate, edge_density: float) -> None:
        session_id = self._open_session()
        if session_id is None:
            return  # no session means no consent, so nothing is remembered (§9)
        timings_ms = dict(candidate.timings_ms)
        # CLIP sees the blurred frame, like every stage after the face blur in §6.3.1.
        clip_emb = self._clip.encode(self._face_blurrer.blur(candidate.image).image, timings_ms)
        enhancement = candidate.enhancement
        self._store.store(
            ProcessedFrame(
                session_id=session_id,
                captured_at=wall_clock(candidate.captured_at),
                image=candidate.image,  # FrameStore blurs it again, and counts the faces
                sharpness=candidate.metrics.sharpness,
                brightness=candidate.metrics.brightness,
                contrast=candidate.metrics.contrast,
                edge_density=edge_density,
                dhash=candidate.dhash,
                keep_reason=candidate.keep_reason,
                clip_emb=clip_emb,
                timings_ms=timings_ms,
                detections=tuple(d.as_stored_detection() for d in candidate.detections),
                # shortcut: no OCR until VA-48, so stored frames have no text and past-text
                # questions find nothing; VA-48 adds EasyOCR here.
                enhanced=enhancement.enhanced,
                enhance_ops=enhancement.enhance_ops,
                original=enhancement.original,
            )
        )
        labels = [detection.label for detection in candidate.detections]
        self._novelty.confirm_saved(candidate.dhash, labels, candidate.captured_at)

    def _open_session(self) -> UUID | None:
        # Its own transaction, so FrameStore's transaction commits instead of becoming a savepoint.
        with self._connection.transaction():
            row = self._connection.execute(
                "SELECT id FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return row["id"] if row else None
