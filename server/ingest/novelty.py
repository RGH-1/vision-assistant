"""dHash novelty and heartbeat decisions for accepted ingest frames (§6.3.2)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from server.settings import CONFIG_PATH

HASH_MASK = 2**64 - 1


@dataclass(frozen=True)
class FrameNoveltyConfig:
    new_distance: int
    duplicate_distance: int
    heartbeat_seconds: float


@dataclass(frozen=True)
class NoveltyDecision:
    """The value and keep reason to persist with one candidate frame."""

    keep: bool
    dhash: int
    distance: int | None
    keep_reason: str | None


def load_frame_novelty_config(path: Path = CONFIG_PATH) -> FrameNoveltyConfig:
    """Read dHash and heartbeat thresholds from the committed configuration."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["frame_novelty"]
    return FrameNoveltyConfig(**values)


def dhash(frame: np.ndarray) -> int:
    """Return a 64-bit difference hash in the signed range PostgreSQL BIGINT accepts."""
    grayscale = _as_grayscale(frame)
    small = cv2.resize(grayscale, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value if value < 2**63 else value - 2**64


def hamming_distance(first: int, second: int) -> int:
    """Count changed bits while treating persisted signed BIGINT values as 64-bit hashes."""
    return ((first & HASH_MASK) ^ (second & HASH_MASK)).bit_count()


class FrameNoveltySelector:
    """Keep first, novel, newly-labelled, or heartbeat frames; drop other duplicates."""

    def __init__(self, config: FrameNoveltyConfig) -> None:
        if config.duplicate_distance < 0:
            raise ValueError("duplicate_distance must not be negative")
        if config.new_distance <= config.duplicate_distance:
            raise ValueError("new_distance must exceed duplicate_distance")
        if config.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._config = config
        self._last_saved_hash: int | None = None
        self._last_saved_labels: frozenset[str] = frozenset()
        self._last_saved_at: float | None = None

    def select(
        self, frame: np.ndarray, labels: Iterable[str], captured_at: float
    ) -> NoveltyDecision:
        """Hash a quality-approved frame and make its stateful retention decision."""
        return self.select_hash(dhash(frame), labels, captured_at)

    def select_hash(
        self,
        frame_hash: int,
        labels: Iterable[str],
        captured_at: float,
    ) -> NoveltyDecision:
        """Make a decision from an already-computed hash for pipeline composition and tests."""
        label_set = frozenset(labels)
        if self._last_saved_hash is None:
            return self._keep(frame_hash, label_set, captured_at, None, "novel")

        distance = hamming_distance(frame_hash, self._last_saved_hash)
        if distance >= self._config.new_distance:
            return self._keep(frame_hash, label_set, captured_at, distance, "novel")
        if label_set - self._last_saved_labels:
            return self._keep(frame_hash, label_set, captured_at, distance, "new_objects")
        if captured_at - self._last_saved_at >= self._config.heartbeat_seconds:
            return self._keep(frame_hash, label_set, captured_at, distance, "heartbeat")
        return NoveltyDecision(False, frame_hash, distance, None)

    def _keep(
        self,
        frame_hash: int,
        labels: frozenset[str],
        captured_at: float,
        distance: int | None,
        keep_reason: str,
    ) -> NoveltyDecision:
        self._last_saved_hash = frame_hash
        self._last_saved_labels = labels
        self._last_saved_at = captured_at
        return NoveltyDecision(True, frame_hash, distance, keep_reason)


def _as_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("dHash requires a grayscale or BGR image")
