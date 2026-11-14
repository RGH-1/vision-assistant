"""Persist processed ingest frames and notify the dashboard (design §6.3.1, §7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import cv2
import numpy as np
import psycopg

from server.common.db import insert_frame
from server.common.text_norm import normalize_text
from server.ingest.faceblur import FaceBlurrer

MAX_FRAME_SIDE = 1280
THUMBNAIL_SIDE = 320
TEXT_PREVIEW_LENGTH = 160


class EventPublisher(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class StoredDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    area_ratio: float
    position: str


@dataclass(frozen=True)
class StoredOcr:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]
    from_crop: bool = False


@dataclass(frozen=True)
class ProcessedFrame:
    """A frame after the quality, enhancement, blur and OCR stages have run."""

    session_id: UUID
    captured_at: datetime
    image: np.ndarray
    sharpness: float
    brightness: float
    contrast: float
    edge_density: float
    dhash: int
    keep_reason: str
    clip_emb: np.ndarray
    timings_ms: dict[str, float]
    detections: tuple[StoredDetection, ...] = ()
    ocr: tuple[StoredOcr, ...] = ()
    enhanced: bool = False
    enhance_ops: tuple[str, ...] = ()
    original: np.ndarray | None = None
    faces_blurred: int = 0


class FrameStore:
    """Write processed JPEGs and their related database records as one ingest operation."""

    def __init__(
        self,
        data_dir: Path,
        connection: psycopg.Connection,
        face_blurrer: FaceBlurrer,
        hub: EventPublisher | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._connection = connection
        self._face_blurrer = face_blurrer
        self._hub = hub

    def store(self, frame: ProcessedFrame) -> int:
        """Save a post-blur frame, then publish after its database transaction commits."""
        if frame.enhanced and frame.original is None:
            raise ValueError("enhanced frames must include their pre-enhancement original")

        blurred = self._face_blurrer.blur(frame.image, frame.timings_ms)
        image = _resize_long_side(blurred.image, MAX_FRAME_SIDE)
        original = (
            _resize_long_side(self._face_blurrer.blur(frame.original).image, MAX_FRAME_SIDE)
            if frame.original is not None
            else None
        )
        thumbnail = _resize_long_side(image, THUMBNAIL_SIDE)
        paths = _frame_paths(has_original=original is not None)
        written_paths: list[Path] = []
        try:
            _write_jpeg(self._data_dir / paths["image_path"], image)
            written_paths.append(self._data_dir / paths["image_path"])
            _write_jpeg(self._data_dir / paths["thumb_path"], thumbnail)
            written_paths.append(self._data_dir / paths["thumb_path"])
            if original is not None:
                _write_jpeg(self._data_dir / paths["original_path"], original)
                written_paths.append(self._data_dir / paths["original_path"])

            scene_text = build_scene_text(frame.detections, frame.ocr)
            with self._connection.transaction():
                frame_id = insert_frame(
                    self._connection,
                    {
                        **paths,
                        "session_id": frame.session_id,
                        "captured_at": frame.captured_at,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "sharpness": frame.sharpness,
                        "brightness": frame.brightness,
                        "contrast": frame.contrast,
                        "edge_density": frame.edge_density,
                        "enhanced": frame.enhanced,
                        "enhance_ops": list(frame.enhance_ops),
                        "dhash": frame.dhash,
                        "keep_reason": frame.keep_reason,
                        "faces_blurred": blurred.faces_blurred,
                        "scene_text": scene_text,
                        "clip_emb": frame.clip_emb,
                        "timings_ms": frame.timings_ms,
                    },
                )
                _insert_detections(self._connection, frame_id, frame.detections)
                _insert_ocr(self._connection, frame_id, frame.ocr)
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
            raise

        if self._hub is not None:
            self._hub.publish(
                {
                    "type": "frame_stored",
                    "frame_id": frame_id,
                    "captured_at": frame.captured_at,
                    "thumb_url": f"/media/{paths['thumb_path']}",
                    "labels": [detection.label for detection in frame.detections],
                    "text_preview": " ".join(ocr.text for ocr in frame.ocr)[:TEXT_PREVIEW_LENGTH],
                    "keep_reason": frame.keep_reason,
                }
            )
        return frame_id


def build_scene_text(detections: tuple[StoredDetection, ...], ocr: tuple[StoredOcr, ...]) -> str:
    """Build normalized object and OCR text for full-text search."""
    objects = [
        normalize_text(f"{detection.label} {detection.position}") for detection in detections
    ]
    text = [normalize_text(item.text) for item in ocr]
    return " | ".join(part for part in ("; ".join(objects), " ".join(text)) if part)


def _frame_paths(*, has_original: bool) -> dict[str, str | None]:
    file_name = f"{uuid4()}.jpg"
    return {
        "image_path": f"frames/{file_name}",
        "thumb_path": f"thumbs/{file_name}",
        "original_path": f"originals/{file_name}" if has_original else None,
    }


def _resize_long_side(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= maximum:
        return image
    scale = maximum / longest
    return cv2.resize(image, (round(width * scale), round(height * scale)), cv2.INTER_AREA)


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not encoded:
        raise ValueError(f"could not encode JPEG: {path}")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data.tobytes())
    temporary.replace(path)


def _insert_detections(
    connection: psycopg.Connection,
    frame_id: int,
    detections: tuple[StoredDetection, ...],
) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    frame_id,
                    item.label,
                    item.confidence,
                    list(item.bbox),
                    item.area_ratio,
                    item.position,
                )
                for item in detections
            ],
        )


def _insert_ocr(
    connection: psycopg.Connection,
    frame_id: int,
    ocr: tuple[StoredOcr, ...],
) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox, from_crop)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    frame_id,
                    item.text,
                    normalize_text(item.text),
                    item.confidence,
                    list(item.bbox),
                    item.from_crop,
                )
                for item in ocr
            ],
        )
