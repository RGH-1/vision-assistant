"""EasyOCR on saved frames and pre-processed sign crops (§6.3.3, §6.3.5)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from server.ingest.keep import SIGN_LABELS
from server.ingest.store import StoredDetection, StoredOcr
from server.settings import CONFIG_PATH


class EasyOcrReader(Protocol):
    def readtext(self, image: np.ndarray, **kwargs: Any) -> list[tuple[Any, str, float]]: ...


EasyOcrFactory = Callable[[list[str], bool], EasyOcrReader]


@dataclass(frozen=True)
class OcrConfig:
    full_frame_max_side: int
    crop_scale: int
    clahe_clip_limit: float
    clahe_tile_size: int
    denoise_h: int
    confidence_min: float
    min_text_length: int
    merge_iou: float


def load_ocr_config(path: Path = CONFIG_PATH) -> OcrConfig:
    """Read the committed OCR thresholds and crop-processing parameters."""
    return OcrConfig(**yaml.safe_load(path.read_text(encoding="utf-8"))["ocr"])


class FrameOcr:
    """One Arabic-and-English EasyOCR reader, constructed in the ingest thread."""

    def __init__(self, config: OcrConfig, *, reader_factory: EasyOcrFactory | None = None) -> None:
        if config.full_frame_max_side <= 0 or config.crop_scale <= 0:
            raise ValueError("OCR image sizes must be positive")
        if config.clahe_clip_limit <= 0 or config.clahe_tile_size <= 0 or config.denoise_h < 0:
            raise ValueError("OCR crop parameters must be non-negative")
        if not 0 <= config.confidence_min <= 1 or config.min_text_length < 1:
            raise ValueError("OCR confidence and minimum text length are invalid")
        if not 0 <= config.merge_iou <= 1:
            raise ValueError("OCR merge_iou must be in [0, 1]")
        factory = reader_factory or _easyocr_reader
        self._config = config
        # EasyOCR supports Arabic together with English in one reader (§6.3.5).
        self._reader = factory(["ar", "en"], gpu=True)

    def read(
        self,
        frame: np.ndarray,
        detections: Sequence[StoredDetection],
        timings_ms: dict[str, float],
    ) -> tuple[StoredOcr, ...]:
        """Read a full frame and sign crops, retaining only useful de-duplicated lines."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("OCR requires a BGR image with three channels")
        started_at = perf_counter()
        image = _resize_long_side(frame, self._config.full_frame_max_side)
        findings = self._read_full_frame(image)
        findings.extend(self._read_sign_crops(image, detections))
        timings_ms["ocr"] = (perf_counter() - started_at) * 1000
        return _merge_overlaps(findings, self._config.merge_iou)

    def _read_full_frame(self, image: np.ndarray) -> list[StoredOcr]:
        height, width = image.shape[:2]
        return [
            ocr
            for result in self._reader.readtext(image, detail=1, paragraph=False)
            if (
                ocr := _stored_ocr(
                    result,
                    width,
                    height,
                    from_crop=False,
                    confidence_min=self._config.confidence_min,
                    min_text_length=self._config.min_text_length,
                )
            )
            is not None
        ]

    def _read_sign_crops(
        self, image: np.ndarray, detections: Sequence[StoredDetection]
    ) -> list[StoredOcr]:
        findings: list[StoredOcr] = []
        height, width = image.shape[:2]
        for detection in detections:
            if detection.label not in SIGN_LABELS:
                continue
            left, top, right, bottom = _pixel_box(detection.bbox, width, height)
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue
            processed = _prepare_sign_crop(crop, self._config)
            crop_height, crop_width = crop.shape[:2]
            processed_height, processed_width = processed.shape[:2]
            for result in self._reader.readtext(processed, detail=1, paragraph=False):
                ocr = _stored_ocr(
                    result,
                    processed_width,
                    processed_height,
                    from_crop=True,
                    offset=(left, top),
                    scale=(processed_width / crop_width, processed_height / crop_height),
                    frame_size=(width, height),
                    confidence_min=self._config.confidence_min,
                    min_text_length=self._config.min_text_length,
                )
                if ocr is not None:
                    findings.append(ocr)
        return findings


def _easyocr_reader(languages: list[str], gpu: bool) -> EasyOcrReader:
    """Delay the heavy import and model loading until the ingest thread first runs."""
    import easyocr

    return easyocr.Reader(languages, gpu=gpu)


def _resize_long_side(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(width, height)
    if longest <= maximum:
        return image
    scale = maximum / longest
    return cv2.resize(image, (round(width * scale), round(height * scale)), cv2.INTER_AREA)


def _prepare_sign_crop(image: np.ndarray, config: OcrConfig) -> np.ndarray:
    """Upscale and normalize a sign crop before EasyOCR sees its small lettering."""
    enlarged = cv2.resize(
        image,
        None,
        fx=config.crop_scale,
        fy=config.crop_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_size, config.clahe_tile_size),
    )
    corrected = cv2.cvtColor(
        cv2.merge((clahe.apply(lightness), a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    return cv2.fastNlMeansDenoisingColored(
        corrected,
        None,
        config.denoise_h,
        config.denoise_h,
        config.clahe_tile_size,
        config.clahe_tile_size * 3,
    )


def _stored_ocr(
    result: tuple[Any, str, float],
    width: int,
    height: int,
    *,
    from_crop: bool,
    offset: tuple[int, int] = (0, 0),
    scale: tuple[float, float] = (1, 1),
    frame_size: tuple[int, int] | None = None,
    confidence_min: float,
    min_text_length: int,
) -> StoredOcr | None:
    polygon, text, confidence = result
    value = str(text).strip()
    score = float(confidence)
    if score < confidence_min or len(value) < min_text_length:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return None
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    if frame_size is None:
        frame_width, frame_height = width, height
        x0, y0 = 0, 0
        scale_x, scale_y = 1.0, 1.0
    else:
        frame_width, frame_height = frame_size
        x0, y0 = offset
        scale_x, scale_y = scale
    box = (
        float(np.clip((x0 + left / scale_x) / frame_width, 0, 1)),
        float(np.clip((y0 + top / scale_y) / frame_height, 0, 1)),
        float(np.clip((x0 + right / scale_x) / frame_width, 0, 1)),
        float(np.clip((y0 + bottom / scale_y) / frame_height, 0, 1)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return StoredOcr(value, score, box, from_crop)


def _pixel_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, round(left * width)),
        max(0, round(top * height)),
        min(width, round(right * width)),
        min(height, round(bottom * height)),
    )


def _merge_overlaps(findings: list[StoredOcr], threshold: float) -> tuple[StoredOcr, ...]:
    kept: list[StoredOcr] = []
    for finding in sorted(findings, key=lambda item: item.confidence, reverse=True):
        if all(_iou(finding.bbox, previous.bbox) <= threshold for previous in kept):
            kept.append(finding)
    return tuple(kept)


def _iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0
