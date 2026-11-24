"""
Pydantic models for API and WebSocket payloads, not database rows.

Source of truth: contracts/openapi.yaml and contracts/events.schema.json.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

# Shared contract types from events.schema.json, $defs.
Lang = Literal["en", "ar", "fr"]
Intent = Literal[
    "LIVE_SCENE",
    "READ_TEXT_NOW",
    "READ_TEXT_PAST",
    "MEMORY",
    "FIND_OBJECT",
    "IDENTITY",
    "OUT_OF_SCOPE",
]
Confidence = Literal["high", "medium", "low", "none"]
KeepReason = Literal["novel", "new_objects", "text", "heartbeat"]
AlertKind = Literal["door_ahead", "stairs_ahead", "obstacle_ahead", "vehicle_near"]
ErrorCode = Literal[
    "STT_FAILED",
    "GEMINI_FAILED",
    "NO_STREAM",
    "BAD_AUDIO",
    "BAD_REQUEST",
    "NOT_FOUND",
    "INTERNAL",
]
TimingsMs = dict[str, float]
BboxCoordinate = Annotated[float, Field(ge=0, le=1)]
Bbox = Annotated[list[BboxCoordinate], Field(min_length=4, max_length=4)]


class ContractModel(BaseModel):
    """Reject undocumented fields in API and event payloads."""

    model_config = ConfigDict(extra="forbid")


# --- REST models, contracts/openapi.yaml ---
class ErrorDetail(ContractModel):
    code: ErrorCode
    message: str
    speak: str


class Error(ContractModel):
    error: ErrorDetail


class SessionCreate(ContractModel):
    app_lang: Lang
    consent_at: AwareDatetime
    consent_ver: str


class SessionCreated(ContractModel):
    session_id: UUID
    whip_url: str
    ws_url: str


class AskTextRequest(ContractModel):
    """Inline JSON request schema of POST /v1/ask_text."""

    session_id: UUID
    text: str = Field(min_length=1)
    lang: Lang


class AskEvidence(ContractModel):
    """Live-buffer answers may have null frame_id and thumb_url."""

    frame_id: int | None
    seconds_ago: float
    thumb_url: str | None
    score: float


class AskResponse(ContractModel):
    question_id: int
    transcript: str
    lang: Lang
    intent: Intent
    answer: str
    confidence: Confidence
    abstained: bool
    evidence: list[AskEvidence]
    timings_ms: TimingsMs


class FrameSummary(ContractModel):
    frame_id: int
    captured_at: AwareDatetime
    thumb_url: str
    labels: list[str]
    text_preview: str
    keep_reason: KeepReason
    score: float | None  # required in JSON, even when null


class FrameDetection(ContractModel):
    label: str
    confidence: float
    bbox: Bbox
    area_ratio: float
    position: Literal["left", "center", "right"]


class FrameOCR(ContractModel):
    text: str
    confidence: float
    bbox: Bbox
    from_crop: bool


class FrameDetail(ContractModel):
    frame_id: int
    session_id: UUID
    captured_at: AwareDatetime
    image_url: str
    thumb_url: str
    original_url: str | None
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    edge_density: float
    enhanced: bool
    enhance_ops: list[str]
    keep_reason: KeepReason
    faces_blurred: int
    scene_text: str
    timings_ms: TimingsMs
    detections: list[FrameDetection]
    ocr: list[FrameOCR]
    # Intentionally no dhash or clip_emb: those are internal DB fields.


# --- Shared objects, contracts/events.schema.json ---
class Alert(ContractModel):
    alert_id: int
    kind: AlertKind
    message: str
    lang: Lang
    frame_id: int | None  # required but nullable
    thumb_url: str | None  # required but nullable
    created_at: AwareDatetime


class Status(ContractModel):
    session_id: UUID | None
    stream_connected: bool
    input_fps: float
    analysed_fps: float
    saved_per_min: float
    frames_in_db: int
    retention_min: int
    gpu_mem_mb: float | None
    last_answer_at: AwareDatetime | None
    groq_ok: bool
    gemini_ok: bool


class QuestionEvidenceParts(ContractModel):
    clip: float
    text: float
    obj: float
    rec: float
    q: float


class QuestionEvidence(ContractModel):
    """Evidence records refer only to frames saved in PostgreSQL."""

    frame_id: int
    rank: int
    score: float
    parts: QuestionEvidenceParts
    thumb_url: str


class QuestionRecord(ContractModel):
    question_id: int
    session_id: UUID
    asked_at: AwareDatetime
    input_mode: Literal["voice", "text"]
    stt_engine: Literal["groq", "local"] | None
    transcript: str
    lang: Lang
    intent: Intent
    t_from: AwareDatetime | None
    t_to: AwareDatetime | None
    answer: str
    confidence: Confidence
    abstained: bool
    safety_rewrite: bool
    model: str | None
    timings_ms: TimingsMs
    evidence: list[QuestionEvidence]


# --- Flat WebSocket messages: one top-level type field ---
# Required Literal fields intentionally have NO defaults.
class HelloMessage(ContractModel):
    type: Literal["hello"]
    session_id: UUID
    app_lang: Lang


class AlertMessage(Alert):
    type: Literal["alert"]


class StatusMessage(Status):
    type: Literal["status"]


class FrameStoredMessage(ContractModel):
    type: Literal["frame_stored"]
    frame_id: int
    captured_at: AwareDatetime
    thumb_url: str
    labels: list[str]
    text_preview: str
    keep_reason: KeepReason


class QueryAnsweredMessage(QuestionRecord):
    type: Literal["query_answered"]


class PingMessage(ContractModel):
    type: Literal["ping"]


class PongMessage(ContractModel):
    type: Literal["pong"]


WebSocketMessage = Annotated[
    HelloMessage
    | AlertMessage
    | StatusMessage
    | FrameStoredMessage
    | QueryAnsweredMessage
    | PingMessage
    | PongMessage,
    Field(discriminator="type"),
]
