"""Pydantic models for the REST contract (contracts/openapi.yaml, contracts/events.schema.json)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

ErrorCode = Literal[
    "STT_FAILED",
    "GEMINI_FAILED",
    "NO_STREAM",
    "BAD_AUDIO",
    "BAD_REQUEST",
    "NOT_FOUND",
    "INTERNAL",
]


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    speak: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Status(BaseModel):
    session_id: UUID | None
    stream_connected: bool
    input_fps: float
    analysed_fps: float
    saved_per_min: float
    frames_in_db: int
    retention_min: int
    gpu_mem_mb: float | None
    last_answer_at: datetime | None
    groq_ok: bool
    gemini_ok: bool
