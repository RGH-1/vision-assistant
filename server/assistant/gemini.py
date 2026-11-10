"""Gemini answers for questions about the live camera (design §6.5.4, Appendix A)."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from server.common.schemas import Lang
from server.ingest.live_buffer import Detection
from server.settings import Settings

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
LANGUAGE_NAMES: dict[Lang, str] = {"en": "English", "ar": "Arabic", "fr": "French"}


class GeminiAnswer(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    used_frames: list[str]
    reason_if_unsure: str


def frame_label(seconds_ago: float, detections: Sequence[Detection]) -> str:
    objects = ", ".join(
        f"{d['label']} ({d['position']}, {d['confidence']:.2f})" for d in detections
    )
    # The live buffer has no OCR, so there is never text here; READ_TEXT_NOW adds it later.
    return f"Frame A — {seconds_ago:.0f} seconds ago — objects: {objects or 'none'} — text: none"


async def describe_live_scene(
    jpeg: bytes,
    seconds_ago: float,
    detections: Sequence[Detection],
    question: str,
    lang: Lang,
    settings: Settings,
) -> GeminiAnswer:
    """Raises on any failure (no key, quota, timeout, invalid JSON); the caller handles it."""
    # shortcut: one model, one frame, no Flash-Lite retry; VA-72 adds the full generator.
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            "Question type: LIVE_SCENE (time window: now)\n"
            f"Question: {question}\n"
            f"{frame_label(seconds_ago, detections)}",
        ],
        config=types.GenerateContentConfig(
            # replace, not format: the prompt's JSON example has braces of its own.
            system_instruction=SYSTEM_PROMPT.replace("{language}", LANGUAGE_NAMES[lang]),
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GeminiAnswer,
            # No retry_options, so the SDK doesn't retry and a failure is reported at once.
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
        ),
    )
    return GeminiAnswer.model_validate_json(response.text)
