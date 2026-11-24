"""Gemini answers from up to three labelled frames (design §6.5.4, Appendix A)."""

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from server.common.schemas import Intent, Lang
from server.ingest.live_buffer import Detection
from server.settings import Settings

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
LANGUAGE_NAMES: dict[Lang, str] = {"en": "English", "ar": "Arabic", "fr": "French"}
# A low-confidence answer starts with these words, the same as the gate's "I'm not sure" (§6.5.5).
NOT_SURE: dict[Lang, str] = {
    "en": "I'm not sure",
    "ar": "لست متأكدًا",
    "fr": "Je ne suis pas certain",
}


class GeminiAnswer(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    used_frames: list[str]
    reason_if_unsure: str


@dataclass(frozen=True)
class Frame:
    """One image for Gemini, with what ingest found in it."""

    jpeg: bytes
    seconds_ago: float
    detections: Sequence[Detection] = ()
    texts: Sequence[str] = ()
    frame_id: int | None = None  # None for a live frame, which isn't stored


def frame_label(letter: str, frame: Frame) -> str:
    objects = ", ".join(
        f"{d['label']} ({d['position']}, {d['confidence']:.2f})" for d in frame.detections
    )
    texts = ", ".join(f'"{text}"' for text in frame.texts)
    return (
        f"Frame {letter} — {frame.seconds_ago:.0f} seconds ago — "
        f"objects: {objects or 'none'} — text: {texts or 'none'}"
    )


def time_window(t_from: datetime | None, t_to: datetime | None, asked_at: datetime) -> str:
    """The question's time window in words, as Appendix A's user message shows it."""
    if t_from is None or t_to is None:
        return "now"
    start = round((asked_at - t_from).total_seconds())
    end = round((asked_at - t_to).total_seconds())
    return f"last {start} seconds" if end <= 0 else f"{start} to {end} seconds ago"


def load_jpeg(path: Path, max_side: int) -> bytes | None:
    """A stored frame, at most max_side px on its long side; None if retention deleted it."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image)[1].tobytes()


def cache_file(
    settings: Settings, system: str, request: str, frames: Sequence[Frame]
) -> Path | None:
    """Where an eval run keeps this reply, so a rerun spends no quota (§10.2)."""
    frame_ids = [frame.frame_id for frame in frames]
    if settings.gemini_cache_dir is None or None in frame_ids:
        return None  # only eval runs cache, and only stored frames, which keep their ids
    key = json.dumps([settings.gemini_model, system, request, frame_ids], ensure_ascii=False)
    return settings.gemini_cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


async def generate(model: str, contents: list, system: str, settings: Settings) -> GeminiAnswer:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=settings.gemini_temperature,
            response_mime_type="application/json",
            response_schema=GeminiAnswer,
            # No retry_options, so the SDK doesn't retry and a failure is reported at once.
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
        ),
    )
    return GeminiAnswer.model_validate_json(response.text)


async def answer(
    frames: Sequence[Frame],
    question: str,
    intent: Intent,
    window: str,
    lang: Lang,
    settings: Settings,
) -> tuple[GeminiAnswer, str]:
    """The answer and the model that wrote it. Raises if both models fail; the caller handles it."""
    # replace, not format: the prompt's JSON example has braces of its own.
    system = SYSTEM_PROMPT.replace("{language}", LANGUAGE_NAMES[lang])
    request = f"Question type: {intent} (time window: {window})\nQuestion: {question}"
    labels = [frame_label(chr(ord("A") + i), frame) for i, frame in enumerate(frames)]
    contents = [
        *(types.Part.from_bytes(data=frame.jpeg, mime_type="image/jpeg") for frame in frames),
        "\n".join([request, *labels]),
    ]

    # The labels' "seconds ago" are left out of the key, so a rerun on the same frames hits it.
    cache = cache_file(settings, system, request, frames)
    if cache is not None and cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        reply, model = GeminiAnswer.model_validate(cached["reply"]), cached["model"]
    else:
        model = settings.gemini_model
        try:
            reply = await generate(model, contents, system, settings)
        except errors.APIError as error:
            if error.code != 429 and error.code < 500:
                raise
            # Busy or over quota: one more try on Flash-Lite, which has more free quota (§6.5.4).
            model = settings.gemini_fallback_model
            reply = await generate(model, contents, system, settings)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            body = {"model": model, "reply": reply.model_dump()}
            cache.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    if reply.confidence == "low" and not reply.answer.startswith(NOT_SURE[lang]):
        # shortcut: an unsure answer worded differently ("I am not sure…") gets the prefix too;
        # ask for the exact words in prompt rule 3 if the eval shows it happening.
        reply = reply.model_copy(update={"answer": f"{NOT_SURE[lang]}. {reply.answer}"})
    return reply, model
