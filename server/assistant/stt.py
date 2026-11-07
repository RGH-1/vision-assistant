"""Speech to text with Groq Whisper (design §6.5.1)."""

import time
from dataclasses import dataclass
from typing import Literal, get_args

from groq import AsyncGroq

from server.common.schemas import Lang
from server.settings import Settings

DIDNT_CATCH: dict[Lang, str] = {
    "en": "Sorry, I didn't catch that. Please try again.",
    "ar": "عذرًا، لم أسمع ذلك جيدًا. من فضلك حاول مرة أخرى.",
    "fr": "Désolé, je n'ai pas bien entendu. Veuillez réessayer.",
}


@dataclass(frozen=True)
class Transcript:
    text: str  # empty when the audio wasn't understood: speak DIDNT_CATCH[lang] instead
    engine: Literal["groq"]
    time_ms: float


def is_understood(text: str, segments: list[dict], settings: Settings) -> bool:
    if not any(character.isalpha() for character in text):
        return False
    return any(
        segment["avg_logprob"] >= settings.stt_min_avg_logprob
        and segment["no_speech_prob"] <= settings.stt_max_no_speech_prob
        for segment in segments
    )


async def transcribe(audio: bytes, lang: str, settings: Settings) -> Transcript:
    """Raises ValueError for an unsupported language and groq.GroqError if Groq fails."""
    if lang not in get_args(Lang):
        raise ValueError(f"Unsupported language: {lang}")

    started = time.perf_counter()
    # The audio is only ever in memory, so no file can be left behind (§6.5.1, §9).
    # No retries, so a failure is reported at once instead of leaving the user waiting.
    async with AsyncGroq(timeout=settings.stt_timeout_s, max_retries=0) as client:
        result = await client.audio.transcriptions.create(
            model=settings.stt_model,
            file=("question.wav", audio),
            language=lang,
            temperature=0,
            response_format="verbose_json",
        )
    time_ms = (time.perf_counter() - started) * 1000

    text = result.text.strip()
    if not is_understood(text, getattr(result, "segments", []), settings):
        text = ""
    return Transcript(text=text, engine="groq", time_ms=time_ms)