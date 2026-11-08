"""Speech to text with Groq Whisper (design §6.5.1)."""

import asyncio
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, get_args

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps
from groq import AsyncGroq

from server.common.schemas import Lang
from server.settings import Settings

MAX_AUDIO_S = 10 # the upload limit in the contract (§8.1)

DIDNT_CATCH: dict[Lang, str] = {
    "en": "Sorry, I didn't catch that. Please try again.",
    "ar": "عذرًا، لم أسمع ذلك جيدًا. من فضلك حاول مرة أخرى.",
    "fr": "Désolé, je n'ai pas bien entendu. Veuillez réessayer.",
}


@dataclass(frozen=True)
class Transcript:
    text: str  # empty when the audio wasn't understood: speak DIDNT_CATCH[lang] instead
    engine: Literal["groq"] | None  # None when the audio had no speech, so no engine ran
    time_ms: float


def has_speech(audio: bytes) -> bool:
    """Raises ValueError unless the audio is a WAV in the upload format (§8)."""
    try:
        with wave.open(BytesIO(audio)) as recording:
            audio_format = (
                recording.getframerate(),
                recording.getnchannels(),
                recording.getsampwidth(),
            )
            frames = recording.readframes(recording.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Not a valid WAV file: {error}") from error
    if audio_format != (16000, 1, 2):
        raise ValueError(f"Expected 16 kHz mono 16-bit WAV, got {audio_format}")
    if len(frames) > MAX_AUDIO_S * 16000 * 2:
        raise ValueError(f"Audio is longer than {MAX_AUDIO_S} s")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    return bool(get_speech_timestamps(samples, VadOptions()))


def is_understood(text: str, segments: list[dict], settings: Settings) -> bool:
    if not any(character.isalpha() for character in text):
        return False
    return any(segment["avg_logprob"] >= settings.stt_min_avg_logprob for segment in segments)


async def transcribe(audio: bytes, lang: str, settings: Settings) -> Transcript:
    """Raises ValueError for a bad language or audio format, groq.GroqError if Groq fails."""
    if lang not in get_args(Lang):
        raise ValueError(f"Unsupported language: {lang}")

    started = time.perf_counter()
    if not await asyncio.to_thread(has_speech, audio):
        return Transcript(text="", engine=None, time_ms=(time.perf_counter() - started) * 1000)
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
