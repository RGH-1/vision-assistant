"""Speech to text with Groq Whisper, and local faster-whisper when Groq fails (design §6.5.1)."""

import asyncio
import logging
import threading
import time
import wave
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from typing import Literal, get_args

import groq
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from groq import AsyncGroq

from server.common.schemas import Lang
from server.settings import Settings

logger = logging.getLogger(__name__)

MAX_AUDIO_S = 10  # the upload limit in the contract (§8.1)
LOCAL_MODEL = "medium"  # the CPU fallback in §6.5.1
# One local transcription at a time: the model loads only once, and the CPU is busy anyway.
LOCAL_LOCK = threading.Lock()

DIDNT_CATCH: dict[Lang, str] = {
    "en": "Sorry, I didn't catch that. Please try again.",
    "ar": "عذرًا، لم أسمع ذلك جيدًا. من فضلك حاول مرة أخرى.",
    "fr": "Désolé, je n'ai pas bien entendu. Veuillez réessayer.",
}


@dataclass(frozen=True)
class Transcript:
    text: str  # empty when the audio wasn't understood: speak DIDNT_CATCH[lang] instead
    engine: Literal["groq", "local"] | None  # None when the audio had no speech, so no engine ran
    time_ms: float


def read_samples(audio: bytes) -> np.ndarray:
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

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768


def has_speech(audio: bytes) -> bool:
    return bool(get_speech_timestamps(read_samples(audio), VadOptions()))


def is_understood(text: str, avg_logprobs: list[float], settings: Settings) -> bool:
    if not any(character.isalpha() for character in text):
        return False
    return any(logprob >= settings.stt_min_avg_logprob for logprob in avg_logprobs)


@cache
def load_local_model() -> WhisperModel:
    # Loaded on first use, so it takes no RAM unless Groq fails (§6.5.1).
    return WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")


def transcribe_locally(audio: bytes, lang: str) -> tuple[str, list[float]]:
    # Samples, not the WAV bytes: faster-whisper's own decoder fails with the installed av 19.
    samples = read_samples(audio)
    with LOCAL_LOCK:
        segments, _ = load_local_model().transcribe(samples, language=lang, temperature=0)
        segments = list(segments)  # the transcription runs while this list is filled
    return "".join(segment.text for segment in segments), [s.avg_logprob for s in segments]


async def transcribe(audio: bytes, lang: str, settings: Settings) -> Transcript:
    """Raises ValueError for bad audio or language; any other error means both engines failed."""
    if lang not in get_args(Lang):
        raise ValueError(f"Unsupported language: {lang}")

    started = time.perf_counter()
    if not await asyncio.to_thread(has_speech, audio):
        return Transcript(text="", engine=None, time_ms=(time.perf_counter() - started) * 1000)
    engine: Literal["groq", "local"] = "groq"
    try:
        # The audio is only ever in memory, so no file can be left behind (§6.5.1, §9).
        # No retries, so a failure switches to local Whisper at once.
        async with AsyncGroq(timeout=settings.stt_timeout_s, max_retries=0) as client:
            result = await client.audio.transcriptions.create(
                model=settings.stt_model,
                file=("question.wav", audio),
                language=lang,
                temperature=0,
                response_format="verbose_json",
            )
        text = result.text
        avg_logprobs = [segment["avg_logprob"] for segment in getattr(result, "segments", [])]
    except groq.GroqError:
        # No key, a timeout, a 429 or an outage: the demo must not go silent (§10.3).
        logger.warning("Groq failed, so local Whisper transcribes this question", error)
        engine = "local"
        text, avg_logprobs = await asyncio.to_thread(transcribe_locally, audio, lang)
    time_ms = (time.perf_counter() - started) * 1000

    text = text.strip()
    if not is_understood(text, avg_logprobs, settings):
        text = ""
    return Transcript(text=text, engine=engine, time_ms=time_ms)
