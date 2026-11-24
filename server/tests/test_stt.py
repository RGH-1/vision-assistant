"""VA-63 and VA-66 checks. Runs the real Groq library, with only the network replaced."""

import asyncio
import wave
from functools import partial
from io import BytesIO
from types import SimpleNamespace
from typing import get_args

import groq
import httpx
import numpy as np
import pytest

from server.assistant import stt
from server.common.schemas import Lang

SETTINGS = SimpleNamespace(
    stt_model="test-model",
    stt_timeout_s=1,
    stt_min_avg_logprob=-0.5,
)
GOOD = {"avg_logprob": -0.2}


def silent_wav(seconds: float = 1, rate: int = 16000, channels: int = 1) -> bytes:
    file = BytesIO()
    with wave.open(file, "wb") as recording:
        recording.setframerate(rate)
        recording.setnchannels(channels)
        recording.setsampwidth(2)
        recording.writeframes(bytes(int(seconds * rate) * channels * 2))
    return file.getvalue()


def fake_groq(monkeypatch: pytest.MonkeyPatch, status: int, body: dict) -> list[httpx.Request]:
    """Answer every Groq request with this response, and return the requests that were made."""
    monkeypatch.setattr(stt, "has_speech", lambda audio: True)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        stt, "AsyncGroq", partial(groq.AsyncGroq, api_key="test", http_client=client)
    )
    return requests


def test_sends_the_design_settings_and_returns_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = fake_groq(monkeypatch, 200, {"text": " ما هذا؟ ", "segments": [GOOD]})

    transcript = asyncio.run(stt.transcribe(b"audio bytes", "ar", SETTINGS))

    assert transcript.text == "ما هذا؟"
    assert transcript.engine == "groq"
    assert transcript.time_ms >= 0
    sent = requests[0].read().decode()
    for field, value in [
        ("model", "test-model"),
        ("language", "ar"),
        ("temperature", "0"),
        ("response_format", "verbose_json"),
    ]:
        assert f'name="{field}"\r\n\r\n{value}\r\n' in sent
    assert "audio bytes" in sent


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        ("", []),
        ("...  !?", [GOOD]),
        ("mumble", [{"avg_logprob": -0.74}]),
    ],
)
def test_empty_or_low_confidence_transcripts_are_not_understood(
    monkeypatch: pytest.MonkeyPatch, text: str, segments: list[dict]
) -> None:
    fake_groq(monkeypatch, 200, {"text": text, "segments": segments})

    assert asyncio.run(stt.transcribe(b"audio", "en", SETTINGS)).text == ""


def test_one_good_segment_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"avg_logprob": -2.0}
    fake_groq(monkeypatch, 200, {"text": "Where is the door?", "segments": [bad, GOOD]})

    assert asyncio.run(stt.transcribe(b"audio", "en", SETTINGS)).text == "Where is the door?"


def test_unsupported_language_is_rejected_without_calling_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = fake_groq(monkeypatch, 200, {"text": "hello", "segments": [GOOD]})

    with pytest.raises(ValueError, match="de"):
        asyncio.run(stt.transcribe(b"audio", "de", SETTINGS))

    assert not requests


def fake_local_whisper(
    monkeypatch: pytest.MonkeyPatch, text: str, avg_logprobs: list[float]
) -> list[tuple]:
    """Replace local Whisper with this result, and return the calls it was given."""
    calls: list[tuple] = []

    def transcribe_locally(audio: bytes, lang: str) -> tuple[str, list[float]]:
        calls.append((audio, lang))
        return text, avg_logprobs

    monkeypatch.setattr(stt, "transcribe_locally", transcribe_locally)
    return calls


@pytest.mark.parametrize("status", [401, 429, 500])
def test_groq_errors_switch_to_local_whisper_without_retrying(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    requests = fake_groq(monkeypatch, status, {"error": {"message": "no"}})
    calls = fake_local_whisper(monkeypatch, " Where is the door? ", [-0.2])

    transcript = asyncio.run(stt.transcribe(b"audio", "en", SETTINGS))

    assert (transcript.text, transcript.engine) == ("Where is the door?", "local")
    assert calls == [(b"audio", "en")]
    assert len(requests) == 1


def test_a_missing_groq_key_switches_to_local_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(stt, "has_speech", lambda audio: True)
    fake_local_whisper(monkeypatch, "Where is the door?", [-0.2])

    assert asyncio.run(stt.transcribe(b"audio", "en", SETTINGS)).engine == "local"


def test_local_transcripts_pass_the_same_confidence_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_groq(monkeypatch, 500, {"error": {"message": "no"}})
    fake_local_whisper(monkeypatch, "mumble", [-0.74])

    assert asyncio.run(stt.transcribe(b"audio", "en", SETTINGS)).text == ""


def test_local_whisper_gets_samples_in_the_users_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def transcribe(samples: np.ndarray, **options: object) -> tuple:
        calls.append({"samples": samples, **options})
        segments = [
            SimpleNamespace(text=" ما", avg_logprob=-0.1),
            SimpleNamespace(text=" هذا؟", avg_logprob=-0.3),
        ]
        return iter(segments), None

    monkeypatch.setattr(stt, "load_local_model", lambda: SimpleNamespace(transcribe=transcribe))

    assert stt.transcribe_locally(silent_wav(1), "ar") == (" ما هذا؟", [-0.1, -0.3])
    [call] = calls
    assert call["samples"].dtype == np.float32
    assert len(call["samples"]) == 16000
    assert (call["language"], call["temperature"]) == ("ar", 0)


def test_audio_without_speech_is_not_sent_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = fake_groq(monkeypatch, 200, {"text": "Thank you.", "segments": [GOOD]})
    monkeypatch.setattr(stt, "has_speech", lambda audio: False)

    transcript = asyncio.run(stt.transcribe(b"audio", "en", SETTINGS))

    assert transcript.text == ""
    assert transcript.engine is None
    assert not requests


@pytest.mark.parametrize("seconds", [0, 1, 10])
def test_silence_has_no_speech(seconds: int) -> None:
    assert not stt.has_speech(silent_wav(seconds))


@pytest.mark.parametrize(
    "audio", [b"not a wav", b"", silent_wav(rate=48000), silent_wav(channels=2), silent_wav(10.1)]
)
def test_audio_outside_the_upload_format_is_rejected(audio: bytes) -> None:
    with pytest.raises(ValueError):
        stt.has_speech(audio)


def test_every_language_has_a_didnt_catch_message() -> None:
    assert set(stt.DIDNT_CATCH) == set(get_args(Lang))
