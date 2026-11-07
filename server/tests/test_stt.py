"""VA-63 checks. Runs the real Groq library, with only the network replaced."""

import asyncio
from functools import partial
from types import SimpleNamespace
from typing import get_args

import groq
import httpx
import pytest

from server.assistant import stt
from server.common.schemas import Lang

SETTINGS = SimpleNamespace(
    stt_model="test-model",
    stt_timeout_s=1,
    stt_min_avg_logprob=-0.5,
    )
GOOD = {"avg_logprob": -0.2}


def fake_groq(monkeypatch: pytest.MonkeyPatch, status: int, body: dict) -> list[httpx.Request]:
    """Answer every Groq request with this response, and return the requests that were made."""
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


@pytest.mark.parametrize("status", [429, 500])
def test_groq_errors_are_raised_without_retrying(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    requests = fake_groq(monkeypatch, status, {"error": {"message": "no"}})

    with pytest.raises(groq.APIStatusError):
        asyncio.run(stt.transcribe(b"audio", "en", SETTINGS))

    assert len(requests) == 1


def test_every_language_has_a_didnt_catch_message() -> None:
    assert set(stt.DIDNT_CATCH) == set(get_args(Lang))
