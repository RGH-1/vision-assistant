"""VA-64 checks for the Gemini request. eval/check_ask.py tries the real Gemini."""

import asyncio
from types import SimpleNamespace

import pydantic
import pytest

from server.assistant import gemini

SETTINGS = SimpleNamespace(gemini_model="test-model", gemini_timeout_s=30, gemini_temperature=0.2)
REPLY_JSON = (
    '{"answer": "A door.", "confidence": "high", "used_frames": ["Frame A"], '
    '"reason_if_unsure": ""}'
)


def fake_gemini(monkeypatch: pytest.MonkeyPatch, reply_text: str) -> list[dict]:
    """Replace the Gemini client, and return the requests it was given."""
    calls: list[dict] = []

    async def generate_content(**request: object) -> SimpleNamespace:
        calls.append(request)
        return SimpleNamespace(text=reply_text)

    models = SimpleNamespace(generate_content=generate_content)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    monkeypatch.setattr(gemini.genai, "Client", lambda api_key: client)
    return calls


def test_sends_one_labelled_frame_and_the_question_in_the_users_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_gemini(monkeypatch, REPLY_JSON)
    detections = [{"label": "door", "position": "left", "confidence": 0.823}]

    reply = asyncio.run(
        gemini.describe_live_scene(b"jpeg", 3.4, detections, "شو في قدامي؟", "ar", SETTINGS)
    )

    assert reply.answer == "A door."
    [request] = calls
    assert request["model"] == "test-model"
    image, text = request["contents"]
    assert image.inline_data.data == b"jpeg"
    assert text == (
        "Question type: LIVE_SCENE (time window: now)\n"
        "Question: شو في قدامي؟\n"
        "Frame A — 3 seconds ago — objects: door (left, 0.82) — text: none"
    )
    config = request["config"]
    assert "Answer in Arabic." in config.system_instruction
    assert "{language}" not in config.system_instruction
    assert config.temperature == 0.2
    assert config.http_options.timeout == 30_000


def test_a_frame_without_detections_says_none() -> None:
    assert "objects: none" in gemini.frame_label(0.4, ())


def test_a_reply_that_is_not_the_json_asked_for_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gemini(monkeypatch, '{"answer": "A door."}')

    with pytest.raises(pydantic.ValidationError):
        asyncio.run(gemini.describe_live_scene(b"jpeg", 1, (), "What's here?", "en", SETTINGS))