"""VA-72 checks for the Gemini request, with a fake client. eval/check_ask.py tries the real one."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pydantic
import pytest
from google.genai import errors

from server.assistant import gemini

SETTINGS = SimpleNamespace(
    gemini_model="test-model",
    gemini_fallback_model="test-lite",
    gemini_timeout_s=30,
    gemini_temperature=0.2,
    gemini_cache_dir=None,
)
REPLY_JSON = (
    '{"answer": "A door.", "confidence": "high", "used_frames": ["Frame A"], '
    '"reason_if_unsure": ""}'
)
UNSURE_JSON = REPLY_JSON.replace('"high"', '"low"')
LIVE = gemini.Frame(b"live", 3.4, [{"label": "door", "position": "left", "confidence": 0.823}])
SIGN = gemini.Frame(b"sign", 14.2, (), ("Room 204", "Computer Lab"), frame_id=7)


def fake_gemini(monkeypatch: pytest.MonkeyPatch, *replies: str | Exception) -> list[dict]:
    """Replace the Gemini client; call N gets reply N. Returns the requests it was given."""
    calls: list[dict] = []

    async def generate_content(**request: object) -> SimpleNamespace:
        calls.append(request)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)

    models = SimpleNamespace(generate_content=generate_content)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    monkeypatch.setattr(gemini.genai, "Client", lambda api_key: client)
    return calls


def ask(
    frames: tuple[gemini.Frame, ...] = (LIVE,), lang: str = "en", settings: object = SETTINGS
) -> tuple[gemini.GeminiAnswer, str]:
    return asyncio.run(gemini.answer(frames, "What's here?", "LIVE_SCENE", "now", lang, settings))


def test_sends_every_frame_with_its_label_and_the_question_in_the_users_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_gemini(monkeypatch, REPLY_JSON)

    reply, model = asyncio.run(
        gemini.answer((LIVE, SIGN), "شو شفنا قبل شوي؟", "MEMORY", "last 90 seconds", "ar", SETTINGS)
    )

    assert (reply.answer, model) == ("A door.", "test-model")
    [request] = calls
    assert request["model"] == "test-model"
    live, sign, text = request["contents"]
    assert (live.inline_data.data, sign.inline_data.data) == (b"live", b"sign")
    assert text == (
        "Question type: MEMORY (time window: last 90 seconds)\n"
        "Question: شو شفنا قبل شوي؟\n"
        "Frame A — 3 seconds ago — objects: door (left, 0.82) — text: none\n"
        'Frame B — 14 seconds ago — objects: none — text: "Room 204", "Computer Lab"'
    )
    config = request["config"]
    assert "Answer in Arabic." in config.system_instruction
    assert "{language}" not in config.system_instruction
    assert config.temperature == 0.2
    assert config.http_options.timeout == 30_000


@pytest.mark.parametrize("error", [errors.ClientError(429, {}), errors.ServerError(503, {})])
def test_a_busy_or_over_quota_model_is_retried_once_on_flash_lite(
    monkeypatch: pytest.MonkeyPatch, error: errors.APIError
) -> None:
    calls = fake_gemini(monkeypatch, error, REPLY_JSON)

    reply, model = ask()

    assert [call["model"] for call in calls] == ["test-model", "test-lite"]
    assert (reply.answer, model) == ("A door.", "test-lite")


def test_when_flash_lite_fails_too_the_error_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_gemini(monkeypatch, errors.ClientError(429, {}), errors.ServerError(500, {}))

    with pytest.raises(errors.ServerError):
        ask()
    assert len(calls) == 2


def test_a_bad_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_gemini(monkeypatch, errors.ClientError(400, {}))

    with pytest.raises(errors.ClientError):
        ask()
    assert len(calls) == 1


def test_a_reply_that_is_not_the_json_asked_for_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gemini(monkeypatch, '{"answer": "A door."}')

    with pytest.raises(pydantic.ValidationError):
        ask()


@pytest.mark.parametrize(
    ("lang", "answer", "expected"),
    [
        ("en", "Maybe a door.", "I'm not sure. Maybe a door."),
        ("ar", "ربما باب.", "لست متأكدًا. ربما باب."),
        ("en", "I'm not sure, maybe a door.", "I'm not sure, maybe a door."),  # not twice
    ],
)
def test_a_low_confidence_answer_starts_with_im_not_sure(
    monkeypatch: pytest.MonkeyPatch, lang: str, answer: str, expected: str
) -> None:
    fake_gemini(monkeypatch, UNSURE_JSON.replace("A door.", answer))

    reply, _ = ask(lang=lang)

    assert (reply.answer, reply.confidence) == (expected, "low")


def test_an_eval_run_reuses_the_reply_for_the_same_stored_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = fake_gemini(monkeypatch, errors.ServerError(503, {}), UNSURE_JSON)
    settings = SimpleNamespace(**{**vars(SETTINGS), "gemini_cache_dir": tmp_path})
    later = gemini.Frame(b"sign", 60, (), SIGN.texts, frame_id=7)  # same frame, asked later

    first = ask((SIGN,), settings=settings)
    second = ask((later,), settings=settings)

    assert len(calls) == 2  # the first ask only: 503, then Flash-Lite
    assert first == second
    assert second[1] == "test-lite"
    assert second[0].answer.startswith("I'm not sure. ")


def test_a_live_frame_is_never_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = fake_gemini(monkeypatch, REPLY_JSON, REPLY_JSON)
    settings = SimpleNamespace(**{**vars(SETTINGS), "gemini_cache_dir": tmp_path})

    ask((LIVE,), settings=settings)
    ask((LIVE,), settings=settings)

    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


ASKED_AT = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("t_from", "t_to", "expected"),
    [
        (None, None, "now"),
        (ASKED_AT - timedelta(seconds=90), ASKED_AT, "last 90 seconds"),
        (
            ASKED_AT - timedelta(seconds=150),
            ASKED_AT - timedelta(seconds=30),
            "150 to 30 seconds ago",
        ),
    ],
)
def test_the_time_window_is_said_in_seconds_before_the_question(
    t_from: datetime | None, t_to: datetime | None, expected: str
) -> None:
    assert gemini.time_window(t_from, t_to, ASKED_AT) == expected


def test_a_stored_frame_is_shrunk_to_the_size_gemini_gets(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.zeros((640, 1280, 3), np.uint8))

    jpeg = gemini.load_jpeg(path, max_side=1024)

    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (512, 1024)
    assert gemini.load_jpeg(tmp_path / "deleted.jpg", max_side=1024) is None
