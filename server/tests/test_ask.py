"""VA-64 checks: /v1/ask and /v1/ask_text with the real database. Groq and Gemini are faked."""

import asyncio
import time
from collections.abc import Iterator
from typing import Any, get_args
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg import Connection

from server import main
from server.assistant import gemini, stt
from server.common.db import open_ingest_connection
from server.common.schemas import AskResponse, Error, Lang, QueryAnsweredMessage
from server.main import create_app
from server.routes.ask import CANT_SEE, WENT_WRONG
from server.settings import Settings

CONSENT_VER = "test-va-64"
QUESTION = "What's in front of me?"
REPLY = gemini.GeminiAnswer(
    answer="A door is ahead of you.",
    confidence="high",
    used_frames=["Frame A"],
    reason_if_unsure="",
)


class IdleIngestWorker:
    """The tests put frames in the live buffer themselves."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture(scope="module")
def db() -> Iterator[Connection]:
    conn = open_ingest_connection()
    conn.autocommit = True
    yield conn
    conn.close()


# Module scope because connecting to Postgres is slow here, so the tests share one app.
@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("data"),
        retention_minutes=30,
        host="127.0.0.1",
        port=8000,
        mediamtx_webrtc_port=8889,
        cors_origins=[],
        clip_text_model="stub",
        status_interval_s=60,
        ws_ping_interval_s=60,
        ws_send_timeout_s=1,
        stt_model="stub",
        stt_timeout_s=1,
        stt_min_avg_logprob=-0.5,
        gemini_model="test-gemini",
        gemini_timeout_s=1,
        gemini_temperature=0.2,
        default_window_s=180,
    )
    # Same event loop as server/__main__.py, because async psycopg can't use the Windows default.
    options = {"loop_factory": asyncio.SelectorEventLoop}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main, "load_clip_text_encoder", lambda _: "clip-stub")
        patch.setattr(main, "create_ingest_worker", lambda _: IdleIngestWorker())
        with TestClient(create_app(settings), backend_options=options) as test_client:
            yield test_client


@pytest.fixture
def session_id(client: TestClient, db: Connection) -> Iterator[str]:
    body = {"app_lang": "en", "consent_at": "2026-11-08T10:00:00Z", "consent_ver": CONSENT_VER}
    created = client.post("/v1/sessions", json=body).json()["session_id"]
    # A new session empties the live buffer, so the frame goes in after it.
    client.app.state.live_buffer.add(np.zeros((540, 960, 3), np.uint8), time.monotonic() - 2)
    client.app.state.stream_connected = True
    yield created
    db.execute("DELETE FROM sessions WHERE consent_ver = %s", (CONSENT_VER,))


@pytest.fixture
def gemini_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(jpeg, seconds_ago, detections, question, lang, settings) -> gemini.GeminiAnswer:
        calls.append({"jpeg": jpeg, "seconds_ago": seconds_ago, "question": question, "lang": lang})
        return REPLY

    monkeypatch.setattr(gemini, "describe_live_scene", fake)
    return calls


def fake_transcribe(monkeypatch: pytest.MonkeyPatch, result: stt.Transcript | Exception) -> None:
    async def fake(audio: bytes, lang: str, settings: Settings) -> stt.Transcript:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(stt, "transcribe", fake)


def ask_by_voice(client: TestClient, session_id: str, lang: str = "en") -> httpx.Response:
    return client.post(
        "/v1/ask",
        data={"session_id": session_id, "lang": lang},
        files={"audio": ("question.wav", b"wav bytes", "audio/wav")},
    )


def ask_by_text(client: TestClient, session_id: str, lang: str = "en") -> httpx.Response:
    body = {"session_id": session_id, "text": QUESTION, "lang": lang}
    return client.post("/v1/ask_text", json=body)


def saved_questions(db: Connection, session_id: str) -> list[dict[str, Any]]:
    return db.execute("SELECT * FROM questions WHERE session_id = %s", (session_id,)).fetchall()


def assert_error(response: httpx.Response, status_code: int, code: str, speak: str) -> None:
    assert response.status_code == status_code
    error = Error.model_validate(response.json()).error
    assert (error.code, error.speak) == (code, speak)


def test_text_question_is_answered_from_the_live_frame(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    response = ask_by_text(client, session_id)

    assert response.status_code == 200
    answer = AskResponse.model_validate(response.json())
    assert (answer.intent, answer.answer, answer.confidence, answer.abstained) == (
        "LIVE_SCENE",
        REPLY.answer,
        "high",
        False,
    )
    assert answer.evidence[0].frame_id is None
    assert answer.evidence[0].seconds_ago >= 2
    assert set(answer.timings_ms) == {"gemini", "total"}
    assert gemini_calls[0]["question"] == QUESTION
    assert gemini_calls[0]["jpeg"].startswith(b"\xff\xd8")  # a JPEG from the live buffer
    [row] = saved_questions(db, session_id)
    assert row["id"] == answer.question_id
    assert (row["input_mode"], row["stt_engine"], row["transcript"], row["model"]) == (
        "text",
        None,
        QUESTION,
        "test-gemini",
    )


def test_voice_question_is_transcribed_then_answered(
    client: TestClient,
    db: Connection,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transcribe(monkeypatch, stt.Transcript(text=QUESTION, engine="groq", time_ms=480.0))

    response = ask_by_voice(client, session_id)

    assert response.status_code == 200
    answer = AskResponse.model_validate(response.json())
    assert answer.transcript == QUESTION
    assert answer.timings_ms["stt"] == 480
    assert gemini_calls[0]["question"] == QUESTION
    [row] = saved_questions(db, session_id)
    assert (row["input_mode"], row["stt_engine"]) == ("voice", "groq")


def test_answered_question_reaches_the_dashboard(
    client: TestClient,
    db: Connection,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transcribe(monkeypatch, stt.Transcript(text=QUESTION, engine="groq", time_ms=480.0))

    with client.websocket_connect("/v1/ws/dashboard") as dashboard:
        dashboard.send_json({"type": "ping"})
        assert dashboard.receive_json() == {"type": "pong"}  # the server has registered this client

        assert ask_by_voice(client, session_id).status_code == 200
        client.app.state.hub.publish({"type": "ping"})  # so a missing event fails, not hangs
        event = dashboard.receive_json()

    assert event["type"] == "query_answered"
    [row] = saved_questions(db, session_id)
    assert QueryAnsweredMessage.model_validate(event).model_dump(exclude={"type"}) == {
        "question_id": row["id"],
        "session_id": UUID(session_id),
        "asked_at": row["asked_at"],
        "input_mode": "voice",
        "stt_engine": "groq",
        "transcript": QUESTION,
        "lang": "en",
        "intent": "LIVE_SCENE",
        "t_from": None,
        "t_to": None,
        "answer": REPLY.answer,
        "confidence": "high",
        "abstained": False,
        "safety_rewrite": False,
        "model": "test-gemini",
        "timings_ms": row["timings_ms"],
        "evidence": [],
    }


def test_low_confidence_counts_as_abstained(
    client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unsure(*_: Any) -> gemini.GeminiAnswer:
        return REPLY.model_copy(update={"confidence": "low"})

    monkeypatch.setattr(gemini, "describe_live_scene", unsure)

    answer = AskResponse.model_validate(ask_by_text(client, session_id).json())
    assert (answer.confidence, answer.abstained) == ("low", True)


@pytest.mark.parametrize(
    ("result", "status_code", "code", "speak"),
    [
        (stt.Transcript(text="", engine=None, time_ms=5), 400, "BAD_AUDIO", stt.DIDNT_CATCH["ar"]),
        (ValueError("Expected 16 kHz mono 16-bit WAV"), 400, "BAD_AUDIO", stt.DIDNT_CATCH["ar"]),
        (RuntimeError("local Whisper failed too"), 502, "STT_FAILED", WENT_WRONG["ar"]),
    ],
)
def test_speech_problems_get_a_spoken_reply_and_skip_gemini(
    client: TestClient,
    db: Connection,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    result: stt.Transcript | Exception,
    status_code: int,
    code: str,
    speak: str,
) -> None:
    fake_transcribe(monkeypatch, result)

    assert_error(ask_by_voice(client, session_id, "ar"), status_code, code, speak)
    assert not gemini_calls
    assert not saved_questions(db, session_id)


@pytest.mark.parametrize(("connected", "frames"), [(False, True), (True, False)])
def test_without_a_live_frame_the_reply_says_the_camera_is_off(
    client: TestClient,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    connected: bool,
    frames: bool,
) -> None:
    client.app.state.stream_connected = connected
    if not frames:
        client.app.state.live_buffer.clear()

    assert_error(ask_by_text(client, session_id, "fr"), 503, "NO_STREAM", CANT_SEE["fr"])
    assert not gemini_calls


def test_gemini_failure_still_gets_a_spoken_reply(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(*_: Any) -> gemini.GeminiAnswer:
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(gemini, "describe_live_scene", failing)

    assert_error(ask_by_text(client, session_id), 502, "GEMINI_FAILED", WENT_WRONG["en"])
    assert not saved_questions(db, session_id)


def test_unknown_session_is_not_found(
    client: TestClient, gemini_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_transcribe(monkeypatch, stt.Transcript(text=QUESTION, engine="groq", time_ms=1))
    unknown = str(uuid4())

    assert ask_by_text(client, unknown).status_code == 404
    assert ask_by_voice(client, unknown).status_code == 404
    assert not gemini_calls


def test_voice_question_without_audio_is_a_bad_request(client: TestClient, session_id: str) -> None:
    response = client.post("/v1/ask", data={"session_id": session_id, "lang": "en"})

    assert response.status_code == 400
    assert Error.model_validate(response.json()).error.code == "BAD_REQUEST"


def test_every_language_has_its_spoken_replies() -> None:
    assert set(CANT_SEE) == set(WENT_WRONG) == set(gemini.LANGUAGE_NAMES) == set(get_args(Lang))
