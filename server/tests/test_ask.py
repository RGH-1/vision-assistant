"""VA-64 checks: /v1/ask and /v1/ask_text with the real database. Groq and Gemini are faked."""

import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, get_args
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg import Connection

from server import main
from server.assistant import gate, gemini, stt
from server.common.db import insert_frame, open_ingest_connection
from server.common.schemas import AskResponse, Error, Lang, QueryAnsweredMessage
from server.ingest.faceblur import FaceBlurResult
from server.main import create_app
from server.routes.ask import CANT_SEE, NO_FRENCH, WENT_WRONG
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


class FakeClipText:
    """Every question means the same thing, so stored frames are ranked by their own vectors."""

    def encode(self, text: str) -> np.ndarray:
        return unit_vector(0)


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


class FakeFaceBlurrer:
    def blur(self, image: np.ndarray) -> FaceBlurResult:
        return FaceBlurResult(image.copy(), 0)


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
        patch.setattr(main, "load_clip_text_encoder", lambda _: FakeClipText())
        patch.setattr(main, "create_ingest_worker", lambda _: IdleIngestWorker())
        with TestClient(create_app(settings), backend_options=options) as test_client:
            yield test_client


@pytest.fixture
def session_id(client: TestClient, db: Connection) -> Iterator[str]:
    body = {"app_lang": "en", "consent_at": "2026-11-08T10:00:00Z", "consent_ver": CONSENT_VER}
    created = client.post("/v1/sessions", json=body).json()["session_id"]
    # A new session empties the live buffer, so the frame goes in after it.
    client.app.state.live_buffer.add(
        np.zeros((540, 960, 3), np.uint8), time.monotonic() - 2, face_blurrer=FakeFaceBlurrer()
    )
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


def ask_by_text(
    client: TestClient, session_id: str, lang: str = "en", text: str = QUESTION
) -> httpx.Response:
    body = {"session_id": session_id, "text": text, "lang": lang}
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
    assert set(answer.timings_ms) == {"understand", "gemini", "total"}
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

    response = ask_by_text(client, session_id, "ar", "شو في قدامي؟")

    assert_error(response, 503, "NO_STREAM", CANT_SEE["ar"])
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
    langs = set(get_args(Lang))
    assert set(CANT_SEE) == set(WENT_WRONG) == set(gemini.LANGUAGE_NAMES) == langs
    assert set(gate.NOT_SURE) == langs
    assert all(set(reply) == langs for reply in gate.FIXED_REPLIES.values())


@pytest.mark.parametrize(
    ("lang", "text", "intent"),
    [
        ("en", "Who is this?", "IDENTITY"),
        ("ar", "مين هيدا؟", "IDENTITY"),
        ("en", "What's the capital of France?", "OUT_OF_SCOPE"),
    ],
)
def test_identity_and_out_of_scope_get_a_fixed_reply_without_gemini(
    client: TestClient,
    db: Connection,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    lang: str,
    text: str,
    intent: str,
) -> None:
    response = ask_by_text(client, session_id, lang, text)

    assert response.status_code == 200
    answer = AskResponse.model_validate(response.json())
    assert (answer.intent, answer.answer, answer.confidence, answer.abstained) == (
        intent,
        gate.FIXED_REPLIES[intent][lang],
        "none",
        True,
    )
    assert answer.evidence == []
    assert not gemini_calls
    [row] = saved_questions(db, session_id)
    assert (row["intent"], row["model"]) == (intent, "rule")


def test_a_memory_question_with_no_evidence_is_not_sure_without_gemini(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    response = ask_by_text(client, session_id, "en", "Did you see a fire extinguisher?")

    answer = AskResponse.model_validate(response.json())
    assert (answer.intent, answer.answer, answer.confidence, answer.abstained) == (
        "FIND_OBJECT",
        gate.NOT_SURE["en"],
        "none",
        True,
    )
    assert not gemini_calls
    [row] = saved_questions(db, session_id)
    assert row["model"] == "template"
    # The window understand() chose is saved for the report: the whole 30-minute retention.
    assert (row["t_to"], row["t_from"]) == (
        row["asked_at"],
        row["asked_at"] - timedelta(minutes=30),
    )


@pytest.mark.parametrize(("connected", "model"), [(True, "test-gemini"), (False, "template")])
def test_a_question_nothing_matches_uses_the_live_frame_when_there_is_one(
    client: TestClient,
    db: Connection,
    session_id: str,
    gemini_calls: list[dict[str, Any]],
    connected: bool,
    model: str,
) -> None:
    client.app.state.stream_connected = connected

    answer = AskResponse.model_validate(ask_by_text(client, session_id, "en", "Hmm, okay").json())

    assert answer.intent == "MEMORY"
    assert len(gemini_calls) == connected
    [row] = saved_questions(db, session_id)
    assert row["model"] == model


def test_french_is_refused_until_it_has_a_lexicon(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    response = ask_by_text(client, session_id, "fr", "Qui est-ce ?")

    assert_error(response, 400, "BAD_REQUEST", NO_FRENCH)
    assert not gemini_calls
    assert not saved_questions(db, session_id)


def store_frame(
    db: Connection, session_id: str, seconds_ago: float, dhash: int, labels: list[str]
) -> int:
    frame_id = insert_frame(
        db,
        {
            "session_id": session_id,
            "captured_at": datetime.now(UTC) - timedelta(seconds=seconds_ago),
            "image_path": f"frames/{dhash}.jpg",
            "thumb_path": f"thumbs/{dhash}.jpg",
            "width": 960,
            "height": 540,
            "sharpness": 100.0,
            "brightness": 120.0,
            "contrast": 40.0,
            "edge_density": 0.1,
            "dhash": dhash,
            "keep_reason": "novel",
            "clip_emb": unit_vector(1),
        },
    )
    for label in labels:
        db.execute(
            """
            INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
            VALUES (%s, %s, 0.9, ARRAY[0, 0, 1, 1], 0.1, 'center')
            """,
            (frame_id, label),
        )
    return frame_id


def test_a_memory_question_saves_and_publishes_its_ranked_evidence(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    keys = store_frame(db, session_id, seconds_ago=60, dhash=0b0001, labels=["keys"])
    store_frame(db, session_id, seconds_ago=60.5, dhash=-1, labels=[])  # 0.5 s from the keys
    chair = store_frame(db, session_id, seconds_ago=20, dhash=0xF0F0, labels=["chair"])
    store_frame(db, session_id, seconds_ago=40 * 60, dhash=-(2**40), labels=["keys"])  # too old

    with client.websocket_connect("/v1/ws/dashboard") as dashboard:
        dashboard.send_json({"type": "ping"})
        assert dashboard.receive_json() == {"type": "pong"}  # the server has registered this client
        response = ask_by_text(client, session_id, "en", "Where did I leave my keys?")
        client.app.state.hub.publish({"type": "ping"})  # so a missing event fails, not hangs
        event = QueryAnsweredMessage.model_validate(dashboard.receive_json())

    answer = AskResponse.model_validate(response.json())
    assert answer.intent == "FIND_OBJECT"
    assert [item.frame_id for item in answer.evidence] == [keys, chair]
    assert answer.evidence[0].thumb_url == "/media/thumbs/1.jpg"
    assert answer.evidence[0].score > answer.evidence[1].score
    assert "retrieve" in answer.timings_ms
    # shortcut: until VA-72 a memory answer is the template, even with good evidence.
    assert (answer.answer, answer.abstained) == (gate.NOT_SURE["en"], True)
    assert not gemini_calls
    rows = db.execute(
        "SELECT frame_id, rank, score, parts FROM question_evidence WHERE question_id = %s "
        "ORDER BY rank",
        (answer.question_id,),
    ).fetchall()
    assert [(row["frame_id"], row["rank"]) for row in rows] == [(keys, 1), (chair, 2)]
    assert rows[0]["score"] == pytest.approx(answer.evidence[0].score)
    assert rows[0]["parts"]["obj"] == 1  # the only frame in the window with keys
    # The dashboard hears the same evidence, with the score parts that explain the ranking.
    assert [item.model_dump() for item in event.evidence] == [
        {**row, "thumb_url": item.thumb_url}
        for row, item in zip(rows, answer.evidence, strict=True)
    ]
