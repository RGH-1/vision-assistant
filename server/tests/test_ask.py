"""VA-64/72 checks: /v1/ask and /v1/ask_text with the real database. Groq and Gemini are faked."""

import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, get_args
from uuid import UUID, uuid4

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg import Connection

from server import main
from server.assistant import gate, gemini, safety, stt
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
        gemini_fallback_model="test-gemini-lite",
        gemini_timeout_s=1,
        gemini_temperature=0.2,
        gemini_max_side=1024,
        answer_max_sentences=2,
        answer_max_words=30,
        evidence_min=0.35,
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

    async def fake(frames, question, intent, window, lang, settings) -> tuple[Any, str]:
        calls.append({"frames": frames, "question": question, "window": window, "lang": lang})
        return REPLY, settings.gemini_model

    monkeypatch.setattr(gemini, "answer", fake)
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
    [frame] = gemini_calls[0]["frames"]
    assert frame.jpeg.startswith(b"\xff\xd8")  # a JPEG from the live buffer
    assert (frame.frame_id, gemini_calls[0]["window"]) == (None, "now")
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


def test_low_confidence_counts_as_abstained_and_the_model_that_answered_is_saved(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unsure_on_flash_lite(*_: Any) -> tuple[gemini.GeminiAnswer, str]:
        return REPLY.model_copy(update={"confidence": "low"}), "test-gemini-lite"

    monkeypatch.setattr(gemini, "answer", unsure_on_flash_lite)

    answer = AskResponse.model_validate(ask_by_text(client, session_id).json())
    assert (answer.confidence, answer.abstained) == ("low", True)
    [row] = saved_questions(db, session_id)
    assert row["model"] == "test-gemini-lite"


def test_an_answer_that_says_safe_is_rewritten_saved_and_published_as_such(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def says_safe(*_: Any) -> tuple[gemini.GeminiAnswer, str]:
        reply = "There is a door ahead. You can walk through it safely. It is open."
        return REPLY.model_copy(update={"answer": reply}), "test-gemini"

    monkeypatch.setattr(gemini, "answer", says_safe)

    with client.websocket_connect("/v1/ws/dashboard") as dashboard:
        dashboard.send_json({"type": "ping"})
        assert dashboard.receive_json() == {"type": "pong"}  # the server has registered this client
        response = ask_by_text(client, session_id)
        client.app.state.hub.publish({"type": "ping"})  # so a missing event fails, not hangs
        event = QueryAnsweredMessage.model_validate(dashboard.receive_json())

    # Cut to 2 sentences first, then the unsafe one is replaced.
    expected = f"There is a door ahead. {safety.CANE['en']}"
    assert AskResponse.model_validate(response.json()).answer == expected
    [row] = saved_questions(db, session_id)
    assert (row["answer"], row["safety_rewrite"]) == (expected, True)
    assert (event.answer, event.safety_rewrite) == (expected, True)


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


def fail_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing(*_: Any) -> tuple[gemini.GeminiAnswer, str]:
        raise RuntimeError("both Gemini models failed")

    monkeypatch.setattr(gemini, "answer", failing)


def test_when_gemini_fails_a_live_question_describes_the_newest_stored_frame(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_gemini(monkeypatch)
    store_frame(db, session_id, seconds_ago=30, dhash=1, labels=["chair"])
    newest = store_frame(db, session_id, seconds_ago=4, dhash=2, labels=["door"])
    add_text(db, newest, "EXIT")

    answer = AskResponse.model_validate(ask_by_text(client, session_id).json())

    expected = "4 seconds ago I saw a door ahead and a sign that says 'EXIT'."
    assert (answer.intent, answer.answer, answer.confidence) == ("LIVE_SCENE", expected, "medium")
    assert not answer.abstained
    assert "gemini" in answer.timings_ms  # the time spent before falling back is still reported
    [row] = saved_questions(db, session_id)
    assert (row["answer"], row["model"]) == (expected, "template")


def test_when_gemini_fails_a_memory_question_describes_its_best_frame(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_gemini(monkeypatch)
    keys = store_frame(db, session_id, seconds_ago=60, dhash=0b0001, labels=["keys"])
    store_frame(db, session_id, seconds_ago=20, dhash=0xF0F0, labels=["chair"])  # newer, worse
    write_image(client, 0b0001)
    write_image(client, 0xF0F0)
    add_text(db, keys, "Room 204")

    answer = AskResponse.model_validate(
        ask_by_text(client, session_id, "ar", "وين تركت مفاتيحي؟").json()
    )

    expected = "قبل 60 ثانية رأيت: مفتاح أمامك ولافتة مكتوب عليها «Room 204»."
    assert (answer.intent, answer.answer) == ("FIND_OBJECT", expected)
    assert answer.evidence[0].frame_id == keys
    [row] = saved_questions(db, session_id)
    assert row["model"] == "template"


def test_when_gemini_fails_and_nothing_is_stored_the_reply_is_not_sure(
    client: TestClient, db: Connection, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_gemini(monkeypatch)
    # Another session's frames are never described in this one.
    other = db.execute(
        "INSERT INTO sessions (app_lang, consent_at, consent_ver) VALUES ('en', now(), %s) "
        "RETURNING id",
        (CONSENT_VER,),
    ).fetchone()["id"]  # type: ignore
    store_frame(db, str(other), seconds_ago=1, dhash=3, labels=["door"])

    answer = AskResponse.model_validate(ask_by_text(client, session_id).json())

    assert (answer.answer, answer.confidence, answer.abstained) == (
        gate.NOT_SURE["en"],
        "none",
        True,
    )
    [row] = saved_questions(db, session_id)
    assert row["model"] == "template"


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


def add_text(db: Connection, frame_id: int, text: str) -> None:
    db.execute(
        "INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox) "
        "VALUES (%s, %s, lower(%s), 0.8, ARRAY[0, 0, 1, 1])",
        (frame_id, text, text),
    )


def write_image(client: TestClient, dhash: int) -> None:
    """The JPEG store_frame's row points at, as FrameStore would have written it."""
    path = client.app.state.settings.data_dir / "frames" / f"{dhash}.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), np.zeros((540, 960, 3), np.uint8))


def test_a_memory_question_saves_and_publishes_its_ranked_evidence(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    keys = store_frame(db, session_id, seconds_ago=60, dhash=0b0001, labels=["keys"])
    store_frame(db, session_id, seconds_ago=60.5, dhash=-1, labels=[])  # 0.5 s from the keys
    chair = store_frame(db, session_id, seconds_ago=20, dhash=0xF0F0, labels=["chair"])
    store_frame(db, session_id, seconds_ago=40 * 60, dhash=-(2**40), labels=["keys"])  # too old
    write_image(client, 0b0001)
    write_image(client, 0xF0F0)
    db.execute(
        "INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox) "
        "VALUES (%s, 'Room 204', 'room 204', 0.8, ARRAY[0, 0, 1, 1])",
        (keys,),
    )

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
    # Strong evidence, so Gemini answers from the kept frames and what ingest found in them.
    assert (answer.answer, answer.abstained) == (REPLY.answer, False)
    [call] = gemini_calls
    assert [frame.frame_id for frame in call["frames"]] == [keys, chair]
    keys_found = {"label": "keys", "position": "center", "confidence": 0.9}
    assert call["frames"][0].detections == (keys_found,)
    assert (call["frames"][0].texts, call["frames"][1].texts) == (("Room 204",), ())
    assert call["window"] == "last 1800 seconds"  # the whole retention window
    assert "gemini" in answer.timings_ms
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


def test_weak_memory_evidence_is_not_sure_without_gemini(
    client: TestClient, db: Connection, session_id: str, gemini_calls: list[dict[str, Any]]
) -> None:
    # The only frame in the window shows no keys and isn't recent, so it scores below
    # EVIDENCE_MIN, and Gemini could only guess from it (§6.5.5).
    chair = store_frame(db, session_id, seconds_ago=20 * 60, dhash=0xF0F0, labels=["chair"])
    write_image(client, 0xF0F0)

    answer = AskResponse.model_validate(
        ask_by_text(client, session_id, "en", "Where did I leave my keys?").json()
    )

    assert [item.frame_id for item in answer.evidence] == [chair]
    assert answer.evidence[0].score < client.app.state.settings.evidence_min
    assert (answer.answer, answer.confidence) == (gate.NOT_SURE["en"], "none")
    assert not gemini_calls
    [row] = saved_questions(db, session_id)
    assert row["model"] == "template"
