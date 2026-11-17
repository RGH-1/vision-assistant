"""VA-76 checks: the dashboard's REST routes against the real database, one contract test per route.

Each response is parsed with the contract model (extra fields forbidden), so a missing, extra or
wrongly typed field fails the test. /v1/status and /media are checked in test_main.py and
test_sessions.py.
"""

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg import Connection
from psycopg.types.json import Jsonb

from server import main
from server.common.db import insert_alert, insert_frame, insert_question, open_ingest_connection
from server.common.schemas import Alert, FrameDetail, FrameSummary, QuestionRecord
from server.main import create_app
from server.routes.dashboard import named_labels
from server.settings import Settings

CONSENT_VER = "test-va-76"
NOW = datetime.now(UTC).replace(microsecond=0)


class IdleIngestWorker:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeClipText:
    """Every search means "dimension 0", so a frame's own vector decides its meaning score."""

    def encode(self, text: str) -> np.ndarray:
        return unit_vector(0)


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


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
        gemini_model="stub",
        gemini_fallback_model="stub-lite",
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
def session_id(db: Connection) -> Iterator[str]:
    # Inserted directly, because POST /v1/sessions would end the session someone is using.
    row = db.execute(
        "INSERT INTO sessions (app_lang, consent_at, consent_ver) VALUES ('en', now(), %s) "
        "RETURNING id",
        (CONSENT_VER,),
    ).fetchone()
    yield str(row["id"])
    db.execute("DELETE FROM sessions WHERE consent_ver = %s", (CONSENT_VER,))


def add_frame(
    db: Connection,
    session_id: str,
    name: str,
    seconds_ago: float,
    *,
    vector: int = 5,
    labels: tuple[str, ...] = (),
    texts: tuple[str, ...] = (),
    original: bool = False,
) -> int:
    frame_id = insert_frame(
        db,
        {
            "session_id": session_id,
            "captured_at": NOW - timedelta(seconds=seconds_ago),
            "image_path": f"frames/{name}.jpg",
            "thumb_path": f"thumbs/{name}.jpg",
            "original_path": f"originals/{name}.jpg" if original else None,
            "width": 960,
            "height": 540,
            "sharpness": 120.5,
            "brightness": 110.0,
            "contrast": 45.0,
            "edge_density": 0.12,
            "enhanced": original,
            "enhance_ops": ["clahe"] if original else [],
            "dhash": 1,
            "keep_reason": "text" if texts else "novel",
            "faces_blurred": 1,
            "scene_text": " ".join(texts),
            "clip_emb": unit_vector(vector),
            "timings_ms": {"clip": 12.5},
        },
    )
    for label in labels:
        db.execute(
            "INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position) "
            "VALUES (%s, %s, 0.9, ARRAY[0.1, 0.2, 0.5, 0.9], 0.28, 'left')",
            (frame_id, label),
        )
    for text in texts:
        db.execute(
            "INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox, from_crop) "
            "VALUES (%s, %s, lower(%s), 0.8, ARRAY[0.2, 0.1, 0.6, 0.3], true)",
            (frame_id, text, text),
        )
    return frame_id


def frames(client: TestClient, **params: Any) -> list[FrameSummary]:
    response = client.get("/v1/frames", params=params)
    assert response.status_code == 200, response.text
    return [FrameSummary.model_validate(item) for item in response.json()]


def test_frames_are_newest_first_with_labels_and_text(
    client: TestClient, db: Connection, session_id: str
) -> None:
    old = add_frame(db, session_id, "old", 30, labels=("door",), texts=("Room 204",))
    new = add_frame(db, session_id, "new", 10, labels=("stairs", "person"))

    listed = frames(client, session_id=session_id)

    assert [f.frame_id for f in listed] == [new, old]
    assert listed[0].labels == ["stairs", "person"]
    assert listed[1].text_preview == "Room 204"
    assert listed[1].thumb_url == "/media/thumbs/old.jpg"
    assert listed[1].keep_reason == "text"
    assert all(f.score is None for f in listed)  # set only by a search


def test_frame_filters_and_limit(client: TestClient, db: Connection, session_id: str) -> None:
    door = add_frame(db, session_id, "door", 300, labels=("door",))
    sign = add_frame(db, session_id, "sign", 200, texts=("EXIT",))
    late = add_frame(db, session_id, "late", 100)

    def ids(**params: Any) -> list[int]:
        return [f.frame_id for f in frames(client, session_id=session_id, **params)]

    assert ids(label="door") == [door]
    assert ids(text="exit") == [sign]  # trigram match on the normalized OCR text
    in_range = {"from": (NOW - timedelta(seconds=250)).isoformat(), "to": NOW.isoformat()}
    assert ids(**in_range) == [late, sign]
    assert ids(limit=1) == [late]


def test_search_by_meaning_orders_by_score_and_names_labels(
    client: TestClient, db: Connection, session_id: str
) -> None:
    # Newest first would be: unrelated, meaning, stairs.
    stairs = add_frame(db, session_id, "stairs", 30, vector=0, labels=("stairs",))
    meaning = add_frame(db, session_id, "meaning", 20, vector=0)
    unrelated = add_frame(db, session_id, "unrelated", 10, vector=1)

    found = frames(client, session_id=session_id, q="staircase")

    assert [f.frame_id for f in found] == [stairs, meaning, unrelated]
    assert [f.score for f in found] == pytest.approx([1.9, 1.0, 0.0], abs=0.01)
    # The limit keeps the best matches, not the newest.
    assert [f.frame_id for f in frames(client, session_id=session_id, q="staircase", limit=1)] == [
        stairs
    ]


def test_search_in_a_time_range_is_not_crowded_out_by_frames_outside_it(
    client: TestClient, db: Connection, session_id: str
) -> None:
    # The hybrid SQL keeps its best 30, so the range must go into it, not only into the listing.
    for i in range(30):
        add_frame(db, session_id, f"newer{i}", 10 + i, vector=0)
        add_frame(db, session_id, f"older{i}", 500 + i, vector=0)
    inside = add_frame(db, session_id, "inside", 200, vector=1)
    window = {
        "from": (NOW - timedelta(seconds=300)).isoformat(),
        "to": (NOW - timedelta(seconds=100)).isoformat(),
    }

    assert [f.frame_id for f in frames(client, session_id=session_id, q="door", **window)] == [
        inside
    ]


def test_search_reads_english_and_arabic_object_words() -> None:
    assert named_labels("Staircase") == ["stairs"]
    assert named_labels("درج") == ["stairs"]
    assert named_labels("a nice view") == []


def test_frame_detail_has_quality_numbers_detections_and_ocr(
    client: TestClient, db: Connection, session_id: str
) -> None:
    frame_id = add_frame(
        db, session_id, "detail", 5, labels=("door",), texts=("Room 204",), original=True
    )

    response = client.get(f"/v1/frames/{frame_id}")

    assert response.status_code == 200, response.text
    detail = FrameDetail.model_validate(response.json())
    assert detail.image_url == "/media/frames/detail.jpg"
    assert detail.original_url == "/media/originals/detail.jpg"
    assert (detail.sharpness, detail.enhanced, detail.enhance_ops) == (120.5, True, ["clahe"])
    assert detail.timings_ms == {"clip": 12.5}
    assert [(d.label, d.position) for d in detail.detections] == [("door", "left")]
    assert detail.detections[0].bbox == pytest.approx([0.1, 0.2, 0.5, 0.9])
    assert [(o.text, o.from_crop) for o in detail.ocr] == [("Room 204", True)]


def test_frame_without_original_has_null_original_url(
    client: TestClient, db: Connection, session_id: str
) -> None:
    frame_id = add_frame(db, session_id, "plain", 5)

    assert client.get(f"/v1/frames/{frame_id}").json()["original_url"] is None


def test_unknown_frame_is_not_found(client: TestClient) -> None:
    response = client.get("/v1/frames/999999999")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def add_question(db: Connection, session_id: str, transcript: str, seconds_ago: float) -> int:
    return insert_question(
        db,
        {
            "session_id": session_id,
            "asked_at": NOW - timedelta(seconds=seconds_ago),
            "input_mode": "voice",
            "stt_engine": "groq",
            "transcript": transcript,
            "lang": "en",
            "intent": "READ_TEXT_PAST",
            "t_from": NOW - timedelta(seconds=180),
            "t_to": NOW,
            "answer": "The sign said Room 204.",
            "confidence": "high",
            "abstained": False,
            "safety_rewrite": True,
            "model": "gemini-test",
            "timings_ms": {"stt": 500.0, "total": 3000.0},
        },
    )


def test_questions_are_newest_first_with_ranked_evidence(
    client: TestClient, db: Connection, session_id: str
) -> None:
    first = add_frame(db, session_id, "evidence1", 60)
    second = add_frame(db, session_id, "evidence2", 50)
    old = add_question(db, session_id, "What was on the sign?", 40)
    new = add_question(db, session_id, "Where are my keys?", 20)
    parts = {"clip": 0.9, "text": 1.0, "obj": 0.0, "rec": 0.5, "q": 0.7}
    # Rank 1 is the newer frame and goes in last, so only ORDER BY rank puts it first.
    for rank, frame_id in ((2, first), (1, second)):
        db.execute(
            "INSERT INTO question_evidence (question_id, frame_id, rank, score, parts) "
            "VALUES (%s, %s, %s, 0.75, %s)",
            (old, frame_id, rank, Jsonb(parts)),
        )

    response = client.get("/v1/questions", params={"session_id": session_id})

    assert response.status_code == 200, response.text
    listed = [QuestionRecord.model_validate(item) for item in response.json()]
    assert [q.question_id for q in listed] == [new, old]
    assert listed[0].evidence == []
    assert [(e.frame_id, e.rank) for e in listed[1].evidence] == [(second, 1), (first, 2)]
    assert listed[1].evidence[0].thumb_url == "/media/thumbs/evidence2.jpg"
    assert listed[1].evidence[0].parts.model_dump() == pytest.approx(parts)
    assert (listed[1].safety_rewrite, listed[1].model) == (True, "gemini-test")
    limited = client.get("/v1/questions", params={"session_id": session_id, "limit": 1}).json()
    assert [q["question_id"] for q in limited] == [new]


def test_alerts_are_newest_first_with_a_thumbnail_when_the_frame_exists(
    client: TestClient, db: Connection, session_id: str
) -> None:
    frame_id = add_frame(db, session_id, "alert", 5)
    with_frame = insert_alert(
        db,
        {
            "session_id": session_id,
            "frame_id": frame_id,
            "kind": "door_ahead",
            "message": "Door ahead.",
            "lang": "en",
        },
    )
    without = insert_alert(
        db,
        {"session_id": session_id, "kind": "stairs_ahead", "message": "درج أمامك.", "lang": "ar"},
    )

    response = client.get("/v1/alerts", params={"session_id": session_id})

    assert response.status_code == 200, response.text
    listed = [Alert.model_validate(item) for item in response.json()]
    assert [a.alert_id for a in listed] == [without, with_frame]
    assert (listed[0].frame_id, listed[0].thumb_url) == (None, None)
    assert listed[1].thumb_url == "/media/thumbs/alert.jpg"
    limited = client.get("/v1/alerts", params={"session_id": session_id, "limit": 1}).json()
    assert [a["alert_id"] for a in limited] == [without]


@pytest.mark.parametrize("path", ["/v1/frames", "/v1/questions", "/v1/alerts"])
@pytest.mark.parametrize(
    "params",
    [
        {},  # session_id is required
        {"session_id": "not-a-uuid"},
        {"session_id": "3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01", "limit": 0},
    ],
)
def test_bad_list_requests_return_bad_request(
    client: TestClient, path: str, params: dict[str, Any]
) -> None:
    response = client.get(path, params=params)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_a_time_without_a_timezone_is_rejected(client: TestClient) -> None:
    params = {"session_id": "3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01", "from": "2026-11-17T10:00:00"}

    assert client.get("/v1/frames", params=params).status_code == 400
