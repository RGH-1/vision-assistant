import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from psycopg import Connection

from server import main
from server.common.db import insert_alert, insert_frame, insert_question, open_ingest_connection
from server.main import create_app
from server.settings import Settings

CONSENT_VER = "test-va-62"
NEW_SESSION = {"app_lang": "en", "consent_at": "2026-11-06T10:00:00Z", "consent_ver": CONSENT_VER}


@pytest.fixture(scope="module")
def db() -> Iterator[Connection]:
    conn = open_ingest_connection()
    conn.commit()
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


# Module scope because connecting to Postgres is slow here, so the tests share one app.
@pytest.fixture(scope="module")
def client(data_dir: Path) -> Iterator[TestClient]:
    settings = Settings(
        data_dir=data_dir,
        retention_minutes=30,
        host="127.0.0.1",
        port=8000,
        mediamtx_webrtc_port=8889,
        cors_origins=[],
        clip_text_model="stub",
        status_interval_s=60,
        ws_ping_interval_s=60,
        ws_send_timeout_s=1,
    )
    # Same event loop as server/__main__.py, because async psycopg can't use the Windows default.
    options = {"loop_factory": asyncio.SelectorEventLoop}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main, "load_clip_text_encoder", lambda _: "clip-stub")
        with TestClient(create_app(settings), backend_options=options) as test_client:
            yield test_client


@pytest.fixture(autouse=True)
def remove_test_sessions(db: Connection) -> Iterator[None]:
    yield
    db.execute("DELETE FROM sessions WHERE consent_ver = %s", (CONSENT_VER,))


def new_session(client: TestClient) -> str:
    return client.post("/v1/sessions", json=NEW_SESSION).json()["session_id"]


def session_row(db: Connection, session_id: str) -> dict[str, Any] | None:
    return db.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()


def add_session_data(db: Connection, session_id: str, data_dir: Path, name: str) -> list[Path]:
    """Save a frame with its files, a question and an alert, as ingest and /v1/ask will later."""
    stored = [f"frames/{name}.jpg", f"thumbs/{name}.jpg", f"originals/{name}.jpg"]
    files = [data_dir / path for path in stored]
    for file in files:
        file.parent.mkdir(exist_ok=True)
        file.write_bytes(b"jpeg")

    now = datetime.now(UTC)
    insert_frame(
        db,
        {
            "session_id": session_id,
            "captured_at": now,
            "image_path": stored[0],
            "thumb_path": stored[1],
            "original_path": stored[2],
            "width": 960,
            "height": 540,
            "sharpness": 1,
            "brightness": 1,
            "contrast": 1,
            "edge_density": 1,
            "dhash": 1,
            "keep_reason": "novel",
            "clip_emb": np.zeros(512),
        },
    )
    insert_question(
        db,
        {
            "session_id": session_id,
            "asked_at": now,
            "input_mode": "text",
            "transcript": "test",
            "lang": "en",
            "intent": "LIVE_SCENE",
            "answer": "test",
            "confidence": "low",
            "abstained": False,
            "timings_ms": {},
        },
    )
    insert_alert(
        db, {"session_id": session_id, "kind": "door_ahead", "message": "test", "lang": "en"}
    )
    return files


def row_counts(db: Connection, session_id: str) -> list[int]:
    return [
        db.execute(
            f"SELECT count(*) AS n FROM {table} WHERE session_id = %s", (session_id,)
        ).fetchone()["n"]
        for table in ("frames", "questions", "alerts")
    ]


def test_create_stores_consent_and_returns_urls_for_the_callers_host(
    client: TestClient, db: Connection
) -> None:
    response = client.post("/v1/sessions", json=NEW_SESSION)

    assert response.status_code == 201
    body = response.json()
    assert body["whip_url"] == "http://testserver:8889/phone/whip"
    assert body["ws_url"] == f"ws://testserver:8000/v1/ws/phone?session_id={body['session_id']}"

    row = session_row(db, body["session_id"])
    assert row["app_lang"] == "en"
    assert row["consent_at"] == datetime(2026, 11, 6, 10, tzinfo=UTC)
    assert row["consent_ver"] == CONSENT_VER
    assert row["ended_at"] is None


def test_creating_a_session_ends_the_previous_one(client: TestClient, db: Connection) -> None:
    first = new_session(client)
    second = new_session(client)

    assert session_row(db, first)["ended_at"] is not None
    assert session_row(db, second)["ended_at"] is None


def test_end_keeps_the_data_and_the_first_end_time(
    client: TestClient, db: Connection, data_dir: Path
) -> None:
    session_id = new_session(client)
    files = add_session_data(db, session_id, data_dir, "kept")
    client.app.state.live_buffer.add(np.zeros((540, 960, 3), dtype=np.uint8), 1.0)

    assert client.post(f"/v1/sessions/{session_id}/end").status_code == 204
    first_end = session_row(db, session_id)["ended_at"]
    assert first_end is not None
    assert client.post(f"/v1/sessions/{session_id}/end").status_code == 204

    assert session_row(db, session_id)["ended_at"] == first_end
    assert row_counts(db, session_id) == [1, 1, 1]
    assert all(file.exists() for file in files)
    assert len(client.app.state.live_buffer) == 0


def test_delete_leaves_no_rows_or_files_and_spares_other_sessions(
    client: TestClient, db: Connection, data_dir: Path
) -> None:
    other = new_session(client)
    other_files = add_session_data(db, other, data_dir, "other")
    session_id = new_session(client)
    files = add_session_data(db, session_id, data_dir, "deleted")
    client.app.state.live_buffer.add(np.zeros((540, 960, 3), dtype=np.uint8), 1.0)

    assert client.delete(f"/v1/sessions/{session_id}").status_code == 204

    assert session_row(db, session_id) is None
    assert row_counts(db, session_id) == [0, 0, 0]
    assert not any(file.exists() for file in files)
    assert row_counts(db, other) == [1, 1, 1]
    assert all(file.exists() for file in other_files)
    assert len(client.app.state.live_buffer) == 0


def test_failed_file_delete_keeps_the_rows_and_can_be_retried(
    client: TestClient, db: Connection, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = new_session(client)
    files = add_session_data(db, session_id, data_dir, "locked")
    files[0].unlink()  # the retry must also cope with a file that is already gone

    def fail(*_: object) -> None:
        raise PermissionError("file in use")

    with monkeypatch.context() as patch:
        patch.setattr("server.routes.sessions.delete_files", fail)
        with pytest.raises(PermissionError):
            client.delete(f"/v1/sessions/{session_id}")
    assert row_counts(db, session_id) == [1, 1, 1]

    assert client.delete(f"/v1/sessions/{session_id}").status_code == 204
    assert row_counts(db, session_id) == [0, 0, 0]
    assert not any(file.exists() for file in files)


@pytest.mark.parametrize("method, path", [("post", "/{id}/end"), ("delete", "/{id}")])
def test_unknown_session_returns_not_found(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, "/v1/sessions" + path.format(id=uuid4()))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize(
    "body",
    [
        {**NEW_SESSION, "consent_at": "2026-11-06T10:00:00"},  # no timezone
        {**NEW_SESSION, "app_lang": "de"},
        {**NEW_SESSION, "extra_field": 1},
        {"app_lang": "en"},
    ],
)
def test_bad_create_request_returns_bad_request(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/v1/sessions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"
    assert response.json()["error"]["speak"]


def test_status_reports_the_active_session_and_frame_count(
    client: TestClient, db: Connection, data_dir: Path
) -> None:
    before = client.get("/v1/status").json()["frames_in_db"]
    session_id = new_session(client)
    add_session_data(db, session_id, data_dir, "status")

    status = client.get("/v1/status").json()

    assert status["session_id"] == session_id
    assert status["frames_in_db"] == before + 1
    assert status["last_answer_at"] is not None


def test_session_id_that_is_not_a_uuid_returns_bad_request(client: TestClient) -> None:
    assert client.delete("/v1/sessions/not-a-uuid").status_code == 400
