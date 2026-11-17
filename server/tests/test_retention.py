"""Retention uses the real Postgres schema and a temporary media directory (design §7.5)."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from psycopg import Connection

from server.common.db import insert_frame, insert_question, open_ingest_connection
from server.ingest.retention import RetentionConfig, RetentionJob

CONSENT_VER = "retention-test"


@pytest.fixture
def db() -> Iterator[Connection]:
    connection = open_ingest_connection()
    yield connection
    connection.execute("DELETE FROM sessions WHERE consent_ver = %s", (CONSENT_VER,))
    connection.commit()
    connection.close()


def make_session(db: Connection) -> str:
    return db.execute(
        """
        INSERT INTO sessions (app_lang, consent_at, consent_ver)
        VALUES ('en', now(), %s) RETURNING id
        """,
        (CONSENT_VER,),
    ).fetchone()["id"]


def add_frame(
    db: Connection,
    data_dir: Path,
    session_id: str,
    name: str,
    captured_at: datetime,
    *,
    media: bytes = b"jpeg",
) -> tuple[int, list[Path]]:
    stored_paths = [f"frames/{name}.jpg", f"thumbs/{name}.jpg", f"originals/{name}.jpg"]
    files = [data_dir / path for path in stored_paths]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(media)
    frame_id = insert_frame(
        db,
        {
            "session_id": session_id,
            "captured_at": captured_at,
            "image_path": stored_paths[0],
            "thumb_path": stored_paths[1],
            "original_path": stored_paths[2],
            "width": 2,
            "height": 2,
            "sharpness": 1,
            "brightness": 1,
            "contrast": 1,
            "edge_density": 1,
            "dhash": abs(hash(name)),
            "keep_reason": "novel",
            "clip_emb": np.zeros(512, dtype=np.float32),
        },
    )
    return frame_id, files


def add_question(db: Connection, session_id: str, asked_at: datetime) -> int:
    return insert_question(
        db,
        {
            "session_id": session_id,
            "asked_at": asked_at,
            "input_mode": "text",
            "transcript": "test",
            "lang": "en",
            "intent": "MEMORY",
            "answer": "test",
            "confidence": "low",
            "abstained": False,
            "timings_ms": {},
        },
    )


def test_purge_removes_expired_frame_media_and_question_history(
    db: Connection, tmp_path: Path
) -> None:
    now = datetime.now(UTC)
    session_id = make_session(db)
    expired_id, expired_files = add_frame(
        db, tmp_path, session_id, "expired", now - timedelta(minutes=2)
    )
    current_id, current_files = add_frame(db, tmp_path, session_id, "current", now)
    db.execute(
        """
        INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
        VALUES (%s, 'door', 0.9, ARRAY[0, 0, 1, 1], 1, 'center')
        """,
        (expired_id,),
    )
    current_question = add_question(db, session_id, now)
    db.execute(
        """
        INSERT INTO question_evidence (question_id, frame_id, rank, score, parts)
        VALUES (%s, %s, 1, 1, '{}')
        """,
        (current_question, expired_id),
    )
    old_question = add_question(db, session_id, now - timedelta(hours=25))
    db.commit()

    result = RetentionJob(tmp_path, db, retention_minutes=1, now=lambda: now).purge()

    assert result.frames_deleted == 1
    assert result.questions_deleted == 1
    assert db.execute("SELECT id FROM frames WHERE id = %s", (expired_id,)).fetchone() is None
    assert db.execute("SELECT id FROM frames WHERE id = %s", (current_id,)).fetchone() is not None
    assert (
        db.execute(
            "SELECT count(*) AS n FROM detections WHERE frame_id = %s", (expired_id,)
        ).fetchone()["n"]
        == 0
    )
    assert (
        db.execute(
            "SELECT count(*) AS n FROM question_evidence WHERE question_id = %s",
            (current_question,),
        ).fetchone()["n"]
        == 0
    )
    assert db.execute("SELECT id FROM questions WHERE id = %s", (old_question,)).fetchone() is None
    assert not any(path.exists() for path in expired_files)
    assert all(path.exists() for path in current_files)


def test_safety_cap_deletes_oldest_frames_first(db: Connection, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    session_id = make_session(db)
    oldest_id, oldest_files = add_frame(
        db, tmp_path, session_id, "oldest", now - timedelta(seconds=3)
    )
    middle_id, middle_files = add_frame(
        db, tmp_path, session_id, "middle", now - timedelta(seconds=2)
    )
    newest_id, newest_files = add_frame(
        db, tmp_path, session_id, "newest", now - timedelta(seconds=1)
    )
    db.commit()
    config = RetentionConfig(
        interval_seconds=60, max_frame_count=2, max_frame_bytes=1000, question_retention_hours=24
    )

    result = RetentionJob(
        tmp_path, db, retention_minutes=30, config=config, now=lambda: now
    ).purge()

    assert result.frames_deleted == 1
    assert db.execute("SELECT id FROM frames WHERE id = %s", (oldest_id,)).fetchone() is None
    assert db.execute("SELECT id FROM frames WHERE id = %s", (middle_id,)).fetchone() is not None
    assert db.execute("SELECT id FROM frames WHERE id = %s", (newest_id,)).fetchone() is not None
    assert not any(path.exists() for path in oldest_files)
    assert all(path.exists() for path in middle_files + newest_files)


def test_byte_safety_cap_deletes_oldest_frames_first(db: Connection, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    session_id = make_session(db)
    oldest_id, oldest_files = add_frame(
        db, tmp_path, session_id, "oldest-bytes", now - timedelta(seconds=2), media=b"abc"
    )
    newest_id, newest_files = add_frame(
        db, tmp_path, session_id, "newest-bytes", now - timedelta(seconds=1), media=b"abc"
    )
    db.commit()
    config = RetentionConfig(
        interval_seconds=60, max_frame_count=5_000, max_frame_bytes=9, question_retention_hours=24
    )

    result = RetentionJob(
        tmp_path, db, retention_minutes=30, config=config, now=lambda: now
    ).purge()

    assert result.frames_deleted == 1
    assert db.execute("SELECT id FROM frames WHERE id = %s", (oldest_id,)).fetchone() is None
    assert db.execute("SELECT id FROM frames WHERE id = %s", (newest_id,)).fetchone() is not None
    assert not any(path.exists() for path in oldest_files)
    assert all(path.exists() for path in newest_files)


def test_run_if_due_waits_for_its_interval(db: Connection, tmp_path: Path) -> None:
    ticks = iter((100.0, 120.0, 160.0))
    job = RetentionJob(
        tmp_path,
        db,
        retention_minutes=1,
        config=RetentionConfig(
            interval_seconds=60,
            max_frame_count=5_000,
            max_frame_bytes=2**31,
            question_retention_hours=24,
        ),
        monotonic=lambda: next(ticks),
    )

    assert job.run_if_due() is not None
    assert job.run_if_due() is None
    assert job.run_if_due() is not None
