from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
import pytest
from psycopg import Connection

from server.common.db import (
    TRIGRAM_SIMILARITY_THRESHOLD,
    hybrid_search_frames,
    insert_frame,
    open_ingest_connection,
)


@pytest.fixture
def search_data() -> tuple[Connection, UUID, datetime, list[int]]:
    connection = open_ingest_connection()
    session_id = None
    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'hybrid-search-test')
            RETURNING id
            """
        ).fetchone()["id"]
        captured_at = datetime.now(UTC).replace(microsecond=0)
        ids = [
            _insert_test_frame(connection, session_id, captured_at, "match", _vector(0)),
            _insert_test_frame(connection, session_id, captured_at, "other", _vector(1)),
            _insert_test_frame(
                connection, session_id, captured_at - timedelta(minutes=31), "old", _vector(0)
            ),
        ]
        connection.execute(
            """
            INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox, from_crop)
            VALUES (%s, 'Room 204', 'room 204', 0.9, ARRAY[0, 0, 1, 1], false)
            """,
            (ids[0],),
        )
        connection.execute(
            """
            INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
            VALUES (%s, 'door', 0.95, ARRAY[0, 0, 1, 1], 0.1, 'center')
            """,
            (ids[0],),
        )
        connection.execute(
            """
            INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
            VALUES (%s, 'door', 0.4, ARRAY[0, 0, 1, 1], 0.1, 'left')
            """,
            (ids[1],),
        )
        connection.commit()
        yield connection, session_id, captured_at, ids
    finally:
        if session_id is not None:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()
        connection.close()


def _vector(index: int) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[index] = 1
    return vector


def _insert_test_frame(
    connection: Connection, session_id: UUID, captured_at: datetime, name: str, vector: np.ndarray
) -> int:
    return insert_frame(
        connection,
        {
            "session_id": session_id,
            "captured_at": captured_at,
            "image_path": f"frames/{name}.jpg",
            "thumb_path": f"thumbs/{name}.jpg",
            "width": 1280,
            "height": 720,
            "sharpness": 90.0,
            "brightness": 120.0,
            "contrast": 40.0,
            "edge_density": 0.1,
            "dhash": {"match": 11, "other": 22, "old": 33}[name],
            "keep_reason": "heartbeat",
            "clip_emb": vector,
        },
    )


def test_hybrid_search_combines_clip_trigram_ocr_objects_and_time(search_data) -> None:
    connection, session_id, captured_at, ids = search_data

    rows = hybrid_search_frames(
        connection,
        session_id,
        time_from=captured_at - timedelta(seconds=1),
        time_to=captured_at + timedelta(seconds=1),
        clip_vector=_vector(0),
        text="R00m 204",
        target_labels=["door"],
        asked_at=captured_at + timedelta(seconds=5),
    )

    assert [row["id"] for row in rows] == [ids[0], ids[1]]
    best = rows[0]
    assert best["dhash"] == 11
    assert best["sharpness"] == 90.0
    assert best["s_clip"] == pytest.approx(1)
    assert best["s_text"] > 0.2
    assert best["s_obj"] == pytest.approx(0.95)
    assert best["seconds_ago"] == pytest.approx(5)
    assert rows[1]["s_text"] == 0
    assert rows[1]["s_obj"] == pytest.approx(0.4)


def test_hybrid_search_configures_the_noisy_ocr_threshold(search_data) -> None:
    connection, *_ = search_data

    value = connection.execute("SHOW pg_trgm.similarity_threshold").fetchone()[
        "pg_trgm.similarity_threshold"
    ]

    assert float(value) == TRIGRAM_SIMILARITY_THRESHOLD


def test_hybrid_search_validates_time_and_vector(search_data) -> None:
    connection, session_id, captured_at, _ = search_data
    params = {
        "session_id": session_id,
        "time_from": captured_at,
        "time_to": captured_at,
        "clip_vector": _vector(0),
        "text": "",
        "target_labels": [],
        "asked_at": captured_at,
    }

    with pytest.raises(ValueError, match="time_from"):
        hybrid_search_frames(
            connection,
            **{**params, "time_from": captured_at + timedelta(seconds=1)},
        )
    with pytest.raises(ValueError, match="512"):
        hybrid_search_frames(connection, **{**params, "clip_vector": np.zeros(511)})
