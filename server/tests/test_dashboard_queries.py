from datetime import UTC, datetime, timedelta
from time import perf_counter

import numpy as np
import pytest

from server.common.db import (
    get_dashboard_frame_detail,
    insert_alert,
    insert_frame,
    insert_question,
    list_dashboard_alerts,
    list_dashboard_frames,
    list_questions_with_evidence,
    open_ingest_connection,
)


@pytest.fixture
def dashboard_data():
    connection = open_ingest_connection()
    session_id = None
    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'dashboard-query-test')
            RETURNING id
            """
        ).fetchone()["id"]
        captured_at = datetime.now(UTC).replace(microsecond=0)
        frame_ids = []
        for index, scene_text in enumerate(("room 204", "door near stairs", "exit sign")):
            frame_ids.append(
                insert_frame(
                    connection,
                    {
                        "session_id": session_id,
                        "captured_at": captured_at + timedelta(seconds=index),
                        "image_path": f"frames/{index}.jpg",
                        "thumb_path": f"thumbs/{index}.jpg",
                        "width": 1280,
                        "height": 720,
                        "sharpness": 100.0 + index,
                        "brightness": 120.0,
                        "contrast": 40.0,
                        "edge_density": 0.1,
                        "dhash": index,
                        "keep_reason": "heartbeat",
                        "scene_text": scene_text,
                        "clip_emb": np.eye(1, 512, dtype=np.float32)[0],
                        "timings_ms": {"clip": 1.0},
                    },
                )
            )
        connection.execute(
            """
            INSERT INTO detections (frame_id, label, confidence, bbox, area_ratio, position)
            VALUES (%s, 'door', 0.9, ARRAY[1, 2, 3, 4], 0.2, 'center')
            """,
            (frame_ids[1],),
        )
        connection.execute(
            """
            INSERT INTO ocr_texts (frame_id, text, text_norm, confidence, bbox, from_crop)
            VALUES (%s, 'Room 204', 'room 204', 0.9, ARRAY[1, 2, 3, 4], false)
            """,
            (frame_ids[0],),
        )
        question_id = insert_question(
            connection,
            {
                "session_id": session_id,
                "asked_at": captured_at + timedelta(seconds=4),
                "input_mode": "text",
                "transcript": "What was written?",
                "lang": "en",
                "intent": "READ_TEXT_PAST",
                "answer": "Room 204.",
                "confidence": "high",
                "abstained": False,
                "timings_ms": {"total": 1.0},
            },
        )
        connection.execute(
            """
            INSERT INTO question_evidence (question_id, frame_id, rank, score, parts)
            VALUES (%s, %s, 1, 0.9, '{"clip": 0.2, "text": 0.5, "obj": 0, "rec": 0.1, "q": 0.1}')
            """,
            (question_id, frame_ids[0]),
        )
        insert_alert(
            connection,
            {
                "session_id": session_id,
                "frame_id": frame_ids[1],
                "kind": "door_ahead",
                "message": "Door ahead.",
                "lang": "en",
            },
        )
        connection.commit()
        yield connection, session_id, frame_ids, captured_at
    finally:
        if session_id is not None:
            connection.rollback()
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()
        connection.close()


def test_timeline_filters_by_time_label_text_and_page(dashboard_data) -> None:
    connection, session_id, frame_ids, captured_at = dashboard_data

    timeline = list_dashboard_frames(
        connection,
        session_id,
        captured_from=captured_at,
        captured_to=captured_at + timedelta(seconds=1),
        limit=1,
        offset=0,
    )
    by_label = list_dashboard_frames(connection, session_id, label="door")
    by_text = list_dashboard_frames(connection, session_id, text="r00m 204")

    assert [frame["frame_id"] for frame in timeline] == [frame_ids[1]]
    assert [frame["frame_id"] for frame in by_label] == [frame_ids[1]]
    assert [frame["frame_id"] for frame in by_text] == [frame_ids[0]]
    assert by_label[0]["labels"] == ["door"]


def test_frame_detail_includes_quality_detections_and_ocr(dashboard_data) -> None:
    connection, _session_id, frame_ids, _captured_at = dashboard_data

    detail = get_dashboard_frame_detail(connection, frame_ids[0])

    assert detail is not None
    assert detail["sharpness"] == 100.0
    assert detail["ocr"] == [
        {"text": "Room 204", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0], "from_crop": False}
    ]
    assert get_dashboard_frame_detail(connection, -1) is None


def test_questions_include_evidence_and_alerts_include_thumbnail(dashboard_data) -> None:
    connection, session_id, frame_ids, _captured_at = dashboard_data

    questions = list_questions_with_evidence(connection, session_id)
    alerts = list_dashboard_alerts(connection, session_id)

    assert questions[0]["evidence"][0]["frame_id"] == frame_ids[0]
    assert questions[0]["evidence"][0]["thumb_path"] == "thumbs/0.jpg"
    assert alerts[0]["frame_id"] == frame_ids[1]
    assert alerts[0]["thumb_path"] == "thumbs/1.jpg"


def test_dashboard_queries_are_under_100ms_for_a_30_minute_session(dashboard_data) -> None:
    connection, session_id, frame_ids, captured_at = dashboard_data
    vector = np.eye(1, 512, dtype=np.float32)[0]
    connection.execute(
        """
        INSERT INTO frames (
            session_id, captured_at, image_path, thumb_path, width, height, sharpness,
            brightness, contrast, edge_density, dhash, keep_reason, clip_emb, timings_ms
        )
        SELECT
            %s, %s + item * interval '1 second',
            'frames/perf-' || item || '.jpg', 'thumbs/perf-' || item || '.jpg',
            1280, 720, 100, 120, 40, 0.1, item, 'heartbeat', %s, '{}'
        FROM generate_series(0, 1799) AS item
        """,
        (session_id, captured_at, vector),
    )
    connection.commit()

    queries = {
        "timeline": lambda: list_dashboard_frames(connection, session_id, limit=100),
        "detail": lambda: get_dashboard_frame_detail(connection, frame_ids[0]),
        "questions": lambda: list_questions_with_evidence(connection, session_id),
        "alerts": lambda: list_dashboard_alerts(connection, session_id),
    }
    for name, query in queries.items():
        started_at = perf_counter()
        result = query()
        elapsed_s = perf_counter() - started_at

        assert result, name
        assert elapsed_s < 0.1, f"{name} took {elapsed_s * 1000:.1f} ms"
