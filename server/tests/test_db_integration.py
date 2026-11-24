
from datetime import datetime, timedelta, timezone

import numpy as np

import asyncio
import selectors
import sys

from server.common.db import (
    open_ingest_connection,
    create_web_pool,
    insert_frame,
    get_frame,
    list_frames,
    insert_question,
    insert_alert,
)

def test_insert_and_get_frame_with_vector():

    conn = open_ingest_connection()

    try:
        # 1. Create a temporary session.
        session = conn.execute(
            """
            INSERT INTO sessions (
                app_lang,
                consent_at,
                consent_ver
            )
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()

        session_id = session["id"]

        # 2. Create a vector containing 512 numbers.
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0

        # 3. Prepare a sample frame.
        frame = {
            "session_id": session_id,
            "captured_at": datetime.now(timezone.utc),
            "image_path": "test/frame.jpg",
            "thumb_path": "test/thumb.jpg",
            "width": 1280,
            "height": 720,
            "sharpness": 100.0,
            "brightness": 120.0,
            "contrast": 40.0,
            "edge_density": 0.15,
            "dhash": 12345,
            "keep_reason": "heartbeat",
            "scene_text": "A wooden door",
            "clip_emb": vector,
            "timings_ms": {"clip": 30.0},
        }

        # 4. Save the frame using our helper.
        frame_id = insert_frame(conn, frame)

        # 5. Retrieve the same frame.
        saved = get_frame(conn, frame_id)

        # 6. Automatically verify the result.
        assert saved is not None
        assert saved["id"] == frame_id
        assert saved["session_id"] == session_id
        assert saved["width"] == 1280
        assert saved["height"] == 720
        assert saved["scene_text"] == "A wooden door"

        # 7. Verify the 512-dimensional CLIP vector.
        saved_vector = saved["clip_emb"].to_numpy()

        assert saved_vector.shape == (512,)

        np.testing.assert_allclose(
            saved_vector,
            vector,
        )

    finally:
        # 8. Remove test data, even if an assertion fails.
        conn.rollback()
        conn.close()



def test_list_frames_order_and_pagination():

    conn = open_ingest_connection()

    try:
        # 1. Create a temporary session.
        session = conn.execute(
            """
            INSERT INTO sessions (
                app_lang, consent_at, consent_ver
            )
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()

        session_id = session["id"]

        # 2. Insert five frames.
        base_time = datetime.now(timezone.utc)
        inserted_ids = []

        for i in range(5):
            frame = {
                "session_id": session_id,
                "captured_at": base_time + timedelta(seconds=i),
                "image_path": f"test/frame_{i}.jpg",
                "thumb_path": f"test/thumb_{i}.jpg",
                "width": 1280,
                "height": 720,
                "sharpness": 100.0,
                "brightness": 120.0,
                "contrast": 40.0,
                "edge_density": 0.15,
                "dhash": i,
                "keep_reason": "heartbeat",
                "scene_text": f"Frame {i}",
                "clip_emb": np.zeros(512, dtype=np.float32),
                "timings_ms": {},
            }

            frame_id = insert_frame(conn, frame)
            inserted_ids.append(frame_id)

        # 3. Request the newest three frames.
        latest = list_frames(
            conn,
            session_id,
            limit=3,
        )

        latest_ids = [frame["id"] for frame in latest]

        assert latest_ids == inserted_ids[-3:][::-1]

        # 4. Skip three frames and retrieve the older two.
        older = list_frames(
            conn,
            session_id,
            limit=3,
            offset=3,
        )

        older_ids = [frame["id"] for frame in older]

        assert older_ids == inserted_ids[:2][::-1]

        # 5. Verify all five frames exist.
        all_frames = list_frames(
            conn,
            session_id,
            limit=10,
        )

        assert len(all_frames) == 5

    finally:
        # 6. Remove all temporary test data.
        conn.rollback()
        conn.close()



def test_insert_question():

    conn = open_ingest_connection()

    try:
        # 1. Create a temporary session.
        session = conn.execute(
            """
            INSERT INTO sessions (
                app_lang, consent_at, consent_ver
            )
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()

        session_id = session["id"]

        # 2. Prepare a sample question and answer.
        question = {
            "session_id": session_id,
            "asked_at": datetime.now(timezone.utc),
            "input_mode": "text",
            "transcript": "What is in front of me?",
            "lang": "en",
            "intent": "describe_scene",
            "answer": "There is a wooden door.",
            "confidence": "high",
            "abstained": False,
            "timings_ms": {
                "total": 250.0,
            },
        }

        # 3. Save the question.
        question_id = insert_question(conn, question)

        # 4. Retrieve the saved question.
        saved = conn.execute(
            """
            SELECT *
            FROM questions
            WHERE id = %s
            """,
            (question_id,),
        ).fetchone()

        # 5. Check that the information matches.
        assert saved is not None
        assert saved["id"] == question_id
        assert saved["session_id"] == session_id
        assert saved["transcript"] == question["transcript"]
        assert saved["answer"] == question["answer"]
        assert saved["input_mode"] == "text"
        assert saved["confidence"] == "high"
        assert saved["abstained"] is False

        # Check the JSONB processing time.
        assert saved["timings_ms"]["total"] == 250.0

        # Check that optional fields can be empty.
        assert saved["stt_engine"] is None

    finally:
        # 6. Undo test changes and close connection.
        conn.rollback()
        conn.close()



def test_insert_alert():

    conn = open_ingest_connection()

    try:
        # 1. Create a temporary session.
        session = conn.execute(
            """
            INSERT INTO sessions (
                app_lang, consent_at, consent_ver
            )
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()

        session_id = session["id"]

        # 2. Prepare our sample warning.
        alert = {
            "session_id": session_id,
            "kind": "obstacle",
            "message": "Warning! Stairs ahead.",
            "lang": "en",
        }

        # 3. Save the alert using our helper.
        alert_id = insert_alert(conn, alert)

        # 4. Retrieve the alert from PostgreSQL.
        saved = conn.execute(
            """
            SELECT *
            FROM alerts
            WHERE id = %s
            """,
            (alert_id,),
        ).fetchone()

        # 5. Verify the saved information.
        assert saved is not None
        assert saved["id"] == alert_id
        assert saved["session_id"] == session_id
        assert saved["frame_id"] is None
        assert saved["kind"] == "obstacle"
        assert saved["message"] == "Warning! Stairs ahead."
        assert saved["lang"] == "en"
        assert saved["created_at"] is not None

    finally:
        # 6. Remove temporary changes.
        conn.rollback()
        conn.close()



def test_async_connection_pool():

    async def run_test():
        # 1. Create the pool.
        pool = create_web_pool()

        try:
            # 2. Open the pool.
            await pool.open(wait=True)

            # 3. Borrow a connection.
            async with pool.connection() as conn:

                # 4. Check the PostgreSQL connection.
                result = await conn.execute(
                    "SELECT 1 AS number"
                )

                row = await result.fetchone()

                assert row["number"] == 1

                # 5. Check the async pgvector adapter.
                result = await conn.execute(
                    "SELECT '[1,2,3]'::vector AS embedding"
                )

                row = await result.fetchone()

                saved_vector = row["embedding"].to_numpy()

                np.testing.assert_allclose(
                    saved_vector,
                    np.array([1, 2, 3], dtype=np.float32),
                )

        finally:
            # 6. Always close the pool.
            await pool.close()

    # 7. Use a Psycopg-compatible event loop on Windows.
    if sys.platform == "win32":
        asyncio.run(
            run_test(),
            loop_factory=lambda: asyncio.SelectorEventLoop(
                selectors.SelectSelector()
            ),
        )
    else:
        asyncio.run(run_test())
