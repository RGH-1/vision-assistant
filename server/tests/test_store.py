from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from psycopg.errors import CheckViolation

from server.common.db import open_ingest_connection
from server.ingest.store import FrameStore, ProcessedFrame, StoredDetection, StoredOcr


class FakeHub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


def test_store_writes_variants_rows_and_dashboard_event(tmp_path: Path) -> None:
    connection = open_ingest_connection()
    session_id = None
    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()["id"]
        connection.commit()
        hub = FakeHub()
        store = FrameStore(tmp_path, connection, hub)
        image = np.full((900, 1600, 3), 128, dtype=np.uint8)
        frame_id = store.store(
            ProcessedFrame(
                session_id=session_id,
                captured_at=datetime.now(UTC),
                image=image,
                original=image,
                enhanced=True,
                enhance_ops=("gamma", "clahe"),
                sharpness=42.0,
                brightness=128.0,
                contrast=8.0,
                edge_density=0.2,
                dhash=1,
                keep_reason="heartbeat",
                clip_emb=np.eye(1, 512, dtype=np.float32)[0],
                timings_ms={"clip": 3.5},
                detections=(StoredDetection("Door", 0.9, (1, 2, 3, 4), 0.1, "center"),),
                ocr=(StoredOcr("غرفة ٢٠٤", 0.9, (5, 6, 7, 8), True),),
            )
        )

        row = connection.execute("SELECT * FROM frames WHERE id = %s", (frame_id,)).fetchone()
        detections = connection.execute(
            "SELECT * FROM detections WHERE frame_id = %s", (frame_id,)
        ).fetchall()
        ocr = connection.execute(
            "SELECT * FROM ocr_texts WHERE frame_id = %s", (frame_id,)
        ).fetchall()
        assert row["scene_text"] == "door center | غرفه 204"
        assert row["timings_ms"] == {"clip": 3.5}
        assert len(detections) == len(ocr) == 1
        assert ocr[0]["text_norm"] == "غرفه 204"
        assert cv2.imread(str(tmp_path / row["image_path"])).shape[:2] == (720, 1280)
        assert cv2.imread(str(tmp_path / row["thumb_path"])).shape[:2] == (180, 320)
        assert (tmp_path / row["original_path"]).is_file()
        assert hub.events == [
            {
                "type": "frame_stored",
                "frame_id": frame_id,
                "captured_at": row["captured_at"],
                "thumb_url": f"/media/{row['thumb_path']}",
                "labels": ["Door"],
                "text_preview": "غرفة ٢٠٤",
                "keep_reason": "heartbeat",
            }
        ]
    finally:
        if session_id is not None:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()
        connection.close()


def test_store_removes_files_when_the_database_transaction_fails(tmp_path: Path) -> None:
    connection = open_ingest_connection()
    session_id = None
    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'test-only')
            RETURNING id
            """
        ).fetchone()["id"]
        connection.commit()
        store = FrameStore(tmp_path, connection)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(CheckViolation):
            store.store(
                ProcessedFrame(
                    session_id=session_id,
                    captured_at=datetime.now(UTC),
                    image=image,
                    sharpness=1.0,
                    brightness=1.0,
                    contrast=1.0,
                    edge_density=1.0,
                    dhash=1,
                    keep_reason="heartbeat",
                    clip_emb=np.eye(1, 512, dtype=np.float32)[0],
                    timings_ms={},
                    detections=(StoredDetection("door", 0.9, (1, 2, 3, 4), 0.1, "not-a-position"),),
                )
            )

        assert not list(tmp_path.rglob("*.jpg"))
        assert (
            connection.execute(
                "SELECT count(*) AS n FROM frames WHERE session_id = %s", (session_id,)
            ).fetchone()["n"]
            == 0
        )
    finally:
        if session_id is not None:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()
        connection.close()
