"""Capture the §7.4 query plan at the 30-minute retention-window scale."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from server.common.db import HYBRID_SEARCH_SQL, open_ingest_connection

FRAME_COUNT = 1800  # 30 minutes × one saved frame per second
OUTPUT_PATH = Path(__file__).with_name("VA56_EXPLAIN.md")


def main() -> None:
    """Seed an isolated retention-sized session, save EXPLAIN ANALYZE, then remove it."""
    connection = open_ingest_connection()
    session_id = None
    try:
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'va56-explain')
            RETURNING id
            """
        ).fetchone()["id"]
        asked_at = datetime.now(UTC).replace(microsecond=0)
        start = asked_at - timedelta(minutes=30)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1
        connection.execute(
            """
            INSERT INTO frames (
                session_id, captured_at, image_path, thumb_path, width, height, sharpness,
                brightness, contrast, edge_density, dhash, keep_reason, clip_emb, timings_ms
            )
            SELECT
                %s, %s + sample * interval '1 second',
                'frames/va56-' || sample || '.jpg', 'thumbs/va56-' || sample || '.jpg',
                1280, 720, 100, 120, 40, 0.1, sample, 'heartbeat', %s, '{}'
            FROM generate_series(0, %s) AS sample
            """,
            (session_id, start, vector, FRAME_COUNT - 1),
        )
        connection.commit()
        plan = connection.execute(
            f"EXPLAIN (ANALYZE, BUFFERS) {HYBRID_SEARCH_SQL}",
            {
                "session_id": session_id,
                "time_from": start,
                "time_to": asked_at,
                "clip_vector": vector,
                "text": "",
                "target_labels": [],
                "asked_at": asked_at,
            },
        ).fetchall()
        plan_text = "\n".join(row["QUERY PLAN"] for row in plan)
        plan_text = re.sub(r"'\[[^]]+\]'::vector", "'<query vector>'::vector", plan_text)
        OUTPUT_PATH.write_text(
            "# VA-56 hybrid retrieval: EXPLAIN ANALYZE\n\n"
            f"Generated with {FRAME_COUNT} frames in one 30-minute session.\n\n"
            "The planner uses the `(session_id, captured_at)` B-tree index, then sorts the "
            "small filtered window; it does not use the global HNSW index.\n\n"
            f"```text\n{plan_text}\n```\n",
            encoding="utf-8",
        )
        print(f"Saved {OUTPUT_PATH}")
    finally:
        if session_id is not None:
            connection.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            connection.commit()
        connection.close()


if __name__ == "__main__":
    main()
