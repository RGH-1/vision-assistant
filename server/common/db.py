import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from server.common.text_norm import normalize_text

HYBRID_SEARCH_SQL = (Path(__file__).with_name("queries.sql")).read_text(encoding="utf-8")
TRIGRAM_SIMILARITY_THRESHOLD = 0.2


def open_ingest_connection() -> psycopg.Connection:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    conn = psycopg.connect(
        database_url,
        row_factory=dict_row,
        connect_timeout=5,
    )

    try:
        register_vector(conn)
        conn.execute(f"SET pg_trgm.similarity_threshold = {TRIGRAM_SIMILARITY_THRESHOLD}")
        conn.commit()

    except Exception:
        conn.close()
        raise

    return conn


def create_web_pool() -> AsyncConnectionPool:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    pool = AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={
            "row_factory": dict_row,
            "connect_timeout": 5,
        },
        configure=_configure_web_connection,
    )

    return pool


async def _configure_web_connection(connection: Any) -> None:
    """Configure vector adaptation and noisy-OCR matching for every web-pool connection."""
    await register_vector_async(connection)
    await connection.execute(f"SET pg_trgm.similarity_threshold = {TRIGRAM_SIMILARITY_THRESHOLD}")
    await connection.commit()


def hybrid_search_frames(
    conn: psycopg.Connection,
    session_id: UUID,
    *,
    time_from: datetime,
    time_to: datetime,
    clip_vector: np.ndarray,
    text: str,
    target_labels: list[str],
    asked_at: datetime,
) -> list[dict[str, Any]]:
    """Return the top 30 time-bounded multimodal candidates for M3's final ranking."""
    if time_from > time_to:
        raise ValueError("time_from must not be after time_to")
    vector = np.asarray(clip_vector, dtype=np.float32)
    if vector.shape != (512,):
        raise ValueError("clip_vector must contain exactly 512 numbers")
    if not text:
        text = ""
    if any(not label for label in target_labels):
        raise ValueError("target_labels cannot contain empty labels")

    return conn.execute(
        HYBRID_SEARCH_SQL,
        {
            "session_id": session_id,
            "time_from": time_from,
            "time_to": time_to,
            "clip_vector": vector,
            "text": normalize_text(text),
            "target_labels": target_labels,
            "asked_at": asked_at,
        },
    ).fetchall()


def insert_frame(
    conn: psycopg.Connection,
    frame: dict[str, Any],
) -> int:
    # Prepare the 512-dimensional CLIP vector.
    clip_vector = np.asarray(
        frame["clip_emb"],
        dtype=np.float32,
    )

    if clip_vector.shape != (512,):
        raise ValueError("clip_emb must contain exactly 512 numbers")

    # Prepare values for the SQL query.
    params = {
        "session_id": frame["session_id"],
        "captured_at": frame["captured_at"],
        "image_path": frame["image_path"],
        "thumb_path": frame["thumb_path"],
        "original_path": frame.get("original_path"),
        "width": frame["width"],
        "height": frame["height"],
        "sharpness": frame["sharpness"],
        "brightness": frame["brightness"],
        "contrast": frame["contrast"],
        "edge_density": frame["edge_density"],
        "enhanced": frame.get("enhanced", False),
        "enhance_ops": frame.get("enhance_ops", []),
        "dhash": frame["dhash"],
        "keep_reason": frame["keep_reason"],
        "faces_blurred": frame.get("faces_blurred", 0),
        "scene_text": frame.get("scene_text", ""),
        "clip_emb": clip_vector,
        "timings_ms": Jsonb(frame.get("timings_ms", {})),
    }

    # Insert the frame into PostgreSQL.
    query = """
        INSERT INTO frames (
            session_id,
            captured_at,
            image_path,
            thumb_path,
            original_path,
            width,
            height,
            sharpness,
            brightness,
            contrast,
            edge_density,
            enhanced,
            enhance_ops,
            dhash,
            keep_reason,
            faces_blurred,
            scene_text,
            clip_emb,
            timings_ms
        )
        VALUES (
            %(session_id)s,
            %(captured_at)s,
            %(image_path)s,
            %(thumb_path)s,
            %(original_path)s,
            %(width)s,
            %(height)s,
            %(sharpness)s,
            %(brightness)s,
            %(contrast)s,
            %(edge_density)s,
            %(enhanced)s,
            %(enhance_ops)s,
            %(dhash)s,
            %(keep_reason)s,
            %(faces_blurred)s,
            %(scene_text)s,
            %(clip_emb)s,
            %(timings_ms)s
        )
        RETURNING id
    """

    result = conn.execute(query, params).fetchone()

    return result["id"]


def get_frame(
    conn: psycopg.Connection,
    frame_id: int,
) -> dict[str, Any] | None:

    query = """
        SELECT *
        FROM frames
        WHERE id = %s
    """

    result = conn.execute(
        query,
        (frame_id,),
    ).fetchone()

    return result


def list_frames(
    conn: psycopg.Connection,
    session_id: UUID,
    limit: int = 50,
    offset: int = 0,  # Specifies how many matching frames to skip
) -> list[dict[str, Any]]:

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    if offset < 0:
        raise ValueError("offset cannot be negative")

    query = """
        SELECT *
        FROM frames
        WHERE session_id = %s
        ORDER BY captured_at DESC, id DESC
        LIMIT %s
        OFFSET %s
    """

    result = conn.execute(
        query,
        (session_id, limit, offset),
    ).fetchall()

    return result


def insert_question(
    conn: psycopg.Connection,
    question: dict[str, Any],
) -> int:

    # Prepare the question's information.
    params = {
        "session_id": question["session_id"],
        "asked_at": question["asked_at"],
        "input_mode": question["input_mode"],
        "stt_engine": question.get("stt_engine"),
        "transcript": question["transcript"],
        "lang": question["lang"],
        "intent": question["intent"],
        "t_from": question.get("t_from"),
        "t_to": question.get("t_to"),
        "answer": question["answer"],
        "confidence": question["confidence"],
        "abstained": question["abstained"],
        "safety_rewrite": question.get("safety_rewrite", False),
        "model": question.get("model"),
        "timings_ms": Jsonb(question["timings_ms"]),
    }

    # Insert the question into PostgreSQL.
    query = """
        INSERT INTO questions (
            session_id,
            asked_at,
            input_mode,
            stt_engine,
            transcript,
            lang,
            intent,
            t_from,
            t_to,
            answer,
            confidence,
            abstained,
            safety_rewrite,
            model,
            timings_ms
        )
        VALUES (
            %(session_id)s,
            %(asked_at)s,
            %(input_mode)s,
            %(stt_engine)s,
            %(transcript)s,
            %(lang)s,
            %(intent)s,
            %(t_from)s,
            %(t_to)s,
            %(answer)s,
            %(confidence)s,
            %(abstained)s,
            %(safety_rewrite)s,
            %(model)s,
            %(timings_ms)s
        )
        RETURNING id
    """

    result = conn.execute(query, params).fetchone()

    return result["id"]


def insert_alert(
    conn: psycopg.Connection,
    alert: dict[str, Any],
) -> int:

    # Prepare the alert information.
    params = {
        "session_id": alert["session_id"],
        "frame_id": alert.get("frame_id"),
        "kind": alert["kind"],
        "message": alert["message"],
        "lang": alert["lang"],
    }

    # Insert the alert into PostgreSQL.
    query = """
        INSERT INTO alerts (
            session_id,
            frame_id,
            kind,
            message,
            lang
        )
        VALUES (
            %(session_id)s,
            %(frame_id)s,
            %(kind)s,
            %(message)s,
            %(lang)s
        )
        RETURNING id
    """

    result = conn.execute(query, params).fetchone()

    return result["id"]


def list_dashboard_frames(
    conn: psycopg.Connection,
    session_id: UUID,
    *,
    captured_from: datetime | None = None,
    captured_to: datetime | None = None,
    label: str | None = None,
    text: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return one page of timeline frames with optional object and OCR-trigram filters."""
    _validate_page(limit, offset)
    text_query = normalize_text(text) if text else None
    query = """
        SELECT
            f.id AS frame_id,
            f.captured_at,
            f.thumb_path,
            COALESCE(
                array_agg(DISTINCT d.label) FILTER (WHERE d.label IS NOT NULL), '{}'
            ) AS labels,
            COALESCE(
                NULLIF(left(string_agg(DISTINCT o.text, ' '), 160), ''), f.scene_text
            ) AS text_preview,
            f.keep_reason
        FROM frames AS f
        LEFT JOIN detections AS d ON d.frame_id = f.id
        LEFT JOIN ocr_texts AS o ON o.frame_id = f.id
        WHERE f.session_id = %(session_id)s
          AND (
              %(captured_from)s::timestamptz IS NULL
              OR f.captured_at >= %(captured_from)s::timestamptz
          )
          AND (
              %(captured_to)s::timestamptz IS NULL
              OR f.captured_at <= %(captured_to)s::timestamptz
          )
          AND (
              %(label)s::text IS NULL
              OR EXISTS (
                  SELECT 1
                  FROM detections AS label_detection
                  WHERE label_detection.frame_id = f.id
                    AND label_detection.label = %(label)s::text
              )
          )
          AND (
              %(text)s::text IS NULL
              OR EXISTS (
                  SELECT 1 FROM ocr_texts AS search_ocr
                  WHERE search_ocr.frame_id = f.id
                    AND similarity(search_ocr.text_norm, %(text)s::text) >= 0.2
              )
          )
        GROUP BY f.id
        ORDER BY f.captured_at DESC, f.id DESC
        LIMIT %(limit)s OFFSET %(offset)s
    """
    return conn.execute(
        query,
        {
            "session_id": session_id,
            "captured_from": captured_from,
            "captured_to": captured_to,
            "label": label,
            "text": text_query,
            "limit": limit,
            "offset": offset,
        },
    ).fetchall()


def get_dashboard_frame_detail(conn: psycopg.Connection, frame_id: int) -> dict[str, Any] | None:
    """Return one frame plus its detections, OCR lines and stored quality values."""
    frame = conn.execute("SELECT * FROM frames WHERE id = %s", (frame_id,)).fetchone()
    if frame is None:
        return None
    frame["detections"] = conn.execute(
        """
        SELECT label, confidence, bbox, area_ratio, position
        FROM detections WHERE frame_id = %s ORDER BY id
        """,
        (frame_id,),
    ).fetchall()
    frame["ocr"] = conn.execute(
        """
        SELECT text, confidence, bbox, from_crop
        FROM ocr_texts WHERE frame_id = %s ORDER BY id
        """,
        (frame_id,),
    ).fetchall()
    return frame


def list_questions_with_evidence(
    conn: psycopg.Connection, session_id: UUID, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Return newest questions, grouping their saved-frame evidence in Python."""
    _validate_page(limit, 0)
    questions = conn.execute(
        """
        SELECT * FROM questions WHERE session_id = %s
        ORDER BY asked_at DESC, id DESC LIMIT %s
        """,
        (session_id, limit),
    ).fetchall()
    if not questions:
        return []
    question_ids = [question["id"] for question in questions]
    evidence = conn.execute(
        """
        SELECT
            evidence.question_id, evidence.frame_id, evidence.rank, evidence.score,
            evidence.parts, frame.thumb_path
        FROM question_evidence AS evidence
        JOIN frames AS frame ON frame.id = evidence.frame_id
        WHERE evidence.question_id = ANY(%s)
        ORDER BY evidence.question_id, evidence.rank
        """,
        (question_ids,),
    ).fetchall()
    evidence_by_question: dict[int, list[dict[str, Any]]] = {
        question_id: [] for question_id in question_ids
    }
    for item in evidence:
        evidence_by_question[item["question_id"]].append(item)
    for question in questions:
        question["evidence"] = evidence_by_question[question["id"]]
    return questions


def list_dashboard_alerts(
    conn: psycopg.Connection, session_id: UUID, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Return newest alerts with an optional saved-frame thumbnail path."""
    _validate_page(limit, 0)
    return conn.execute(
        """
        SELECT
            alert.id AS alert_id, alert.kind, alert.message, alert.lang, alert.frame_id,
            alert.created_at, frame.thumb_path
        FROM alerts AS alert
        LEFT JOIN frames AS frame ON frame.id = alert.frame_id
        WHERE alert.session_id = %s
        ORDER BY alert.created_at DESC, alert.id DESC
        LIMIT %s
        """,
        (session_id, limit),
    ).fetchall()


def _validate_page(limit: int, offset: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if offset < 0:
        raise ValueError("offset cannot be negative")
