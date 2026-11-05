

import os
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool




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
        configure=register_vector_async,  #let the adapter be registered on every connection it creates
    )

    return pool


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
        raise ValueError(
            "clip_emb must contain exactly 512 numbers"
        )

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
    offset: int = 0, #Specifies how many matching frames to skip
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
