"""Read-only routes the dashboard loads its timeline, Q&A and alerts from (design §6.7, §8.1)."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated, LiteralString, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AwareDatetime

from server.assistant.understand import load_lexicon
from server.common.db import HYBRID_SEARCH_SQL
from server.common.schemas import Alert, FrameDetail, FrameSummary, QuestionRecord
from server.common.text_norm import build_search_text, normalize_text
from server.ingest.store import TEXT_PREVIEW_LENGTH

router = APIRouter(prefix="/v1")

# Same labels and text preview as the frame_stored event, so a reload shows what was live.
FRAMES_SQL = """
    SELECT f.id AS frame_id, f.captured_at, '/media/' || f.thumb_path AS thumb_url,
           ARRAY(SELECT d.label FROM detections AS d WHERE d.frame_id = f.id
                 ORDER BY d.id) AS labels,
           left(COALESCE((SELECT string_agg(o.text, ' ' ORDER BY o.id) FROM ocr_texts AS o
                          WHERE o.frame_id = f.id), ''), %(preview)s) AS text_preview,
           f.keep_reason
    FROM frames AS f
    WHERE f.session_id = %(session_id)s
      AND (%(from)s::timestamptz IS NULL OR f.captured_at >= %(from)s)
      AND (%(to)s::timestamptz IS NULL OR f.captured_at <= %(to)s)
      AND (%(label)s::text IS NULL OR EXISTS (
          SELECT 1 FROM detections AS d WHERE d.frame_id = f.id AND d.label = %(label)s))
      -- %% is pg_trgm's "similar" operator; its 0.2 threshold is set on every pool connection.
      AND (%(text)s::text IS NULL OR EXISTS (
          SELECT 1 FROM ocr_texts AS o WHERE o.frame_id = f.id AND o.text_norm %% %(text)s))
      AND (%(ids)s::bigint[] IS NULL OR f.id = ANY(%(ids)s))
    -- ids come best first from the search; without one, every position is NULL and time decides.
    ORDER BY array_position(%(ids)s::bigint[], f.id), f.captured_at DESC, f.id DESC
    LIMIT %(limit)s
"""


def named_labels(q: str) -> list[str]:
    """Object labels the search names in English or Arabic, e.g. "staircase" → stairs."""
    text = normalize_text(q)
    return list(
        dict.fromkeys(
            label
            for lang in ("en", "ar")
            for label, pattern in load_lexicon(lang)["synonyms"]
            if pattern.search(text)
        )
    )


@router.get("/frames")
async def list_frames(
    request: Request,
    session_id: UUID,
    from_: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    to: AwareDatetime | None = None,
    label: str | None = None,
    text: str | None = None,
    q: str | None = None,
    limit: Annotated[int, Query(ge=1)] = 100,
) -> list[FrameSummary]:
    app = request.app
    scores: dict[int, float] | None = None
    async with app.state.pool.connection() as conn:
        if q:
            # "Search by meaning": the assistant's hybrid search, scored by the sum it sorts by.
            clip_vector = await asyncio.to_thread(app.state.clip_text.encode, q)
            now = datetime.now(UTC)
            cursor = await conn.execute(
                cast(LiteralString, HYBRID_SEARCH_SQL),
                {
                    "session_id": session_id,
                    # Without a range, the whole session; retention already drops old frames.
                    "time_from": from_ or datetime.min.replace(tzinfo=UTC),
                    "time_to": to or now,
                    "clip_vector": clip_vector,
                    "text": build_search_text(q),
                    "target_labels": named_labels(q),
                    "asked_at": now,
                },
            )
            # shortcut: the hybrid SQL keeps its top 30 only; raise LIMIT in queries.sql if needed.
            scores = {
                row["id"]: round(float(row["s_clip"] + row["s_text"] + row["s_obj"]), 3)
                for row in await cursor.fetchall()
            }
        cursor = await conn.execute(
            FRAMES_SQL,
            {
                "session_id": session_id,
                "from": from_,
                "to": to,
                "label": label or None,
                "text": normalize_text(text) if text else None,
                "ids": None if scores is None else list(scores),
                "preview": TEXT_PREVIEW_LENGTH,
                "limit": limit,
            },
        )
        rows = await cursor.fetchall()
    return [
        FrameSummary(**row, score=None if scores is None else scores[row["frame_id"]])
        for row in rows
    ]


@router.get("/frames/{frame_id}")
async def get_frame(frame_id: int, request: Request) -> FrameDetail:
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id AS frame_id, session_id, captured_at, '/media/' || image_path AS image_url,
                   '/media/' || thumb_path AS thumb_url, '/media/' || original_path AS original_url,
                   width, height, sharpness, brightness, contrast, edge_density, enhanced,
                   enhance_ops, keep_reason, faces_blurred, scene_text, timings_ms
            FROM frames WHERE id = %s
            """,
            (frame_id,),
        )
        frame = await cursor.fetchone()
        if frame is None:
            raise HTTPException(status_code=404)
        cursor = await conn.execute(
            """
            SELECT label, confidence, bbox, area_ratio, position
            FROM detections WHERE frame_id = %s ORDER BY id
            """,
            (frame_id,),
        )
        detections = await cursor.fetchall()
        cursor = await conn.execute(
            """
            SELECT text, confidence, bbox, from_crop
            FROM ocr_texts WHERE frame_id = %s ORDER BY id
            """,
            (frame_id,),
        )
        ocr = await cursor.fetchall()
    return FrameDetail(**frame, detections=detections, ocr=ocr)


@router.get("/questions")
async def list_questions(
    request: Request, session_id: UUID, limit: Annotated[int, Query(ge=1)] = 50
) -> list[QuestionRecord]:
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT q.id AS question_id, q.session_id, q.asked_at, q.input_mode, q.stt_engine,
                   q.transcript, q.lang, q.intent, q.t_from, q.t_to, q.answer, q.confidence,
                   q.abstained, q.safety_rewrite, q.model, q.timings_ms,
                   ARRAY(SELECT json_build_object('frame_id', e.frame_id, 'rank', e.rank,
                                                  'score', e.score, 'parts', e.parts,
                                                  'thumb_url', '/media/' || f.thumb_path)
                         FROM question_evidence AS e JOIN frames AS f ON f.id = e.frame_id
                         WHERE e.question_id = q.id ORDER BY e.rank) AS evidence
            FROM questions AS q
            WHERE q.session_id = %s
            ORDER BY q.asked_at DESC, q.id DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
    return [QuestionRecord(**row) for row in rows]


@router.get("/alerts")
async def list_alerts(
    request: Request, session_id: UUID, limit: Annotated[int, Query(ge=1)] = 50
) -> list[Alert]:
    async with request.app.state.pool.connection() as conn:
        # LEFT JOIN: an alert's frame may be gone (retention), and then thumb_url is null.
        cursor = await conn.execute(
            """
            SELECT a.id AS alert_id, a.kind, a.message, a.lang, a.frame_id,
                   '/media/' || f.thumb_path AS thumb_url, a.created_at
            FROM alerts AS a LEFT JOIN frames AS f ON f.id = a.frame_id
            WHERE a.session_id = %s
            ORDER BY a.created_at DESC, a.id DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
    return [Alert(**row) for row in rows]
