"""Memory search: M2's hybrid SQL, then per-intent ranking and duplicate removal (design §6.5.3)."""

import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, LiteralString, cast
from uuid import UUID

import numpy as np
import yaml
from psycopg import AsyncConnection
from pydantic import BaseModel, ConfigDict

from server.assistant.understand import Understanding
from server.common.db import HYBRID_SEARCH_SQL
from server.common.schemas import Intent, QuestionEvidenceParts
from server.ingest.live_buffer import Detection
from server.settings import CONFIG_PATH

DHASH_BITS = 2**64 - 1  # dhash is stored as a signed BIGINT, so mask before counting bits


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recency_s: float
    duplicate_s: float
    duplicate_dhash_max: int
    keep: int
    # One weight per score part, the same five parts the dashboard shows for each frame.
    weights: dict[Intent, QuestionEvidenceParts]


def load_ranking_config(path: Path = CONFIG_PATH) -> RankingConfig:
    return RankingConfig(**yaml.safe_load(path.read_text(encoding="utf-8"))["ranking"])


CONFIG = load_ranking_config()


@dataclass(frozen=True)
class Ranked:
    frame_id: int
    captured_at: datetime
    seconds_ago: float
    dhash: int
    score: float
    parts: QuestionEvidenceParts
    thumb_url: str | None = None
    # Filled in for the few kept frames only, for the answer (§6.5.3 step 5).
    image_path: str | None = None
    detections: tuple[Detection, ...] = ()
    texts: tuple[str, ...] = ()


def scaled(values: list[float]) -> list[float]:
    """Min-max scale to 0-1, because raw CLIP similarities sit in ~0.15-0.35 (§6.5.3)."""
    low, high = min(values), max(values)
    if high == low:
        # Nothing to compare against (one candidate, or all equal), so keep the raw value:
        # already 0-1 for CLIP, text, objects and recency; sharpness is capped at 1.
        return [min(max(value, 0.0), 1.0) for value in values]
    return [(value - low) / (high - low) for value in values]


def is_near_duplicate(frame: Ranked, better: Ranked, config: RankingConfig) -> bool:
    seconds_apart = abs((frame.captured_at - better.captured_at).total_seconds())
    dhash_distance = ((frame.dhash ^ better.dhash) & DHASH_BITS).bit_count()
    return seconds_apart < config.duplicate_s or dhash_distance <= config.duplicate_dhash_max


def rank(
    rows: list[dict[str, Any]], intent: Intent, config: RankingConfig = CONFIG
) -> list[Ranked]:
    """Score M2's candidates best first, drop near-duplicates of better ones, keep the top few."""
    if not rows:
        return []
    # float(): seconds_ago comes back from Postgres as a Decimal.
    columns = {
        "clip": scaled([float(row["s_clip"]) for row in rows]),
        "text": scaled([float(row["s_text"]) for row in rows]),
        "obj": scaled([float(row["s_obj"]) for row in rows]),
        "rec": scaled([math.exp(-float(row["seconds_ago"]) / config.recency_s) for row in rows]),
        "q": scaled([float(row["sharpness"]) for row in rows]),
    }
    weights = config.weights[intent]
    candidates = []
    for i, row in enumerate(rows):
        parts = QuestionEvidenceParts(**{name: values[i] for name, values in columns.items()})
        candidates.append(
            Ranked(
                frame_id=row["id"],
                captured_at=row["captured_at"],
                seconds_ago=float(row["seconds_ago"]),
                dhash=row["dhash"],
                score=sum(getattr(weights, name) * getattr(parts, name) for name in columns),
                parts=parts,
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    kept: list[Ranked] = []
    for candidate in candidates:
        if not any(is_near_duplicate(candidate, better, config) for better in kept):
            kept.append(candidate)
        if len(kept) == config.keep:
            break
    return kept


async def search_memory(
    conn: AsyncConnection[dict[str, Any]],
    session_id: UUID,
    question: Understanding,
    clip_vector: np.ndarray,
    asked_at: datetime,
) -> list[Ranked]:
    """The best few stored frames for a memory question, best first; empty when none match."""
    cursor = await conn.execute(
        cast(LiteralString, HYBRID_SEARCH_SQL),
        {
            "session_id": session_id,
            "time_from": question.t_from,
            "time_to": question.t_to,
            "clip_vector": clip_vector,
            "text": question.search_text,  # already normalized by understand()
            "target_labels": question.target_labels,
            "asked_at": asked_at,
        },
    )
    kept = rank(await cursor.fetchall(), question.intent)
    if not kept:
        return []
    # One more query for the top few: their files, objects and text, which M2's search doesn't
    # return.
    cursor = await conn.execute(
        """
        SELECT f.id, f.thumb_path, f.image_path,
               ARRAY(SELECT json_build_object('label', d.label, 'position', d.position,
                                              'confidence', d.confidence)
                     FROM detections AS d WHERE d.frame_id = f.id
                     ORDER BY d.confidence DESC) AS detections,
               ARRAY(SELECT o.text FROM ocr_texts AS o WHERE o.frame_id = f.id
                     ORDER BY o.confidence DESC) AS texts
        FROM frames AS f
        WHERE f.id = ANY(%s)
        """,
        ([c.frame_id for c in kept],),
    )
    found = {row["id"]: row for row in await cursor.fetchall()}
    # A frame retention deleted in between is skipped.
    return [
        replace(
            c,
            thumb_url=f"/media/{found[c.frame_id]['thumb_path']}",
            image_path=found[c.frame_id]["image_path"],
            detections=tuple(found[c.frame_id]["detections"]),
            texts=tuple(found[c.frame_id]["texts"]),
        )
        for c in kept
        if c.frame_id in found
    ]
