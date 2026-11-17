"""Answers built from stored objects and OCR, for when both Gemini models fail (design §10.3).

They only repeat what ingest saved, so they can't make anything up.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import yaml
from psycopg import AsyncConnection

from server.assistant.gate import NOT_SURE
from server.assistant.gemini import Frame
from server.assistant.understand import LEXICON_DIR
from server.common.schemas import Confidence, Lang

MAX_OBJECTS = 3  # with 2 texts this stays one sentence of about 30 words (answer_max_words)
MAX_TEXTS = 2
PLURAL_LABELS = {"stairs", "keys"}  # English labels that take no "a"
POSITIONS: dict[str, dict[str, str]] = {
    "en": {"left": "on your left", "center": "ahead", "right": "on your right"},
    "ar": {"left": "على يسارك", "center": "أمامك", "right": "على يمينك"},
}
# The first Arabic word the lexicon lists for each label, e.g. stairs → درج.
ARABIC_NAMES: dict[str, str] = {
    label: words[0]
    for label, words in yaml.safe_load((LEXICON_DIR / "ar.yaml").read_text(encoding="utf-8"))[
        "synonyms"
    ].items()
}
# Arabic counted nouns: one, two (dual), 3-10 (plural); 11 and up go back to the singular.
ARABIC_UNITS = {"second": ("ثانية", "ثانيتين", "ثوانٍ"), "minute": ("دقيقة", "دقيقتين", "دقائق")}


def age(seconds_ago: float, lang: Lang) -> str:
    """How long ago, in words: seconds under a minute and a half, then minutes."""
    count, unit = max(1, round(seconds_ago)), "second"
    if count >= 90:
        count, unit = round(count / 60), "minute"
    if lang == "ar":
        one, two, few = ARABIC_UNITS[unit]
        if count <= 2:
            return one if count == 1 else two
        return f"{count} {few if count <= 10 else one}"
    return f"{count} {unit}{'' if count == 1 else 's'}"


def describe(frame: Frame | None, lang: Lang) -> tuple[str, Confidence]:
    """One sentence about the frame's objects (best first) and sign text, e.g.
    "14 seconds ago I saw a door ahead and a sign that says 'EXIT'."
    """
    if frame is None or lang not in POSITIONS:
        # shortcut: no French wording until VA-78, so French gets "I'm not sure".
        return NOT_SURE[lang], "none"
    texts = frame.texts[:MAX_TEXTS]
    objects: dict[str, str] = {}  # label -> position of its best detection
    for detection in frame.detections:
        # With sign text, the text already says "a sign", so the bare "sign" object is left out.
        if not (texts and "sign" in detection["label"]):
            objects.setdefault(detection["label"], detection["position"])
    found = list(objects.items())[:MAX_OBJECTS]
    if not found and not texts:
        return NOT_SURE[lang], "none"

    where = POSITIONS[lang]
    if lang == "ar":
        items = [f"{ARABIC_NAMES.get(label, label)} {where[pos]}" for label, pos in found]
        items += [f"لافتة مكتوب عليها «{text}»" for text in texts]
        return f"قبل {age(frame.seconds_ago, lang)} رأيت: {' و'.join(items)}.", "medium"
    items = [f"{article(label)}{label} {where[pos]}" for label, pos in found]
    items += [f"a sign that says '{text}'" for text in texts]
    listed = items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"
    return f"{age(frame.seconds_ago, lang)} ago I saw {listed}.", "medium"


def article(label: str) -> str:
    if label in PLURAL_LABELS:
        return ""
    return "an " if label[0] in "aeiou" else "a "


async def latest_frame(
    conn: AsyncConnection[dict[str, Any]], session_id: UUID, asked_at: datetime
) -> Frame | None:
    """The newest stored frame, with objects and text best first, as search_memory gives them."""
    cursor = await conn.execute(
        """
        SELECT f.id, EXTRACT(EPOCH FROM (%s - f.captured_at)) AS seconds_ago,
               ARRAY(SELECT json_build_object('label', d.label, 'position', d.position,
                                              'confidence', d.confidence)
                     FROM detections AS d WHERE d.frame_id = f.id
                     ORDER BY d.confidence DESC) AS detections,
               ARRAY(SELECT o.text FROM ocr_texts AS o WHERE o.frame_id = f.id
                     ORDER BY o.confidence DESC) AS texts
        FROM frames AS f
        WHERE f.session_id = %s
        ORDER BY f.captured_at DESC
        LIMIT 1
        """,
        (asked_at, session_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    # No image: a template only reads what ingest found in the frame.
    return Frame(
        b"", float(row["seconds_ago"]), tuple(row["detections"]), tuple(row["texts"]), row["id"]
    )
