"""Query understanding: which of the 7 intents a question is, and where to look (design §6.5.2)."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from server.common.schemas import Intent, Lang
from server.common.text_norm import build_search_text, normalize_text
from server.settings import Settings

LEXICON_DIR = Path(__file__).parent / "lexicon"
PHRASE_LISTS = ["identity", "find", "read", "live", "past", "out_of_scope"]
# Arabic glues و ف ب ل ك ع and ال onto the next word: والباب, عاللافتة, للباب.
ARABIC_PREFIX = "(?:[وفبلكع])?(?:ال|ل)?"
# These intents search stored frames, so they get a time window.
SEARCHES_MEMORY = ("READ_TEXT_PAST", "MEMORY", "FIND_OBJECT")


@dataclass(frozen=True)
class Understanding:
    intent: Intent
    lang: Lang
    t_from: datetime | None  # None for questions that don't search stored frames
    t_to: datetime | None
    target_labels: list[str]  # YOLOE labels the question names, e.g. ["keys"]
    search_text: str
    use_live_frame: bool  # nothing matched: search memory and add the newest live frame


def phrase_pattern(phrase: str, lang: Lang) -> re.Pattern[str]:
    # Whole words only, so "read" doesn't match "already".
    prefix = ARABIC_PREFIX if lang == "ar" else ""
    return re.compile(rf"(?<!\w){prefix}{re.escape(normalize_text(phrase))}(?!\w)")


@cache
def load_lexicon(lang: Lang) -> dict[str, Any]:
    """Raises FileNotFoundError for a language without a lexicon (French until VA-78)."""
    raw = yaml.safe_load((LEXICON_DIR / f"{lang}.yaml").read_text(encoding="utf-8"))
    lexicon: dict[str, Any] = {
        section: [phrase_pattern(phrase, lang) for phrase in raw[section]]
        for section in PHRASE_LISTS
    }
    lexicon["time"] = [
        (name, re.compile(entry["pattern"]), entry.get("window_s"))
        for name, entry in raw["time"].items()
    ]
    lexicon["numbers"] = {normalize_text(word): value for word, value in raw["numbers"].items()}
    lexicon["synonyms"] = [
        (label, phrase_pattern(word, lang))
        for label, words in raw["synonyms"].items()
        for word in words
    ]
    return lexicon


def seconds_ago_window(
    text: str, lexicon: dict[str, Any], retention_minutes: int
) -> tuple[float, float] | None:
    """The first time expression found, as (newest, oldest) seconds before the question."""
    for name, pattern, window_s in lexicon["time"]:
        match = pattern.search(text)
        if not match:
            continue
        if window_s:
            return window_s[0], window_s[1]
        if name == "n_min":
            word = match.group(1)
            minutes = int(word) if word.isdigit() else lexicon["numbers"].get(word)
            if minutes is None:
                return None  # "a few minutes ago": the intent's default window
            return max(minutes - 1, 0) * 60, (minutes + 1) * 60
        return 0, retention_minutes * 60  # "earlier": the whole retention window
    return None


def understand(question: str, lang: Lang, asked_at: datetime, settings: Settings) -> Understanding:
    """Rules in the order of §6.5.2. Raises FileNotFoundError when lang has no lexicon."""
    lexicon = load_lexicon(lang)
    # Whisper sometimes writes "what’s" with a curly apostrophe; the lexicon uses straight ones.
    text = normalize_text(question).replace("\u2019", "'")

    def says(section: str) -> bool:
        return any(pattern.search(text) for pattern in lexicon[section])

    past = says("past")
    intent: Intent
    if says("identity"):  # first, so the refusal always wins
        intent = "IDENTITY"
    elif says("find"):
        intent = "FIND_OBJECT"
    elif says("read"):
        intent = "READ_TEXT_PAST" if past else "READ_TEXT_NOW"
    elif says("live"):
        intent = "MEMORY" if past else "LIVE_SCENE"
    elif says("out_of_scope"):
        intent = "OUT_OF_SCOPE"
    else:
        intent = "MEMORY"  # past markers alone, or nothing matched (rule 7)

    t_from = t_to = None
    if intent in SEARCHES_MEMORY:
        window = seconds_ago_window(text, lexicon, settings.retention_minutes)
        if window is None:
            default_s = (
                settings.retention_minutes * 60
                if intent == "FIND_OBJECT"
                else settings.default_window_s
            )
            window = (0, default_s)
        t_to = asked_at - timedelta(seconds=window[0])
        t_from = asked_at - timedelta(seconds=window[1])

    return Understanding(
        intent=intent,
        lang=lang,
        t_from=t_from,
        t_to=t_to,
        target_labels=list(
            dict.fromkeys(label for label, pattern in lexicon["synonyms"] if pattern.search(text))
        ),
        search_text=build_search_text(question, lang),
        use_live_frame=intent == "MEMORY" and not past,
    )
