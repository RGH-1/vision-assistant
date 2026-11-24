"""VA-68 checks: the keyword lists the intent engine matches questions against (design §6.5.2)."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from server.common.text_norm import normalize_text
from server.ingest.detection import LABEL_ALIASES, load_object_detection_config

LEXICON_DIR = Path(__file__).parents[1] / "assistant" / "lexicon"
LANGS = ["en", "ar"]  # French is optional (VA-78)
INTENT_LISTS = ["identity", "find", "read", "live", "past", "out_of_scope"]


def load(lang: str) -> dict[str, Any]:
    return yaml.safe_load((LEXICON_DIR / f"{lang}.yaml").read_text(encoding="utf-8"))


def first_time_match(lexicon: dict[str, Any], question: str) -> tuple[str, re.Match[str]] | None:
    text = normalize_text(question)
    for name, entry in lexicon["time"].items():
        if match := re.search(entry["pattern"], text):
            return name, match
    return None


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("section", INTENT_LISTS)
def test_every_list_has_five_different_phrases(lang: str, section: str) -> None:
    phrases = [normalize_text(phrase) for phrase in load(lang)[section]]

    assert len(phrases) >= 5
    assert all(phrases)
    assert len(set(phrases)) == len(phrases)  # أمامي and امامي would be the same phrase


@pytest.mark.parametrize("lang", LANGS)
def test_synonyms_cover_every_stored_yoloe_label_once(lang: str) -> None:
    classes = load_object_detection_config().classes
    stored_labels = {LABEL_ALIASES.get(name, name) for name in classes}
    synonyms = load(lang)["synonyms"]
    words = [normalize_text(word) for words in synonyms.values() for word in words]

    assert set(synonyms) == stored_labels
    assert all(synonyms.values())
    assert len(set(words)) == len(words)  # a word that meant two labels would be ambiguous


@pytest.mark.parametrize("lang", LANGS)
def test_patterns_and_numbers_are_already_normalized(lang: str) -> None:
    lexicon = load(lang)

    for entry in lexicon["time"].values():
        assert normalize_text(entry["pattern"]) == entry["pattern"]
        re.compile(entry["pattern"])
    numbers = [normalize_text(word) for word in lexicon["numbers"]]
    assert len(set(numbers)) == len(numbers)


@pytest.mark.parametrize(
    ("lang", "question", "expected"),
    [
        ("en", "What was on the door I just passed?", "just"),
        ("en", "What did the sign say a minute ago?", "minute"),
        ("en", "Where were my keys 5 minutes ago?", "n_min"),
        ("en", "Did you see a bus two minutes ago?", "n_min"),
        ("en", "Did we pass a door earlier?", "earlier"),
        ("en", "What's in front of me?", None),
        ("ar", "شو شفنا قبل شوي؟", "just"),
        ("ar", "شو شفنا ابل شوي؟", "just"),
        ("ar", "شو كان مكتوب قبل دقيقة؟", "minute"),
        ("ar", "وين كانت المفاتيح قبل دقيقتين؟", "two_min"),
        ("ar", "شو شفنا قبل ٥ دقائق؟", "n_min"),
        ("ar", "شو شفنا قبل خمس دقايق؟", "n_min"),
        ("ar", "مرقنا باب قبل؟", "earlier"),
        ("ar", "شو في قدامي؟", None),
    ],
)
def test_time_expressions(lang: str, question: str, expected: str | None) -> None:
    found = first_time_match(load(lang), question)

    assert (found[0] if found else None) == expected


@pytest.mark.parametrize(
    ("lang", "question", "minutes"),
    [
        ("en", "Where were my keys 5 minutes ago?", 5),
        ("en", "Did you see a bus two minutes ago?", 2),
        ("ar", "شو شفنا قبل ٥ دقائق؟", 5),
        ("ar", "شو شفنا قبل خمس دقايق؟", 5),
    ],
)
def test_minutes_are_digits_or_listed_number_words(lang: str, question: str, minutes: int) -> None:
    lexicon = load(lang)
    numbers = {normalize_text(word): value for word, value in lexicon["numbers"].items()}
    _, match = first_time_match(lexicon, question)

    word = match.group(1)
    assert (int(word) if word.isdigit() else numbers[word]) == minutes


@pytest.mark.parametrize(
    "transcript",
    # What Whisper wrote for "شو في قدامي؟" in VA-64 (Groq) and VA-66 (local medium).
    ["شو في الدمي", "شو في ادامة", "شوفيه ادامة"],
)
def test_measured_lebanese_transcripts_contain_a_live_phrase(transcript: str) -> None:
    phrases = [normalize_text(phrase) for phrase in load("ar")["live"]]

    assert any(phrase in normalize_text(transcript) for phrase in phrases)
