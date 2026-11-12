"""VA-67 checks: the rules of the intent engine (design §6.5.2). VA-69 adds the 40-sentence set."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from server.assistant.understand import Understanding, understand

ASKED_AT = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
SETTINGS = SimpleNamespace(retention_minutes=30, default_window_s=180)


def ask(question: str, lang: str = "en") -> Understanding:
    return understand(question, lang, ASKED_AT, SETTINGS)


def seconds_ago(moment: datetime | None) -> float | None:
    return None if moment is None else (ASKED_AT - moment).total_seconds()


@pytest.mark.parametrize(
    ("lang", "question", "intent"),
    [
        ("en", "What's in front of me?", "LIVE_SCENE"),
        ("en", "Read this", "READ_TEXT_NOW"),
        ("en", "What did that sign say?", "READ_TEXT_PAST"),
        ("en", "What was in front of me?", "MEMORY"),
        ("en", "Did you see my keys?", "FIND_OBJECT"),
        ("en", "Who is this?", "IDENTITY"),
        ("en", "What's the capital of France?", "OUT_OF_SCOPE"),
        ("ar", "شو في قدامي؟", "LIVE_SCENE"),
        ("ar", "شو مكتوب هون؟", "READ_TEXT_NOW"),
        ("ar", "شو كان مكتوب عاللافتة؟", "READ_TEXT_PAST"),
        ("ar", "شو شفنا قبل شوي؟", "MEMORY"),
        ("ar", "وين شفت المفاتيح؟", "FIND_OBJECT"),
        ("ar", "مين هيدا؟", "IDENTITY"),
    ],
)
def test_the_design_examples_get_their_intent(lang: str, question: str, intent: str) -> None:
    assert ask(question, lang).intent == intent


def test_identity_wins_over_everything_else() -> None:
    assert ask("Did you see who is at the door?").intent == "IDENTITY"


@pytest.mark.parametrize(
    "transcript",
    # What Whisper wrote for "شو في قدامي؟" in VA-64 (Groq) and VA-66 (local medium).
    ["شو في الدمي", "شو في ادامة", "شوفيه ادامة"],
)
def test_measured_lebanese_transcripts_are_live_questions(transcript: str) -> None:
    assert ask(transcript, "ar").intent == "LIVE_SCENE"


def test_curly_apostrophes_match_like_straight_ones() -> None:
    assert ask("Who’s that?").intent == "IDENTITY"


def test_whole_words_only() -> None:
    # "read" is inside "already" and "ready": neither makes this a reading question.
    assert ask("Am I ready, already?").intent == "MEMORY"


@pytest.mark.parametrize(
    ("lang", "question", "labels"),
    [
        ("en", "Did you see my keys or my wallet?", ["keys", "wallet"]),
        ("en", "Where is the exit sign?", ["sign", "exit sign"]),  # an exit sign is a sign too
        ("ar", "وين المفاتيح؟", ["keys"]),  # ال + مفاتيح
        ("ar", "شو كان مكتوب عالباب؟", ["door"]),  # ع + ال + باب
        ("ar", "وين مفاتيحي؟", ["keys"]),
        ("en", "What's in front of me?", []),
    ],
)
def test_target_labels(lang: str, question: str, labels: list[str]) -> None:
    assert ask(question, lang).target_labels == labels


@pytest.mark.parametrize(
    ("lang", "question", "newest_s", "oldest_s"),
    [
        ("en", "What was on the door I just passed?", 3, 90),
        ("en", "What did the sign say a minute ago?", 30, 150),
        ("en", "Where were my keys 5 minutes ago?", 240, 360),
        ("en", "Did you see a bus two minutes ago?", 60, 180),
        ("en", "Did we pass a door earlier?", 0, 1800),
        ("en", "Was there a door a few minutes ago?", 0, 180),  # "few" isn't a number
        ("ar", "شو شفنا قبل شوي؟", 3, 90),
        ("ar", "وين كانت المفاتيح قبل دقيقتين؟", 60, 180),
        ("ar", "شو شفنا قبل خمس دقايق؟", 240, 360),
        # no time words: memory and past text look 3 minutes back, find looks everywhere
        ("en", "What did that sign say?", 0, 180),
        ("en", "Did you see my keys?", 0, 1800),
    ],
)
def test_time_windows_count_back_from_the_question(
    lang: str, question: str, newest_s: float, oldest_s: float
) -> None:
    understanding = ask(question, lang)

    assert seconds_ago(understanding.t_to) == newest_s
    assert seconds_ago(understanding.t_from) == oldest_s


@pytest.mark.parametrize("question", ["What's in front of me?", "Read this", "Who is this?"])
def test_questions_that_dont_search_memory_have_no_window(question: str) -> None:
    understanding = ask(question)

    assert (understanding.t_from, understanding.t_to) == (None, None)


def test_a_question_nothing_matches_searches_memory_and_the_live_frame() -> None:
    understanding = ask("Hmm, the hallway")

    assert understanding.intent == "MEMORY"
    assert understanding.use_live_frame
    assert seconds_ago(understanding.t_from) == 180
    assert not ask("What was in front of me?").use_live_frame


def test_search_text_drops_stop_words() -> None:
    assert ask("What did the sign say?").search_text == "sign say"


def test_a_language_without_a_lexicon_is_an_error() -> None:
    with pytest.raises(FileNotFoundError):
        ask("Qui est-ce ?", "fr")
