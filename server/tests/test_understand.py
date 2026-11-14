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


# VA-69: every planned demo question (§16) plus natural English and Arabic (standard and
# Lebanese) phrasings, with past/present pairs. The window is (newest, oldest) seconds before
# the question, or None when the question doesn't search stored frames.
INTENT_SET = [
    # --- English ---
    ("en", "What's in front of me?", "LIVE_SCENE", None, []),  # demo step 2
    ("en", "What was in front of me?", "MEMORY", (0, 180), []),
    ("en", "Is there a chair near me?", "LIVE_SCENE", None, ["chair"]),
    ("en", "Was there a chair back there?", "MEMORY", (0, 180), ["chair"]),
    ("en", "Describe what's around me.", "LIVE_SCENE", None, []),
    ("en", "Is it safe to cross?", "LIVE_SCENE", None, []),  # demo step 6
    ("en", "Read this sign for me.", "READ_TEXT_NOW", None, ["sign"]),
    ("en", "What does this say?", "READ_TEXT_NOW", None, []),
    ("en", "What did that sign say?", "READ_TEXT_PAST", (0, 180), ["sign"]),
    # demo step 4: "written" makes it a reading question, so it searches frames with OCR text
    ("en", "What was written on the door I just passed?", "READ_TEXT_PAST", (3, 90), ["door"]),
    (
        "en",
        "What did the room sign say a minute ago?",
        "READ_TEXT_PAST",
        (30, 150),
        ["sign", "room number sign"],
    ),
    ("en", "Did you see a bottle?", "FIND_OBJECT", (0, 1800), ["bottle"]),  # demo step 5
    # demo step 6: not a YOLOE class, so no label; the evidence gate (VA-71) says "I'm not sure"
    ("en", "Did you see a fire extinguisher?", "FIND_OBJECT", (0, 1800), []),
    ("en", "Where did I leave my keys?", "FIND_OBJECT", (0, 1800), ["keys"]),
    ("en", "Where was my backpack 3 minutes ago?", "FIND_OBJECT", (120, 240), ["backpack"]),
    ("en", "What did we pass just now?", "MEMORY", (3, 90), []),
    ("en", "What did I walk past earlier?", "MEMORY", (0, 1800), []),
    ("en", "Who is this?", "IDENTITY", None, []),  # demo step 6
    ("en", "Who's standing next to me?", "IDENTITY", None, []),
    ("en", "How old is the man in front of me?", "IDENTITY", None, ["person"]),
    ("en", "What's the weather like today?", "OUT_OF_SCOPE", None, []),
    ("en", "Tell me a joke.", "OUT_OF_SCOPE", None, []),
    ("en", "What's the capital of Lebanon?", "OUT_OF_SCOPE", None, []),
    ("en", "Hmm, okay.", "MEMORY", (0, 180), []),  # nothing matches: memory + live frame
    pytest.param(
        "en",
        "I just want to know what's ahead.",
        "LIVE_SCENE",
        None,
        [],
        marks=pytest.mark.xfail(
            reason="'just' counts as past tense, which the door-I-just-passed questions need"
        ),
    ),
    # --- Arabic ---
    ("ar", "شو في قدامي؟", "LIVE_SCENE", None, []),
    ("ar", "شو كان قدامي؟", "MEMORY", (0, 180), []),
    ("ar", "ماذا يوجد أمامي؟", "LIVE_SCENE", None, []),
    ("ar", "في شي قدامي؟", "LIVE_SCENE", None, []),
    ("ar", "شو عم تشوف هلق؟", "LIVE_SCENE", None, []),
    ("ar", "في كرسي حدي؟", "LIVE_SCENE", None, ["chair"]),
    ("ar", "صفلي شو حولي", "LIVE_SCENE", None, []),
    ("ar", "هل الطريق آمن للعبور؟", "LIVE_SCENE", None, []),
    ("ar", "شو مكتوب هون؟", "READ_TEXT_NOW", None, []),  # demo step 3
    ("ar", "اقرالي هاللافتة", "READ_TEXT_NOW", None, ["sign"]),
    ("ar", "شو كاتب عالباب؟", "READ_TEXT_NOW", None, ["door"]),
    ("ar", "شو كان مكتوب عاللافتة؟", "READ_TEXT_PAST", (0, 180), ["sign"]),
    ("ar", "شو كان مكتوب عالباب يلي مرقنا حدو من شوي؟", "READ_TEXT_PAST", (3, 90), ["door"]),
    ("ar", "وين شفت المفاتيح؟", "FIND_OBJECT", (0, 1800), ["keys"]),
    ("ar", "شفت قنينة مي؟", "FIND_OBJECT", (0, 1800), ["bottle"]),
    ("ar", "وين حطيت جزداني؟", "FIND_OBJECT", (0, 1800), ["wallet"]),
    ("ar", "وين كانت الشنطة قبل خمس دقايق؟", "FIND_OBJECT", (240, 360), ["bag"]),
    ("ar", "شو شفنا قبل شوي؟", "MEMORY", (3, 90), []),
    ("ar", "شو مرقنا قبل دقيقة؟", "MEMORY", (30, 150), []),
    ("ar", "مرقنا على درج قبل؟", "MEMORY", (0, 1800), ["stairs"]),
    ("ar", "مين هيدا؟", "IDENTITY", None, []),
    ("ar", "مين هيدي يلي قدامي؟", "IDENTITY", None, []),
    ("ar", "من هذا الشخص؟", "IDENTITY", None, ["person"]),
    ("ar", "شو اسمه؟", "IDENTITY", None, []),
    ("ar", "شو الطقس اليوم؟", "OUT_OF_SCOPE", None, []),
    ("ar", "قديش الساعة؟", "OUT_OF_SCOPE", None, []),
    ("ar", "مين ربح المباراة؟", "OUT_OF_SCOPE", None, []),
]


@pytest.mark.parametrize(("lang", "question", "intent", "window", "labels"), INTENT_SET)
def test_intent_set(
    lang: str,
    question: str,
    intent: str,
    window: tuple[float, float] | None,
    labels: list[str],
) -> None:
    understanding = ask(question, lang)
    found_window = (
        None
        if understanding.t_from is None
        else (seconds_ago(understanding.t_to), seconds_ago(understanding.t_from))
    )

    assert (understanding.intent, found_window, understanding.target_labels) == (
        intent,
        window,
        labels,
    )
