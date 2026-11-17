"""VA-75 checks: the phone questions get their intents, and the results table marks failures."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from eval.check_arabic import QUESTIONS, table
from server.assistant.understand import understand

SETTINGS = SimpleNamespace(retention_minutes=30, default_window_s=180)


def test_the_questions_cover_all_seven_intents() -> None:
    assert len(QUESTIONS) == 10
    assert len({intent for _, intent in QUESTIONS}) == 7


@pytest.mark.parametrize(("said", "intent"), QUESTIONS)
def test_each_question_as_said_gets_its_intent(said: str, intent: str) -> None:
    # The phone run checks what Whisper writes; this checks the rules on the words as said.
    assert understand(said, "ar", datetime.now(UTC), SETTINGS).intent == intent


def test_the_table_marks_a_wrong_intent_and_an_answer_not_in_arabic() -> None:
    asked = [
        {"transcript": "شو في قدامي", "intent": "LIVE_SCENE", "answer": "في باب أمامك."},
        {"transcript": "شو مكتوب هون", "intent": "MEMORY", "answer": "I see a | sign."},
    ]

    first, second = table(asked).splitlines()[2:]

    assert first == "| 1 | شو في قدامي؟ | شو في قدامي | ✅ LIVE_SCENE | في باب أمامك. | ✅ |  |  |"
    assert second == (
        "| 2 | شو مكتوب هون؟ | شو مكتوب هون | ❌ MEMORY (expected READ_TEXT_NOW) "
        "| I see a / sign. | ❌ |  |  |"
    )
