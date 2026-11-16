"""VA-73 checks: answers never say "safe", and stay short (design §6.5.5, Appendix C)."""

import pytest

from server.assistant import gate, gemini, safety

UNSAFE = [
    ("en", "The hallway is safe."),
    ("en", "You can walk safely to the door."),
    ("en", "It is clear to cross now."),
    ("en", "All clear ahead."),
    ("en", "You can go straight."),
    ("en", "There are no obstacles in front of you."),
    ("en", "Nothing in your way."),
    ("en", "The path looks clear."),
    ("ar", "الطريق آمن."),
    ("ar", "المكان آمِن للمشي."),  # with a short vowel
    ("ar", "يمكنك المشي بأمان."),
    ("ar", "لا توجد أي عوائق أمامك."),
    ("ar", "يمكنك العبور الآن."),
    ("fr", "La route est sûre."),
    ("fr", "Vous pouvez traverser."),
]
NORMAL = [
    ("en", "There is a door ahead on your left."),
    ("en", "The sign says EXIT."),
    ("en", "Your keys are on the table, about two meters ahead."),
    ("en", "There is a 'Safety first' poster on the wall."),  # "safety" isn't "safe"
    ("en", "The text on the sign is clear: Room 204."),
    ("en", "I see stairs going down to the right."),
    ("en", "A person is standing near the window."),
    ("ar", "هذا الطابق الثامن."),  # contains "امن" inside a word
    ("ar", "المفاتيح على الطاولة أمامك."),
    ("ar", "اللافتة مكتوب عليها: مخرج."),
    ("ar", "يوجد باب على يسارك."),
    ("ar", "ممنوع التدخين."),
    ("fr", "Je ne suis pas sûr."),  # "not sure", not "not safe"
    ("fr", "Bien sûr, la porte est à gauche."),
    ("fr", "Les clés sont sur la table."),
]


@pytest.mark.parametrize(("lang", "sentence"), UNSAFE)
def test_a_sentence_that_says_safe_becomes_the_cane_message(lang: str, sentence: str) -> None:
    assert safety.rewrite(sentence, lang) == (safety.CANE[lang], True)


@pytest.mark.parametrize(("lang", "sentence"), NORMAL)
def test_a_normal_sentence_is_left_alone(lang: str, sentence: str) -> None:
    assert safety.rewrite(sentence, lang) == (sentence, False)


def test_our_own_fixed_replies_pass_the_filter() -> None:
    replies = [
        *(reply for replies in gate.FIXED_REPLIES.values() for reply in replies.items()),
        *gate.NOT_SURE.items(),
        *gemini.NOT_SURE.items(),
    ]
    for lang, reply in replies:
        assert safety.rewrite(reply, lang) == (reply, False)


def test_only_the_unsafe_sentences_change_and_the_cane_message_is_said_once() -> None:
    answer = "There is a door ahead. It looks safe. You can walk safely."

    assert safety.rewrite(answer, "en") == (
        "There is a door ahead. I can't tell if it's safe. Please use your cane.",
        True,
    )


def test_an_arabic_question_mark_ends_a_sentence() -> None:
    answer = "هل ترى الباب؟ الطريق آمن."

    assert safety.rewrite(answer, "ar") == (f"هل ترى الباب؟ {safety.CANE['ar']}", True)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A door. A chair. A table.", "A door. A chair."),  # at most 2 sentences
        ("A door on the left. " + "word " * 30 + "end.", "A door on the left."),  # over 30 words
        ("word " * 40 + "end.", "word " * 40 + "end."),  # the first sentence is always kept
    ],
)
def test_a_long_answer_is_cut_at_a_sentence_end(answer: str, expected: str) -> None:
    assert safety.shorten(answer, max_sentences=2, max_words=30) == expected.strip()
