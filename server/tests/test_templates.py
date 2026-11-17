"""VA-74 checks: template answers built from stored objects and OCR (design §10.3)."""

import pytest

from server.assistant import gate, safety
from server.assistant.gemini import Frame
from server.assistant.templates import age, describe


def seen(label: str, position: str = "center", confidence: float = 0.9) -> dict[str, object]:
    return {"label": label, "position": position, "confidence": confidence}


def test_objects_with_positions_and_sign_text_in_english() -> None:
    frame = Frame(b"", 14.2, (seen("door"), seen("stairs", "right")), ("EXIT",))

    assert describe(frame, "en") == (
        "14 seconds ago I saw a door ahead, stairs on your right and a sign that says 'EXIT'.",
        "medium",
    )


def test_the_same_in_arabic_with_arabic_object_names() -> None:
    frame = Frame(b"", 14.2, (seen("door"), seen("stairs", "right")), ("مخرج",))

    assert describe(frame, "ar") == (
        "قبل 14 ثانية رأيت: باب أمامك ودرج على يمينك ولافتة مكتوب عليها «مخرج».",
        "medium",
    )


def test_a_label_the_arabic_lexicon_lacks_keeps_its_english_name() -> None:
    frame = Frame(b"", 3, (seen("wheelchair", "left"),))

    assert describe(frame, "ar")[0] == "قبل 3 ثوانٍ رأيت: wheelchair على يسارك."


@pytest.mark.parametrize(
    ("label", "spoken"),
    [("elevator", "an elevator"), ("keys", "keys"), ("bus", "a bus"), ("person", "a person")],
)
def test_english_articles(label: str, spoken: str) -> None:
    assert describe(Frame(b"", 5, (seen(label, "left"),)), "en")[0] == (
        f"5 seconds ago I saw {spoken} on your left."
    )


def test_each_label_once_at_its_best_position_and_at_most_three() -> None:
    # Detections come best first, as search_memory and latest_frame return them.
    detections = (
        seen("door", "left"),
        seen("door", "right", 0.5),
        seen("chair"),
        seen("table"),
        seen("person"),
    )

    answer, _ = describe(Frame(b"", 5, detections), "en")

    assert answer == "5 seconds ago I saw a door on your left, a chair ahead and a table ahead."


def test_sign_objects_are_left_out_when_the_text_is_read_and_two_texts_at_most() -> None:
    detections = (seen("exit sign"), seen("sign", "left"), seen("door"))

    answer, _ = describe(Frame(b"", 5, detections, ("EXIT", "Room 204", "Lab")), "en")

    assert answer == (
        "5 seconds ago I saw a door ahead, a sign that says 'EXIT' and a sign that says 'Room 204'."
    )


def test_a_sign_without_text_is_still_named() -> None:
    assert describe(Frame(b"", 5, (seen("exit sign"),)), "en")[0] == (
        "5 seconds ago I saw an exit sign ahead."
    )


@pytest.mark.parametrize("frame", [None, Frame(b"", 5)], ids=["no stored frame", "nothing found"])
@pytest.mark.parametrize("lang", ["en", "ar", "fr"])
def test_nothing_to_describe_is_not_sure(frame: Frame | None, lang: str) -> None:
    assert describe(frame, lang) == (gate.NOT_SURE[lang], "none")


def test_french_is_not_sure_until_it_has_wording() -> None:
    assert describe(Frame(b"", 5, (seen("door"),)), "fr") == (gate.NOT_SURE["fr"], "none")


@pytest.mark.parametrize(
    ("seconds", "english", "arabic"),
    [
        (0.3, "1 second", "ثانية"),
        (2, "2 seconds", "ثانيتين"),
        (7, "7 seconds", "7 ثوانٍ"),
        (45, "45 seconds", "45 ثانية"),
        (89, "89 seconds", "89 ثانية"),
        (120, "2 minutes", "دقيقتين"),
        (300, "5 minutes", "5 دقائق"),
        (60 * 25, "25 minutes", "25 دقيقة"),
    ],
)
def test_age_in_words(seconds: float, english: str, arabic: str) -> None:
    assert (age(seconds, "en"), age(seconds, "ar")) == (english, arabic)


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_a_full_template_is_one_short_sentence_the_safety_filter_keeps(lang: str) -> None:
    detections = (seen("door"), seen("stairs", "left"), seen("person", "right"))
    answer, _ = describe(Frame(b"", 14, detections, ("Room 204", "Computer Lab")), lang)

    assert safety.shorten(answer, 2, 30) == answer
    assert safety.rewrite(answer, lang) == (answer, False)
