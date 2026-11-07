import pytest

from server.common.text_norm import build_search_text, normalize_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("CAFÉ", "cafe"),
        ("Crème brûlée", "creme brulee"),
        ("  Door   ahead ", "door ahead"),
        ("المَخْرَج", "المخرج"),
        ("المخرج", "المخرج"),
        ("مــرحــبــا", "مرحبا"),
        ("أإآ", "ااا"),
        ("فتى", "فتي"),
        ("غرفة", "غرفه"),
        ("غرفة ٢٠٤", "غرفه 204"),
        ("  مَخْرَج\t  Café ", "مخرج cafe"),
    ],
)
def test_normalize_text(source: str, expected: str) -> None:
    assert normalize_text(source) == expected


def test_diacritic_and_plain_arabic_match() -> None:
    assert normalize_text("المَخْرَج") == normalize_text("المخرج")


@pytest.mark.parametrize(
    ("source", "language", "expected"),
    [
        ("What is the door in front of me?", "en", "door front"),
        ("شو في قدامي؟", "ar", "شو قدامي"),
        ("ما هذا المَخْرَج؟", "ar", "المخرج"),
        ("Le panneau est sur la porte", "fr", "panneau porte"),
        ("The café and the stairs", "en", "cafe stairs"),
        ("في غرفة ٢٠٤", None, "غرفه 204"),
        ("the and of", "en", ""),
    ],
)
def test_build_search_text_removes_language_stop_words(
    source: str, language: str | None, expected: str
) -> None:
    assert build_search_text(source, language) == expected  # type: ignore[arg-type]


def test_build_search_text_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        build_search_text("door", "de")  # type: ignore[arg-type]
