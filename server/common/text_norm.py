"""Shared OCR and query normalization for text search (design §6.3.5)."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

_SPACE = re.compile(r"\s+")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_SEARCH_TOKEN = re.compile(r"[a-z0-9]+|[\u0621-\u063a\u0641-\u064a0-9]+")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_LETTERS = str.maketrans("أإآىة", "ااايه")

Language = Literal["en", "ar", "fr"]

STOP_WORDS: dict[Language, frozenset[str]] = {
    "en": frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "at",
            "did",
            "do",
            "for",
            "i",
            "in",
            "is",
            "me",
            "my",
            "of",
            "on",
            "or",
            "that",
            "the",
            "there",
            "this",
            "to",
            "was",
            "were",
            "what",
            "you",
        }
    ),
    "ar": frozenset(
        {
            "انا",
            "او",
            "الى",
            "عن",
            "على",
            "في",
            "كان",
            "كانت",
            "ما",
            "ماذا",
            "مع",
            "من",
            "هذا",
            "هذه",
            "هل",
            "هو",
            "هي",
            "و",
        }
    ),
    "fr": frozenset(
        {
            "a",
            "au",
            "aux",
            "ce",
            "cette",
            "dans",
            "de",
            "des",
            "du",
            "est",
            "et",
            "je",
            "la",
            "le",
            "les",
            "mon",
            "que",
            "qui",
            "sur",
            "tu",
            "un",
            "une",
        }
    ),
}


def normalize_text(value: str) -> str:
    """Return a compact form for matching while callers retain original OCR separately."""
    normalized = unicodedata.normalize("NFKD", value.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("ـ", "").translate(_ARABIC_DIGITS)
    normalized = normalized.translate(_ARABIC_LETTERS)
    return _SPACE.sub(" ", normalized).strip()


def build_search_text(value: str, language: Language | None = None) -> str:
    """Normalize text and drop language stop words before a search query is built."""
    normalized = normalize_text(value)
    selected_language = language or ("ar" if _ARABIC.search(normalized) else "en")
    if selected_language not in STOP_WORDS:
        raise ValueError(f"unsupported language: {selected_language}")
    return " ".join(
        token
        for token in _SEARCH_TOKEN.findall(normalized)
        if token not in STOP_WORDS[selected_language]
    )
