"""Text normalization used for stored searchable scene text (design §6.3.5)."""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670]")
_SPACE = re.compile(r"\s+")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_text(value: str) -> str:
    """Return a compact matching form while retaining the original OCR text separately."""
    normalized = unicodedata.normalize("NFKD", value.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    normalized = normalized.replace("ـ", "").translate(_ARABIC_DIGITS)
    normalized = normalized.translate(str.maketrans("أإآىة", "ااايه"))
    return _SPACE.sub(" ", normalized).strip()
