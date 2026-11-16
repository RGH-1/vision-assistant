"""Never say "safe": rewrite such sentences and keep answers short (design §6.5.5, Appendix C)."""

import re

from server.common.schemas import Lang

# Appendix C. Arabic words need their own boundaries, so "الثامن" (the eighth) doesn't match "امن".
FORBIDDEN = re.compile(
    "|".join(
        [
            r"\bsafe(?:ly)?\b",
            r"\bclear to\b",
            r"\ball clear\b",
            r"\byou can (?:cross|go|walk|proceed)\b",
            r"\bno (?:obstacles?|danger|hazards?)\b",
            r"\bnothing in (?:your|the) way\b",
            # Not in Appendix C, but AGENTS.md forbids "clear" too.
            r"\b(?:path|way|road|hallway|corridor|street) (?:is|looks|seems) (?:clear|free)\b",
            r"(?<!\w)[وف]?(?:ال)?[آأا]من(?:ة|ه|ا|ون|ين)?(?!\w)",  # آمن, آمنة, الآمن, امن
            r"(?<!\w)[وف]?ب[أا]مان(?!\w)",  # بأمان
            r"ما في خطر",
            r"لا (?:يوجد|توجد) (?:أي )?(?:خطر|عوائق|عائق)",
            r"لا عوائق",
            r"فيك تقطع",
            r"يمكنك (?:العبور|المرور|المشي|التقدم)",
            r"الطريق (?:فاضي|خال|سالك)",
            r"(?<!pas )(?<!bien )\bsûr(?:e|s|es)?\b",  # not "pas sûr" (not sure), "bien sûr"
            r"\bsans danger\b",
            r"\ben sécurité\b",
            r"\bvous pouvez (?:traverser|avancer|passer)\b",
            r"\baucun (?:obstacle|danger)\b",
            r"\bla voie est libre\b",
        ]
    ),
    re.IGNORECASE,
)
ARABIC_MARKS = re.compile(r"[\u064b-\u0652\u0640]")  # short vowels and tatweel: "آمِن" is "آمن"
SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
CANE: dict[Lang, str] = {
    "en": "I can't tell if it's safe. Please use your cane.",
    "ar": "ما بقدر إعرف إذا آمن. استعمل العصا من فضلك.",
    "fr": "Je ne peux pas dire si c'est sûr. Utilisez votre canne.",
}


def shorten(answer: str, max_sentences: int, max_words: int) -> str:
    """At most max_sentences, ending early past max_words; the first sentence is always kept."""
    kept: list[str] = []
    for sentence in SENTENCE_END.split(answer.strip()):
        words = len(" ".join([*kept, sentence]).split())
        if len(kept) == max_sentences or (kept and words > max_words):
            break
        kept.append(sentence)
    return " ".join(kept)


def rewrite(answer: str, lang: Lang) -> tuple[str, bool]:
    """Each sentence that says or suggests "safe" becomes the cane message; True if any did."""
    sentences: list[str] = []
    rewritten = False
    for sentence in SENTENCE_END.split(answer.strip()):
        if FORBIDDEN.search(ARABIC_MARKS.sub("", sentence)):
            rewritten = True
            if CANE[lang] in sentences:
                continue  # said once is enough
            sentence = CANE[lang]
        sentences.append(sentence)
    return " ".join(sentences), rewritten
