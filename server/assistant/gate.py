"""Replies that never call Gemini: fixed refusals and "I'm not sure" (design §6.5.5)."""

from server.common.schemas import Intent, Lang

# Refused in code, so these questions never reach Gemini (§6.5.2, §9).
FIXED_REPLIES: dict[Intent, dict[Lang, str]] = {
    "IDENTITY": {
        "en": "I can't identify people, but I can tell you what I see.",
        "ar": "لا أستطيع التعرف على الأشخاص، لكن يمكنني أن أصف لك ما أراه.",
        "fr": "Je ne peux pas identifier les personnes, mais je peux vous dire ce que je vois.",
    },
    "OUT_OF_SCOPE": {
        "en": "I can only answer questions about what the camera has seen.",
        "ar": "يمكنني فقط الإجابة عن الأسئلة حول ما رأته الكاميرا.",
        "fr": "Je peux seulement répondre aux questions sur ce que la caméra a vu.",
    },
}
# The evidence gate's answer when nothing stored can answer the question.
NOT_SURE: dict[Lang, str] = {
    "en": "I'm not sure. I haven't seen anything that answers that.",
    "ar": "لست متأكدًا. لم أرَ شيئًا يجيب عن ذلك.",
    "fr": "Je ne suis pas certain. Je n'ai rien vu qui réponde à cela.",
}
