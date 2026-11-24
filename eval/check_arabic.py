"""VA-75 check: 10 spoken Arabic questions on the real phone, turned into a results table.

1. Start the server and the phone stream, set the app to Arabic, and ask QUESTIONS by voice,
   in order, each once. Point the camera at a sign for the reading questions.
2. Then run from the repo root, with the phone's session id (the newest row in sessions):
     uv run python -m eval.check_arabic <session_id>
   It prints a Markdown table for the VA-75 Notion page. The intent and "answer in Arabic"
   columns are checked here; "sign quoted" and "Arabic voice" are filled in by ear.
"""

import json
import re
import sys
import urllib.request
from typing import Any

SERVER = "http://localhost:8000"
# What to say and the intent it should get: 5 Lebanese, then 5 standard Arabic, all 7 intents.
QUESTIONS = [
    ("شو في قدامي؟", "LIVE_SCENE"),
    ("شو مكتوب هون؟", "READ_TEXT_NOW"),
    ("شو شفنا قبل شوي؟", "MEMORY"),
    ("وين تركت مفاتيحي؟", "FIND_OBJECT"),
    ("مين هيدا؟", "IDENTITY"),
    ("ماذا يوجد أمامي؟", "LIVE_SCENE"),
    ("ماذا كان مكتوبا على اللافتة؟", "READ_TEXT_PAST"),
    ("ماذا رأينا قبل قليل؟", "MEMORY"),
    ("هل رأيت الباب؟", "FIND_OBJECT"),
    ("ما هي عاصمة فرنسا؟", "OUT_OF_SCOPE"),
]
ARABIC_LETTER = re.compile(r"[\u0621-\u064a]")


def table(asked: list[dict[str, Any]]) -> str:
    """One row per question, matched by order to the Arabic voice questions, oldest first."""
    lines = [
        "| # | Said | Transcript | Intent | Answer | In Arabic | Sign quoted | Arabic voice |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for number, ((said, expected), row) in enumerate(zip(QUESTIONS, asked, strict=False), 1):
        intent = row["intent"]
        intent = f"✅ {intent}" if intent == expected else f"❌ {intent} (expected {expected})"
        in_arabic = "✅" if ARABIC_LETTER.search(row["answer"]) else "❌"
        # A | in the text would end the Markdown cell early.
        cells = [str(number), said, row["transcript"], intent, row["answer"], in_arabic, "", ""]
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
    return "\n".join(lines)


def main(session_id: str) -> None:
    url = f"{SERVER}/v1/questions?session_id={session_id}&limit=100"
    with urllib.request.urlopen(url, timeout=30) as response:
        rows = json.load(response)
    # The route returns newest first.
    asked = [r for r in reversed(rows) if r["lang"] == "ar" and r["input_mode"] == "voice"]
    if len(asked) != len(QUESTIONS):
        print(f"Warning: {len(asked)} Arabic voice questions in this session, expected 10.\n")
    print(table(asked))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles may not print Arabic otherwise
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
