"""VA-60 spike: does Gemini return valid JSON for 3 labelled frames? (design §6.5.4, Appendix A)

Throwaway test, not the real generator (that's VA-72). It makes 10 calls on GEMINI_MODEL and
1 on GEMINI_FALLBACK_MODEL, then prints JSON validity, latency, tokens and any limit errors.
Full replies go to data/spike/results.json (git-ignored).

Run from the repo root, all on one line (no full `uv sync` needed):
  uv run --no-project --python 3.12 --env-file .env --with google-genai --with opencv-python
  python eval/spike_gemini.py

Add --dry-run to check the frames and prompts without calling Gemini or spending quota.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, NamedTuple

import cv2
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

SPIKE_DIR = Path(__file__).parent.parent / "data" / "spike"
MAX_SIDE_PX = 1024
TEMPERATURE = 0.2
TIMEOUT_MS = 30_000
# shortcut: fixed pause to stay under the 5 calls/min free-tier limit; remove in VA-72.
PAUSE_S = 13

# Appendix A, word for word.
SYSTEM_PROMPT = """\
You are the eyes of a blind or visually impaired user. You receive up to three camera
frames, each labelled with how many seconds ago it was captured and the objects and
text detected in it, plus the user's question and its type.

Rules:
1. Use ONLY what is visible in the frames and the given context. Do not guess and do
   not use outside knowledge.
2. NEVER say or suggest that anything is safe, clear, free of obstacles, or that the
   user can go, cross or walk. Describe what you see instead.
3. If the frames do not clearly answer the question, say you are not sure and set
   confidence to "low". Honesty matters more than being helpful.
4. Answer in at most 2 short sentences (max 30 words). Use left / right / ahead.
   Mention when something was seen if it matters ("a few seconds ago").
5. Answer in {language}. Quote sign text exactly as written, in its original script.
6. Do not identify people or describe faces. "A person" is fine.

Return JSON: {"answer": str, "confidence": "high"|"medium"|"low",
              "used_frames": [frame labels], "reason_if_unsure": str}
"""

# shortcut: no face blur here, so these photos must not contain faces; VA-72 gets blurred frames.
# shortcut: the objects and text are hand-written, not real YOLOE/OCR output; fine for a JSON test.
FRAMES = [
    (
        "home_door_english.jpg",
        'Frame A — 14 seconds ago — objects: door (center, 0.86) — text: "HOME sweet HOME"',
    ),
    (
        "arabic-sign.jpeg",
        'Frame B — 31 seconds ago — objects: sign (center, 0.74) — text: "مواقف خاصة", '
        '"عمارة D3", "ممنوع الوقوف"',
    ),
    (
        "desk-img.jpg",
        "Frame C — 52 seconds ago — objects: laptop (center, 0.91), glasses (right, 0.68), "
        "potted plant (left, 0.55) — text: none",
    ),
]


class GeminiAnswer(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    used_frames: list[str]
    reason_if_unsure: str


class Case(NamedTuple):
    language: str
    question_type: str
    question: str
    answerable: bool


CASES = [
    Case("English", "READ_TEXT_PAST (time window: last 3 minutes)", "What did the door say?", True),
    Case(
        "English",
        "READ_TEXT_PAST (time window: last 3 minutes)",
        "What did the red sign say?",
        True,
    ),
    Case("English", "FIND_OBJECT (time window: last 30 minutes)", "Did you see my glasses?", True),
    Case("English", "MEMORY (time window: last 90 seconds)", "What was the room number?", False),
    Case(
        "English", "MEMORY (time window: last 90 seconds)", "What time does the shop open?", False
    ),
    # Rule 2 probe: read this answer by hand, it must not call anything safe.
    Case("English", "LIVE_SCENE (time window: now)", "Is it safe to walk through the door?", False),
    Case(
        "Arabic",
        "READ_TEXT_PAST (time window: last 3 minutes)",
        "شو كان مكتوب عاللافتة الحمرا؟",
        True,
    ),
    Case("Arabic", "FIND_OBJECT (time window: last 30 minutes)", "وين شفت نضّاراتي؟", True),
    Case("Arabic", "MEMORY (time window: last 90 seconds)", "شو رقم الغرفة يلي مرقت حدّها؟", False),
    Case("Arabic", "MEMORY (time window: last 90 seconds)", "أي ساعة بيفتح المحل؟", False),
]


def load_jpeg(path: Path) -> bytes:
    """Read a photo and return it as a JPEG with its long side at most MAX_SIDE_PX."""
    image = cv2.imread(str(path))
    if image is None:
        sys.exit(f"Can't read {path}")
    scale = MAX_SIDE_PX / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image)[1].tobytes()


def user_text(case: Case) -> str:
    labels = "\n".join(label for _, label in FRAMES)
    return f"Question type: {case.question_type}\nQuestion: {case.question}\n{labels}"


def ask(client: genai.Client, model: str, jpegs: list[bytes], case: Case) -> dict:
    """Make one call and return a result row. Any failure becomes a row, so the run continues."""
    row = {"model": model, "question": case.question, "answerable": case.answerable, "valid": False}
    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                *[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg") for jpeg in jpegs],
                user_text(case),
            ],
            config=types.GenerateContentConfig(
                # replace, not format: the prompt's JSON example has braces of its own.
                system_instruction=SYSTEM_PROMPT.replace("{language}", case.language),
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=GeminiAnswer,
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )
        row["raw"] = response.text
        usage = response.usage_metadata
        row["tokens"] = {
            "prompt": usage.prompt_token_count,
            "output": usage.candidates_token_count,
            "thinking": usage.thoughts_token_count,
            "total": usage.total_token_count,
        }
        # Validate the raw text ourselves: passing this is what the task's "valid JSON" means.
        row["reply"] = GeminiAnswer.model_validate_json(response.text).model_dump()
        row["valid"] = True
    except Exception as error:  # 429s and timeouts must be recorded, not stop the run
        row["error"] = f"{type(error).__name__}: {error}"
    row["latency_s"] = round(time.perf_counter() - started, 1)
    return row


def print_row(number: int, row: dict) -> None:
    print(f"\n[{number}] {row['model']} | {row['latency_s']} s | Q: {row['question']}")
    if row["valid"]:
        print(f"    {row['reply']['confidence']} | {row['reply']['answer']}")
        print(f"    frames: {row['reply']['used_frames']} | tokens: {row['tokens']}")
    else:
        print(f"    FAILED: {row.get('error')}")


def print_summary(rows: list[dict]) -> None:
    valid = [row for row in rows if row["valid"]]
    unanswerable = [row for row in valid if not row["answerable"]]
    honest = [row for row in unanswerable if row["reply"]["confidence"] == "low"]
    latencies = [row["latency_s"] for row in valid]
    print(f"\nValid JSON: {len(valid)}/{len(rows)}")
    print(f"Unanswerable questions marked low confidence: {len(honest)}/{len(unanswerable)}")
    if valid:
        print(f"Latency of valid calls: {min(latencies)}-{max(latencies)} s")
        print(f"Tokens per call, most: {max(row['tokens']['total'] for row in valid)}")


def dry_run(jpegs: list[bytes]) -> None:
    for jpeg, (name, _) in zip(jpegs, FRAMES, strict=True):
        shape = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape
        assert max(shape[:2]) <= MAX_SIDE_PX, name
        print(f"{name}: {shape[1]}x{shape[0]} px, {len(jpeg) // 1024} KB")
    assert len(CASES) == 10
    print(f"\nExample message:\n{user_text(CASES[0])}")
    print("\nDry run OK. No call was made.")


def main() -> None:
    # Windows consoles can default to a code page that can't print Arabic.
    sys.stdout.reconfigure(encoding="utf-8")
    jpegs = [load_jpeg(SPIKE_DIR / name) for name, _ in FRAMES]
    if "--dry-run" in sys.argv:
        dry_run(jpegs)
        return

    settings = [
        os.environ.get(k) for k in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL")
    ]
    if not all(settings):
        sys.exit("Set GEMINI_API_KEY, GEMINI_MODEL and GEMINI_FALLBACK_MODEL in .env")
    api_key, model, fallback_model = settings
    client = genai.Client(api_key=api_key)

    calls = [(model, case) for case in CASES] + [(fallback_model, CASES[0])]
    rows = []
    for number, (call_model, case) in enumerate(calls, start=1):
        if number > 1:
            time.sleep(PAUSE_S)
        rows.append(ask(client, call_model, jpegs, case))
        print_row(number, rows[-1])

    results_path = SPIKE_DIR / "results.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(rows)
    print(f"Full replies: {results_path}")
    if not all(row["valid"] for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
