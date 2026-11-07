"""VA-63 check: transcribe recorded questions with real Groq, to compare by ear with what was said.

Put 16 kHz mono WAV files in data/stt_check/, named by language: en_1.wav ... ar_5.wav.

Run from the repo root:
  uv run python -m eval.check_stt
"""

import asyncio
import os
import sys

from server.assistant.stt import transcribe
from server.settings import load_settings


async def main() -> None:
    settings = load_settings()
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is empty in .env")

    recordings = sorted((settings.data_dir / "stt_check").glob("*.wav"))
    if not recordings:
        sys.exit(f"No WAV files in {settings.data_dir / 'stt_check'}")

    for recording in recordings:
        lang = recording.stem.split("_")[0]
        transcript = await transcribe(recording.read_bytes(), lang, settings)
        text = transcript.text or "(not understood)"
        print(f"{recording.name}  {transcript.time_ms:.0f} ms  {text}")


if __name__ == "__main__":
    # Windows consoles may not print Arabic otherwise.
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
