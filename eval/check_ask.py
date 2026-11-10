"""VA-64 check: ask real Gemini and Groq about a photo, through the real /v1/ask routes.

A photo stands in for the phone camera, which isn't streaming yet. It uses the real database,
in a session that is deleted at the end. The recording's language comes from its name (en_1.wav).

Run from the repo root, with Postgres running:
  uv run python -m eval.check_ask data/spike/desk-img.jpg data/stt_check/en_1.wav
"""

import asyncio
import sys
import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from server import main
from server.settings import load_settings

TEXT_QUESTIONS = {
    "en": "What's in front of me?",
    "ar": "شو في قدامي؟",
    "fr": "Qu'y a-t-il devant moi ?",
}


class NoIngest:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def show(label: str, response) -> None:
    body = response.json()
    if response.status_code == 200:
        print(f"{label}: {body['transcript']}\n  -> {body['answer']} ({body['confidence']})")
        print(f"  timings: {body['timings_ms']}")
    else:
        print(f"{label}: HTTP {response.status_code} {body['error']}")


def run(photo: Path, recording: Path) -> None:
    image = cv2.imread(str(photo))
    if image is None:
        sys.exit(f"Can't read {photo}")
    lang = recording.stem.split("_")[0]

    # No camera stream, so no ingest thread and no CLIP model: only the answer path runs.
    main.create_ingest_worker = lambda _app: NoIngest()
    main.load_clip_text_encoder = lambda _name: None
    app = main.create_app(load_settings())
    with TestClient(app, backend_options={"loop_factory": asyncio.SelectorEventLoop}) as client:
        session = {"app_lang": lang, "consent_at": "2026-11-08T00:00:00Z", "consent_ver": "check"}
        session_id = client.post("/v1/sessions", json=session).json()["session_id"]
        try:
            app.state.live_buffer.add(image, time.monotonic())
            app.state.stream_connected = True
            text = {"session_id": session_id, "text": TEXT_QUESTIONS[lang], "lang": lang}
            show("Typed", client.post("/v1/ask_text", json=text))
            show(
                "Spoken",
                client.post(
                    "/v1/ask",
                    data={"session_id": session_id, "lang": lang},
                    files={"audio": (recording.name, recording.read_bytes(), "audio/wav")},
                ),
            )
        finally:
            client.delete(f"/v1/sessions/{session_id}")


if __name__ == "__main__":
    # Windows consoles may not print Arabic otherwise.
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
