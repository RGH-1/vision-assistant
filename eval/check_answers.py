"""VA-72 check: ask the running server 10 questions about what the camera saw, with real Gemini.

Stream a video into the server first (docs/commands.md) and let it run for a minute or two.
Edit QUESTIONS so they match what your video shows. Then run from the repo root:
  uv run python -m eval.check_answers <session_id>
Each question costs at most one Gemini call (Flash's free tier allows 20 a day).
"""

import json
import sys
import urllib.error
import urllib.request

SERVER = "http://localhost:8000"
QUESTIONS = [
    ("en", "Did you see my keys?"),
    ("en", "What did we just pass?"),
    ("en", "What did the sign say?"),
    ("en", "Did you see a door?"),
    ("en", "Did you see a fire extinguisher?"),  # never in the video: should be "I'm not sure"
    ("ar", "وين شفت المفاتيح؟"),
    ("ar", "شو شفنا قبل شوي؟"),
    ("ar", "شو كان مكتوب عاللافتة؟"),
    ("ar", "شفت كرسي؟"),
    ("ar", "شو في قدامي؟"),
]


def main(session_id: str) -> None:
    for lang, text in QUESTIONS:
        body = json.dumps({"session_id": session_id, "text": text, "lang": lang}).encode()
        request = urllib.request.Request(
            f"{SERVER}/v1/ask_text", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            # Long enough for Gemini's timeout plus the Flash-Lite retry.
            with urllib.request.urlopen(request, timeout=90) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as error:
            print(f"{text}\n  HTTP {error.code}: {json.load(error)['error']['speak']}")
            continue
        found = f"{reply['intent']}, {reply['confidence']}, {len(reply['evidence'])} frames"
        print(f"{text}\n  [{found}] {reply['answer']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles may not print Arabic otherwise
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])