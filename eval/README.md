# eval/ — Evaluation (M3)

- `check_stt.py` — VA-63 check: transcribes the recordings in `data/stt_check/` with real Groq
- Set `GROQ_API_KEY=bad` in the shell before `check_stt.py` or `check_ask.py` to test the VA-66 local fallback (the first run downloads the model)
- `questions.csv` — 30–40 test questions: `video, ask_at_s, question, lang, intent, relevant_from_s, relevant_to_s, expected_answer, answerable`
- `run.py` — streams a video and sends each question to `/v1/ask_text` at `ask_at_s`
- `score.py` — Recall@1/3, intent accuracy, answer accuracy, made-up rate
- `results/` — outputs that go into `docs/results.md`

Gemini replies are cached in `eval/cache/` (git-ignored) so re-scoring spends no quota. See design §11.
