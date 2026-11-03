# AGENTS.md — rules for coding agents

> Read by every coding agent (Codex, Cursor, Copilot, Gemini CLI, …). Claude Code gets the same rules through `CLAUDE.md`, which only imports this file. **Edit this file, never CLAUDE.md.**

**Source of truth: [Vision_Assistant_Design.md](Vision_Assistant_Design.md).** Read the section named in your task before writing code. If the code and the design disagree, stop and ask — don't silently pick one.

University project, 3 part-time devs, hard demo on **Nov 25, 2026**. **Latency is not graded; a working, honest product is.** Prefer simple, readable code the owner can explain at the demo over clever or fast code.

## Where things are in the design

| Need | Section |
|---|---|
| Tech choices (the only allowed libraries/services) | §4 |
| Architecture, ports, how the thread and web routes share state | §5 |
| Phone app / streaming | §6.1, §6.2 |
| Ingest pipeline (quality, enhancement, dHash, YOLOE, OCR, face blur, CLIP) | §6.3 |
| Assistant (Whisper, 7 intents, ranking, evidence gate, Gemini, safety) | §6.5 |
| Alerts (bonus) | §6.6 |
| Dashboard | §6.7 |
| SQL schema and hybrid search | §7.3, §7.4 |
| REST + WebSocket contracts, `.env` keys | §8 |
| Privacy rules | §9 |
| Fallbacks | §10.3 |
| Evaluation | §11 |

Tasks live in the Notion hub (Tasks database). Each task names its owner (M1/M2/M3/All) and design section.

## The 6 rules

1. **Contracts first.** Build against `contracts/` (§8) and the schema in §7.3. Never change them — that's a team decision (all three members).
2. **One agent, one folder, one task.** Stay inside the folder your task owns (table below). Need a change elsewhere? Say so in your summary instead of making it.
3. **Every task ends with a test or a runnable check** — pytest, `flutter test`, `npm test`, or a test video through the pipeline. Run it and report the real result.
4. **The owner reads and understands every line before merging.** Keep diffs small, explain non-obvious choices in the PR, no drive-by refactors.
5. **No new libraries or services** unless they're in §4. If one is truly needed, propose updating §4 first.
6. **API keys only in `.env`.** Never in code, tests, prompts, logs or commits. `.env.example` holds names only.

## Folder ownership

| Folder | Owner | Notes |
|---|---|---|
| `server/ingest/`, `server/db/`, `server/common/db.py`, `server/common/queries.sql` | M2 | GPU models load **inside the ingest thread** |
| `server/main.py`, `server/events.py`, `server/assistant/`, `server/alerts/`, `server/common/schemas.py`, `eval/` | M3 | |
| `server/common/text_norm.py` | M2 + M3 | used when storing **and** when searching — change both sides together |
| `mobile/`, `dashboard/`, `infra/mediamtx.yml`, `start.ps1` | M1 | |
| `infra/docker-compose.yml` | M2 | |
| `contracts/`, `Vision_Assistant_Design.md`, `AGENTS.md`, `CLAUDE.md` | All | team decision to change |

## Commands

The dev machine is **Windows 11**; the default shell is PowerShell.

```powershell
# Server (Python 3.12 via uv — NOT the system Python 3.14; GPU wheels lag behind)
uv sync                          # install / update .venv
uv run python -m server          # FastAPI on :8000 + ingest thread
uv run pytest                    # tests (server/tests, eval)
uv run ruff check . ; uv run ruff format .
uv add <pkg>                     # only for libraries listed in §4

# Postgres + pgvector
docker compose -f infra/docker-compose.yml up -d

# Fake phone stream (so server work never waits for the app)
ffmpeg -re -stream_loop -1 -i data/videos/walk_01.mp4 -c:v libx264 -preset veryfast -tune zerolatency -f rtsp rtsp://localhost:8554/phone

# Mobile (in mobile/)
flutter analyze ; flutter test ; flutter run

# Dashboard (in dashboard/)
npm install ; npm run dev ; npm run lint ; npm test ; npm run build
```

## Code style

- **Python:** `ruff` clean (config in `pyproject.toml`), type hints everywhere, **pydantic models for every request and response**, `pytest`. Thresholds and weights go in `server/config.yaml`, settings in `server/settings.py` — no magic numbers in code.
- **Dart/Flutter:** `flutter analyze` clean; every control has a `Semantics` label (§6.1.3).
- **TypeScript:** strict mode, ESLint clean; Arabic text rendered with `dir="auto"`.
- Match the surrounding code's naming and comment density. Comments explain *why*, not *what*.

## Invariants — never break these

- **Never say "safe"** (or "clear", "no obstacles") in any answer or alert. Run text through the safety filter (§6.5.5, Appendix C).
- **No face recognition, identity, age, gender or emotion models.** `IDENTITY` questions are refused **in code**, before Gemini (§6.5.2).
- **Faces are blurred before a frame is written to disk or sent anywhere** (§6.3.6).
- **Audio is deleted right after transcription** and never stored (§6.5.1).
- **Weak evidence → "I'm not sure", without calling Gemini** (evidence gate, §6.5.5).
- **The demo never goes silent:** every failure path returns a `speak` message (§8.1) and uses the fallbacks in §10.3.
- **Thread safety (§5.3):** the ingest thread reaches the asyncio loop only via `hub.publish()` (`loop.call_soon_threadsafe`). The live buffer is guarded by its lock. The ingest thread has its own sync `psycopg` connection; web routes use the async pool. Never share one across them.
- Store what will be reported (quality numbers, keep reasons, timings, score parts) — results come from the tables.

## Git

| Branch | Purpose | Merges into |
|---|---|---|
| `main` | Stable, always demo-ready. Only milestone merges from `dev` + tags. Nobody commits directly. | — |
| `dev` | Integration: all finished features land here. Must always run. | `main` at each milestone |
| `m1/…`, `m2/…`, `m3/…`, `all/…` | One feature branch per Notion task, cut from `dev`, short-lived | `dev` via PR |

- Name feature branches `<owner>/va-<id>-<short-name>`, e.g. `m2/va-12-db-schema` (ID from the Notion task).
- Start: `git switch dev ; git pull ; git switch -c m2/va-12-db-schema`. Keep it up to date with `git pull origin dev` (merge, no rebase of pushed branches).
- PR into `dev` → one other member reviews → merge → delete the branch. Put the PR link in the task's **PR / Branch** field.
- Agents work only on a feature branch — never commit to `dev` or `main` directly.
- Fix for a demo-breaking bug on `main`: branch `hotfix/<name>` from `main`, PR into `main`, then merge `main` back into `dev`.
- Commit under the human owner's name (it's the evidence for "who did what"). Never commit `.env`, `data/`, or model weights.
- Milestones: merge `dev` → `main`, then tag `main`: `v0.1-skeleton` (Nov 10), `v0.5-complete` (Nov 17), `v1.0-demo` (Nov 24, 18:00 code freeze).

## When you finish a task

Report: what changed (files), how you checked it (command + real output), anything not done or not verified, and any change you think is needed outside your folder.
