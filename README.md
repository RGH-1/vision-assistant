# Vision Assistant

A voice assistant for blind and visually impaired users. The phone streams live video to a laptop; the laptop remembers sharp, informative frames in a multimedia database (PostgreSQL + pgvector); the user holds the screen, asks a question in English or Arabic, and hears a short, grounded answer.

> **A helper, not a replacement for a cane or guide dog.**

Course project — Multimedia Databases and Image Processing, 2026–2027. Demo: **Nov 25, 2026**.

- **Design (source of truth):** [Vision_Assistant_Design.md](Vision_Assistant_Design.md)
- **Architecture diagram:** [docs/architecture.drawio](docs/architecture.drawio)
- **Agent rules:** [AGENTS.md](AGENTS.md) (Claude Code reads it via `CLAUDE.md`)

## Layout

| Folder | What | Owner |
|---|---|---|
| `contracts/` | Frozen REST + WebSocket contracts | All |
| `infra/` | Postgres (Docker), MediaMTX config | M1, M2 |
| `server/` | FastAPI + ingest thread (`uv run python -m server`) | M2, M3 |
| `mobile/` | Flutter Android app | M1 |
| `dashboard/` | React + Vite + TS dashboard | M1 |
| `eval/` | Test questions, runner, scoring | M3 |
| `docs/` | Diagram, results, consent form, slides | All |
| `data/` | Frames, videos, model files — **not in git** | — |

## Setup (work in progress)

Requirements: Python **3.12** via [uv](https://docs.astral.sh/uv/), Docker Desktop, Node 20+, Flutter (Android), an NVIDIA GPU with CUDA.

```powershell
copy .env.example .env        # then fill in your own API keys
uv sync                       # creates .venv with Python 3.12
uv run pytest                 # server tests
```

Start-up (`start.ps1`), firewall rules and the demo steps are added as those tasks land (design §5.2, §16).

## Branches

`main` (stable, demo-ready) ← `dev` (integration) ← feature branches `m1/va-<id>-<name>`, `m2/…`, `m3/…` (one per Notion task, cut from `dev`, merged back by reviewed PR). Details in [AGENTS.md](AGENTS.md#git).

## Team

| | Area |
|---|---|
| **M1** | Mobile, streaming, dashboard |
| **M2** | Images and database |
| **M3** | AI assistant and evaluation |
