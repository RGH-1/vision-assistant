# Vision Assistant

A voice assistant for blind and visually impaired users. The phone streams live video to a laptop; the laptop remembers sharp, informative frames in a multimedia database (PostgreSQL + pgvector); the user holds the screen, asks a question in English or Arabic, and hears a short, grounded answer.

> **A helper, not a replacement for a cane or guide dog.**

Course project — Multimedia Databases and Image Processing, 2026–2027. Demo: **Dec 2, 2026**.

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

## Setup (once)

Requirements: Windows 11, Python **3.12** via [uv](https://docs.astral.sh/uv/), Docker Desktop, Node 20+, Flutter with an Android phone in developer mode, an NVIDIA GPU with CUDA.

1. `copy .env.example .env`, then fill in your own keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, the `POSTGRES_*` values). `ELEVENLABS_API_KEY` is optional; without it the phone uses its own voice.
2. `uv sync` (creates `.venv` with Python 3.12).
3. `.\infra\scripts\get_mediamtx.ps1` downloads `mediamtx.exe` into `infra/` (it isn't in git).
4. In an **Administrator** PowerShell: `.\infra\scripts\firewall.ps1`. It opens ports 8000, 8889 and 8189 for the phone; the firewall itself stays on. The rules only apply on **Private** networks, so set the Wi-Fi to Private (Settings → Network & internet → Wi-Fi → your network).
5. `cd dashboard ; npm install`
6. Check: `uv run --env-file .env pytest`

The first server start downloads the model files, which takes a few minutes.

## Run

1. `.\start.ps1` finds the laptop's LAN IP and writes it into `infra/mediamtx.yml`, the phone app and the dashboard. Then it starts Postgres, MediaMTX and the server, each in its own window. Don't commit the files it changed.
2. **Phone app**, only if the laptop's IP changed since the app was last built: plug the phone in by USB, then `cd mobile ; flutter run --release -d <device-id>`. (You can check device Id with `flutter devices`)
3. **Dashboard:** `cd dashboard ; npm run dev`, then open http://localhost:5173.
4. **Stop:** close the server and MediaMTX windows, then `docker compose --env-file .env -f infra/docker-compose.yml stop`.

The phone and laptop must be on the same Wi-Fi, with any VPN turned off. More commands (tests, a fake phone stream, API calls): [docs/commands.md](docs/commands.md).

## Branches

`main` (stable, demo-ready) ← `dev` (integration) ← feature branches `m1/va-<id>-<name>`, `m2/…`, `m3/…` (one per Notion task, cut from `dev`, merged back by reviewed PR). Details in [AGENTS.md](AGENTS.md#git).

## Team

| | Area |
|---|---|
| **M1** | Mobile, streaming, dashboard |
| **M2** | Images and database |
| **M3** | AI assistant and evaluation |
