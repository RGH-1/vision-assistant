# Run and test commands

Run everything from the repository root. Most commands are the same in PowerShell, Git Bash and cmd; only the parts in the table below differ.

## Shell differences

| | PowerShell (5.1) | Git Bash | cmd |
|---|---|---|---|
| Run several commands | `a ; b` (no `&&` in 5.1) | `a && b` | `a && b` |
| Set a variable for this window | `$env:GROQ_API_KEY="bad"` | `export GROQ_API_KEY=bad` | `set GROQ_API_KEY=bad` |
| Remove it | `Remove-Item Env:GROQ_API_KEY` | `unset GROQ_API_KEY` | `set GROQ_API_KEY=` |
| Run MediaMTX | `.\infra\mediamtx.exe infra\mediamtx.yml` | `./infra/mediamtx.exe infra/mediamtx.yml` | `infra\mediamtx.exe infra\mediamtx.yml` |
| Run a `.ps1` script | `.\start.ps1` | `powershell -ExecutionPolicy Bypass -File start.ps1` | `powershell -ExecutionPolicy Bypass -File start.ps1` |
| Call the API | `Invoke-RestMethod` | `curl` | `curl.exe` |

In PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`, so use `Invoke-RestMethod` (or `curl.exe`). A variable set in the shell wins over `.env`.

## Same in every shell

```
uv sync
uv run --env-file .env pytest
uv run --env-file .env pytest server/tests/test_pipeline.py -v
uv run ruff check <files> ; uv run ruff format <files>
docker compose --env-file .env -f infra/docker-compose.yml up -d
uv run python -m server
uv run python -m server.ingest.detection --warmup
uv run python -m eval.try_ranking
uv run python -m eval.check_stt
docker exec vision-assistant-db-1 psql -U va -d vision -c "SELECT keep_reason, count(*) FROM frames GROUP BY 1;"
```

Tests that use the database fail without `--env-file .env`. Git Bash uses `&&` instead of `;` between the two ruff commands.

## Live check: video → server → database

Use four windows: MediaMTX, the server, the video, and one for the calls below.

Stream a test video to `rtsp://localhost:8554/phone`. Until it runs, the server logs `DESCRIBE failed: 404 Not Found`; that's expected.

```
powershell -File infra/scripts/push_test_video.ps1 -VideoPath data/videos/walk_01.mp4
```

### PowerShell

```powershell
$sid = (Invoke-RestMethod -Method Post http://localhost:8000/v1/sessions -ContentType application/json -Body '{"app_lang":"en","consent_at":"2026-11-16T12:00:00Z","consent_ver":"manual-check"}').session_id ; $sid
(Invoke-RestMethod http://localhost:8000/v1/status).frames_in_db
Invoke-RestMethod -Method Post http://localhost:8000/v1/ask_text -ContentType application/json -Body (@{session_id=$sid; text="Where did I leave my keys?"; lang="en"} | ConvertTo-Json) | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Delete http://localhost:8000/v1/sessions/$sid
```

### Git Bash

```bash
curl -s -X POST http://localhost:8000/v1/sessions -H "Content-Type: application/json" -d '{"app_lang":"en","consent_at":"2026-11-16T12:00:00Z","consent_ver":"manual-check"}'
sid=<paste the session_id from the reply>
curl -s http://localhost:8000/v1/status
curl -s -X POST http://localhost:8000/v1/ask_text -H "Content-Type: application/json" -d "{\"session_id\":\"$sid\",\"text\":\"Where did I leave my keys?\",\"lang\":\"en\"}"
curl -s -X DELETE http://localhost:8000/v1/sessions/$sid
```

### cmd

```bat
curl.exe -s -X POST http://localhost:8000/v1/sessions -H "Content-Type: application/json" -d "{\"app_lang\":\"en\",\"consent_at\":\"2026-11-16T12:00:00Z\",\"consent_ver\":\"manual-check\"}"
set SID=<paste the session_id from the reply>
curl.exe -s http://localhost:8000/v1/status
curl.exe -s -X POST http://localhost:8000/v1/ask_text -H "Content-Type: application/json" -d "{\"session_id\":\"%SID%\",\"text\":\"Where did I leave my keys?\",\"lang\":\"en\"}"
curl.exe -s -X DELETE http://localhost:8000/v1/sessions/%SID%
```

Deleting the session also deletes its frames and their files.