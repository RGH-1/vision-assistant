# infra/

| File | Owner | Design |
|---|---|---|
| `docker-compose.yml` — PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, port 5432) | M2 | §5.2, §7 |
| `mediamtx.yml` — WHIP in :8889, WebRTC UDP :8189, RTSP :8554, path `phone` | M1 | §6.2 |
| MediaMTX download script (binary is git-ignored) | M1 | §6.2 |


## PostgreSQL Database (M2)

Uses PostgreSQL 16 with pgvector, running in Docker.

### First-time setup

From the repository root:

1. Copy `.env.example` to `.env`.
2. Add the following database variables to `.env`:

   POSTGRES_USER=va
   POSTGRES_PASSWORD=va
   POSTGRES_DB=vision

3. Keep DATABASE_URL consistent with those values.
4. Start Docker Desktop.

Never commit `.env`.

### Start database

```powershell
docker compose --env-file .env -f infra/docker-compose.yml up -d
```

### Check database status

```powershell
docker compose --env-file .env -f infra/docker-compose.yml ps
```

The database should report `healthy`.

### Stop database

```powershell
docker compose --env-file .env -f infra/docker-compose.yml down
```

The named volume preserves the database data.

### Reset database (DELETES ALL DATABASE DATA)

```powershell
docker compose --env-file .env -f infra/docker-compose.yml down -v
docker compose --env-file .env -f infra/docker-compose.yml up -d
```

### Database initialization

On a fresh database volume, PostgreSQL automatically executes SQL files in `server/db/`, mounted at `/docker-entrypoint-initdb.d/`.

The schema `server/db/001_init.sql` will be added in Task VA-9.

If the database already exists when that SQL file is added, execute it manually:

```powershell
docker compose --env-file .env -f infra/docker-compose.yml exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/001_init.sql'
```

Run this only after `001_init.sql` exists. Do not rerun it blindly against a database where the schema has already been applied.
