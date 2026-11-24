# Recorded-session fixture

Generate the fixture from a clean local database:

```powershell
uv run python -m server.ingest.fixture
```

This creates `fixtures/recorded_session/pg_dump`, `images/`, and `manifest.json`.
They are intentionally ignored because they are generated data and can be large.

Restore a generated fixture on another development machine:

```powershell
.\restore_fixture.ps1
```

The restore script starts the database container, replaces the fixture database, and replaces only
`DATA_DIR/frames`, `DATA_DIR/thumbs`, and `DATA_DIR/originals`.
