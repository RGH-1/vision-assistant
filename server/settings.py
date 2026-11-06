"""Settings from .env (machine-specific, secret) and config.yaml (tunable, in git). Design §8.3."""

import os
from pathlib import Path

# shortcut: PyYAML is only installed through ultralytics; declare it once §4 lists it.
import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config.yaml"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled config.yaml key fails at startup

    data_dir: Path
    retention_minutes: int
    host: str
    port: int
    mediamtx_webrtc_port: int
    cors_origins: list[str]
    clip_text_model: str
    ingest_stub_heartbeat_s: float


def load_env_file(path: Path) -> None:
    """Copy KEY=VALUE lines into os.environ, where server/common/db.py also reads them."""
    # shortcut: no quotes or inline comments, which .env.example doesn't use;
    # switch to python-dotenv (add it to §4 first) if a value ever needs them.
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())  # a real env var wins over .env


def load_settings() -> Settings:
    load_env_file(ROOT / ".env")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return Settings(
        data_dir=ROOT / os.environ["DATA_DIR"],
        retention_minutes=int(os.environ["RETENTION_MINUTES"]),
        **config,
    )
