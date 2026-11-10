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
    rtsp_url: str = "rtsp://localhost:8554/phone"
    sample_hz: float = 5.0
    ingest_no_frame_timeout_s: float = 3.0
    ingest_reconnect_delay_s: float = 0.25
    status_interval_s: float
    ws_ping_interval_s: float
    ws_send_timeout_s: float
    stt_model: str
    stt_timeout_s: float
    stt_min_avg_logprob: float
    gemini_model: str
    gemini_timeout_s: float
    gemini_temperature: float


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
    config.pop("frame_selection")  # loaded directly by server.ingest.selection
    config.pop("frame_enhancement")  # loaded directly by server.ingest.enhance
    config.pop("frame_novelty")  # loaded directly by server.ingest.novelty
    config.pop("object_detection")  # loaded directly by server.ingest.detection
    return Settings(
        data_dir=ROOT / os.environ["DATA_DIR"],
        retention_minutes=int(os.environ["RETENTION_MINUTES"]),
        rtsp_url=os.environ.get("RTSP_URL", "rtsp://localhost:8554/phone"),
        sample_hz=float(os.environ.get("SAMPLE_HZ", "5")),
        stt_model=os.environ["STT_MODEL"],
        gemini_model=os.environ["GEMINI_MODEL"],
        **config,
    )
