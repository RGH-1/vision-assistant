"""Create a portable recorded-session fixture from one local video (design §12)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Protocol
from urllib.parse import urlparse
from uuid import UUID

import cv2
import numpy as np
import psycopg

from server.common.db import open_ingest_connection
from server.ingest.embeddings import ClipImageEncoder
from server.ingest.store import FrameStore, ProcessedFrame
from server.settings import ROOT, load_env_file


class ImageEncoder(Protocol):
    def encode(self, frame: np.ndarray, timings_ms: dict[str, float]) -> np.ndarray: ...


class FixtureStore(Protocol):
    def store(self, frame: ProcessedFrame) -> int: ...


@dataclass(frozen=True)
class FixtureResult:
    session_id: str
    frame_count: int
    source_video: str


def process_video(
    video_path: Path,
    session_id: UUID,
    store: FixtureStore,
    encoder: ImageEncoder,
    *,
    sample_interval_s: float = 1.0,
) -> int:
    """Store one representative frame per interval from a recorded video."""
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)

    base_time = datetime.now(UTC)
    next_sample_s = 0.0
    stored = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            position_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if position_s + 1e-6 < next_sample_s:
                continue
            next_sample_s += sample_interval_s
            timings_ms: dict[str, float] = {}
            started_at = perf_counter()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            timings_ms["fixture"] = (perf_counter() - started_at) * 1000
            store.store(
                ProcessedFrame(
                    session_id=session_id,
                    captured_at=base_time + timedelta(seconds=position_s),
                    image=image,
                    sharpness=sharpness,
                    brightness=float(gray.mean()),
                    contrast=float(gray.std()),
                    edge_density=float((cv2.Canny(gray, 100, 200) > 0).mean()),
                    dhash=_dhash(gray),
                    keep_reason="heartbeat",
                    clip_emb=encoder.encode(image, timings_ms),
                    timings_ms=timings_ms,
                )
            )
            stored += 1
    finally:
        capture.release()
    return stored


def create_fixture(video_path: Path, output_dir: Path, sample_interval_s: float) -> FixtureResult:
    """Create an empty-database fixture so no unrelated local data is exported."""
    connection = open_ingest_connection()
    try:
        _require_empty_database(connection)
        session_id = connection.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES ('en', now(), 'fixture-v1')
            RETURNING id
            """
        ).fetchone()["id"]
        connection.commit()

        output_dir.mkdir(parents=True, exist_ok=True)
        store = FrameStore(output_dir / "images", connection)
        stored = process_video(
            video_path, session_id, store, ClipImageEncoder(), sample_interval_s=sample_interval_s
        )
        if stored == 0:
            raise ValueError("the video contains no readable frames")
        _dump_database(output_dir / "pg_dump")
        result = FixtureResult(str(session_id), stored, str(video_path))
        (output_dir / "manifest.json").write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        connection.close()


def _require_empty_database(connection: psycopg.Connection) -> None:
    row = connection.execute(
        "SELECT count(*) AS sessions, (SELECT count(*) FROM frames) AS frames FROM sessions"
    ).fetchone()
    if row["sessions"] or row["frames"]:
        raise RuntimeError("refusing to dump a non-empty database; use a clean fixture database")


def _dump_database(destination: Path) -> None:
    database_url = _required_env("DATABASE_URL")
    pg_dump = shutil.which("pg_dump")
    if pg_dump:
        command = [pg_dump, "--format=custom", "--file", str(destination), database_url]
        subprocess.run(command, check=True)
        return

    parsed = urlparse(database_url)
    user = parsed.username
    database = parsed.path.removeprefix("/")
    if not user or not database:
        raise RuntimeError("DATABASE_URL must include a user and database")
    container = _compose_output("ps", "-q", "db")
    with destination.open("wb") as dump_file:
        subprocess.run(
            ["docker", "exec", "-i", container, "pg_dump", "-U", user, "-Fc", database],
            check=True,
            stdout=dump_file,
        )


def _compose_output(*args: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(ROOT / ".env"),
            "-f",
            str(ROOT / "infra" / "docker-compose.yml"),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("PostgreSQL container is not running; start docker compose first")
    return output


def _required_env(name: str) -> str:
    import os

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value if value < 2**63 else value - 2**64


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable Vision Assistant fixture")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "videos" / "walk_01.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures" / "recorded_session")
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    result = create_fixture(args.video, args.output, args.sample_interval_s)
    print(f"Stored {result.frame_count} frames for session {result.session_id}")


if __name__ == "__main__":
    main()
