"""Live ingest checks: sampled frames reach Postgres and disk (design §6.3.1). Models are faked."""

import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from psycopg import Connection

from server.common.db import open_ingest_connection
from server.ingest.detection import Detection
from server.ingest.faceblur import FaceBlurResult
from server.ingest.pipeline import FramePipeline
from server.ingest.store import StoredOcr

KEYS = Detection("keys", 0.9, (0.4, 0.4, 0.6, 0.6), 0.04, "center")


class FakeDetector:
    def detect(self, frame: np.ndarray, timings_ms: dict[str, float]) -> tuple[Detection, ...]:
        return (KEYS,)


class FakeClip:
    def encode(self, frame: np.ndarray, timings_ms: dict[str, float]) -> np.ndarray:
        return np.eye(1, 512, dtype=np.float32)[0]


class FakeOcr:
    def read(self, _: np.ndarray, __: tuple, timings_ms: dict[str, float]) -> tuple[StoredOcr, ...]:
        timings_ms["ocr"] = 2.0
        return (StoredOcr("Room 204", 0.9, (0.1, 0.1, 0.2, 0.2)),)


class FakeFaceBlurrer:
    def blur(self, image: np.ndarray, _: dict[str, float] | None = None) -> FaceBlurResult:
        return FaceBlurResult(image.copy(), 0)


class FakeHub:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def noise(seed: int) -> np.ndarray:
    """A sharp, well-exposed frame; each seed gives a very different dHash."""
    return np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)


def sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


@pytest.fixture
def db() -> Iterator[Connection]:
    connection = open_ingest_connection()
    yield connection
    connection.close()


@pytest.fixture
def session_id(db: Connection) -> Iterator[str]:
    session_id = db.execute(
        "INSERT INTO sessions (app_lang, consent_at, consent_ver) "
        "VALUES ('en', now(), 'pipeline-test') RETURNING id"
    ).fetchone()["id"]
    db.commit()
    yield session_id
    db.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    db.commit()


@pytest.fixture
def run(tmp_path: Path, db: Connection) -> Iterator[tuple[FramePipeline, FakeHub]]:
    hub = FakeHub()
    connection = open_ingest_connection()  # its own, like the real ingest thread
    pipeline = FramePipeline(
        tmp_path,
        FakeFaceBlurrer(),
        hub,
        models=lambda: (FakeDetector(), FakeClip(), FakeOcr()),
        connect=lambda: connection,
    )
    yield pipeline, hub
    connection.close()


def start_of_a_window() -> int:
    return math.floor(time.monotonic())


def test_the_sharpest_frame_of_each_second_is_saved_with_its_objects(
    run: tuple[FramePipeline, FakeHub], db: Connection, session_id: str, tmp_path: Path
) -> None:
    pipeline, hub = run
    base = start_of_a_window()
    first, second, third = noise(1), noise(2), noise(3)

    pipeline.process(first, base + 0.0)
    pipeline.process(second, base + 0.2)  # same 1-second window as the first
    pipeline.process(third, base + 1.0)  # closes the first window: its winner is saved
    pipeline.process(noise(4), base + 2.0)  # closes the second window

    rows = db.execute(
        "SELECT id, captured_at, sharpness, keep_reason, image_path, thumb_path, clip_emb, "
        "timings_ms FROM frames WHERE session_id = %s ORDER BY captured_at",
        (session_id,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["sharpness"] == pytest.approx(max(sharpness(first), sharpness(second)))
    assert rows[1]["sharpness"] == pytest.approx(sharpness(third))
    assert {row["keep_reason"] for row in rows} == {"novel"}
    assert (time.time() - rows[0]["captured_at"].timestamp()) < 30  # a real time, not monotonic
    assert (tmp_path / rows[0]["image_path"]).is_file()
    assert (tmp_path / rows[0]["thumb_path"]).is_file()
    assert rows[0]["clip_emb"] is not None
    assert set(rows[0]["timings_ms"]) == {
        "quality",
        "enhance",
        "yoloe",
        "face_blur",
        "ocr",
        "clip",
        "store",
    }
    assert all(value >= 0 for value in rows[0]["timings_ms"].values())
    labels = db.execute(
        "SELECT label FROM detections WHERE frame_id = %s", (rows[0]["id"],)
    ).fetchall()
    assert [row["label"] for row in labels] == ["keys"]
    ocr = db.execute(
        "SELECT text, text_norm FROM ocr_texts WHERE frame_id = %s", (rows[0]["id"],)
    ).fetchall()
    assert ocr == [{"text": "Room 204", "text_norm": "room 204"}]
    assert [event["frame_id"] for event in hub.events] == [row["id"] for row in rows]


def test_the_same_view_is_saved_once_until_the_heartbeat(
    run: tuple[FramePipeline, FakeHub], db: Connection, session_id: str
) -> None:
    pipeline, _ = run
    base = start_of_a_window()
    still = noise(1)

    for second in range(4):  # the camera doesn't move for 4 seconds
        pipeline.process(still, base + second)

    count = db.execute(
        "SELECT count(*) AS n FROM frames WHERE session_id = %s", (session_id,)
    ).fetchone()["n"]
    assert count == 1


def test_nothing_is_saved_without_an_open_session(
    run: tuple[FramePipeline, FakeHub], db: Connection, session_id: str
) -> None:
    db.execute("UPDATE sessions SET ended_at = now() WHERE id = %s", (session_id,))
    db.commit()
    others = db.execute("SELECT count(*) AS n FROM sessions WHERE ended_at IS NULL").fetchone()
    if others["n"]:
        pytest.skip("another session is open in this database, so frames would go there")
    before = db.execute("SELECT count(*) AS n FROM frames").fetchone()["n"]
    pipeline, hub = run
    base = start_of_a_window()

    pipeline.process(noise(1), base)
    pipeline.process(noise(2), base + 1)

    assert db.execute("SELECT count(*) AS n FROM frames").fetchone()["n"] == before
    assert not hub.events
