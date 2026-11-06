"""VA-59 checks. The database and CLIP model are stubbed; the real startup is `python -m server`."""

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import main
from server.common.schemas import Error, Status
from server.events import EventHub
from server.ingest.live_buffer import LiveBuffer
from server.main import create_app
from server.settings import ROOT, Settings, load_env_file


class FakePool:
    async def __aenter__(self) -> "FakePool":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeIngestWorker:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        retention_minutes=30,
        host="127.0.0.1",
        port=8000,
        cors_origins=["http://localhost:5173"],
        clip_text_model="stub",
        rtsp_url="rtsp://example.invalid/phone",
        sample_hz=5,
        ingest_no_frame_timeout_s=3,
        ingest_reconnect_delay_s=0.01,
    )


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "create_web_pool", FakePool)
    monkeypatch.setattr(main, "load_clip_text_encoder", lambda _: "clip-stub")
    created_workers: list[FakeIngestWorker] = []

    def create_worker(_: object) -> "FakeIngestWorker":
        worker = FakeIngestWorker()
        created_workers.append(worker)
        return worker

    monkeypatch.setattr(main, "create_ingest_worker", create_worker)
    app = create_app(settings)
    app.state.created_workers = created_workers

    @app.get("/test/needs-int")
    async def needs_int(number: int) -> int:
        return number

    @app.get("/test/crash")
    async def crash() -> None:
        raise RuntimeError("secret detail")

    # Without raise_server_exceptions=False the client re-raises instead of returning the 500.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def assert_error(response, status_code: int, code: str) -> Error:
    assert response.status_code == status_code
    body = Error.model_validate(response.json())
    assert body.error.code == code
    assert body.error.speak  # the phone always has something to say
    return body


def test_status_has_exactly_the_contract_fields(client: TestClient) -> None:
    response = client.get("/v1/status")

    assert response.status_code == 200
    schema = json.loads((ROOT / "contracts" / "events.schema.json").read_text(encoding="utf-8"))
    assert set(response.json()) == set(schema["$defs"]["Status"]["required"])
    assert set(Status.model_fields) == set(schema["$defs"]["Status"]["required"])
    assert response.json()["retention_min"] == 30


def test_startup_loads_everything_and_shutdown_stops_ingest(client: TestClient) -> None:
    state = client.app.state
    assert state.clip_text == "clip-stub"
    assert isinstance(state.pool, FakePool)
    assert isinstance(state.hub, EventHub)
    assert isinstance(state.live_buffer, LiveBuffer)
    assert state.created_workers[0].started

    client.__exit__(None, None, None)
    assert state.created_workers[0].stopped


def test_hub_delivers_on_the_loop_thread_when_published_from_another_thread() -> None:
    async def run() -> tuple[dict, int, int]:
        received: asyncio.Future = asyncio.get_running_loop().create_future()
        hub = EventHub(asyncio.get_running_loop())
        hub._deliver = lambda event: received.set_result((event, threading.get_ident()))
        publisher = threading.Thread(target=hub.publish, args=({"type": "status"},))
        publisher.start()
        event, delivery_thread = await asyncio.wait_for(received, timeout=2)
        publisher.join()
        return event, delivery_thread, threading.get_ident()

    event, delivery_thread, loop_thread = asyncio.run(run())

    assert event == {"type": "status"}
    assert delivery_thread == loop_thread


def test_cors_allows_the_dashboard_dev_server_only(client: TestClient) -> None:
    allowed = client.get("/v1/status", headers={"Origin": "http://localhost:5173"})
    other = client.get("/v1/status", headers={"Origin": "http://example.com"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers


def test_media_serves_only_the_contract_folders(client: TestClient, settings: Settings) -> None:
    (settings.data_dir / "frames" / "a.jpg").write_bytes(b"jpeg")
    (settings.data_dir / "videos").mkdir()
    (settings.data_dir / "videos" / "walk.mp4").write_bytes(b"video")
    (settings.data_dir / "secret.txt").write_text("secret")

    assert client.get("/media/frames/a.jpg").content == b"jpeg"
    assert_error(client.get("/media/frames/missing.jpg"), 404, "NOT_FOUND")
    assert_error(client.get("/media/videos/walk.mp4"), 404, "NOT_FOUND")
    assert_error(client.get("/media/frames/..%2Fsecret.txt"), 404, "NOT_FOUND")


def test_errors_use_the_contract_format(client: TestClient) -> None:
    assert_error(client.get("/no/such/route"), 404, "NOT_FOUND")
    assert_error(client.post("/v1/status"), 400, "BAD_REQUEST")

    rejected = assert_error(client.get("/test/needs-int?number=abc"), 400, "BAD_REQUEST")
    assert "number" in rejected.error.message
    assert "abc" not in rejected.error.message

    crashed = assert_error(client.get("/test/crash"), 500, "INTERNAL")
    assert "secret detail" not in crashed.error.message


def test_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("VA_TEST_URL", "VA_TEST_EMPTY", "VA_TEST_REAL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VA_TEST_REAL", "from environment")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n\nVA_TEST_URL=postgresql://a:b@host/db?x=1\nVA_TEST_EMPTY=\n"
        "VA_TEST_REAL=from file\nnot a setting\n"
    )

    load_env_file(env_file)
    load_env_file(tmp_path / "missing.env")

    assert os.environ["VA_TEST_URL"] == "postgresql://a:b@host/db?x=1"  # keeps the second "="
    assert os.environ["VA_TEST_EMPTY"] == ""
    assert os.environ["VA_TEST_REAL"] == "from environment"
