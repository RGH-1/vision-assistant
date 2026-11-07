"""VA-59 and VA-61 checks. Database and CLIP are stubbed; the real startup is `python -m server`."""

import asyncio
import json
import os
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server import main
from server.common.schemas import Error, Status
from server.events import MESSAGES, EventHub
from server.ingest.live_buffer import LiveBuffer
from server.main import create_app
from server.settings import ROOT, Settings, load_env_file


class FakePool:
    """Stands in for the pool, its connections and their cursors, to keep the stub short."""

    async def __aenter__(self) -> "FakePool":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def connection(self) -> "FakePool":
        return self

    async def execute(self, _: str) -> "FakePool":
        return self

    async def fetchone(self) -> dict:
        return {"session_id": None, "frames_in_db": 0, "last_answer_at": None}


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
        mediamtx_webrtc_port=8889,
        cors_origins=["http://localhost:5173"],
        clip_text_model="stub",
        status_interval_s=3600,  # out of the way; the test that needs them sets its own
        ws_ping_interval_s=3600,
        ws_send_timeout_s=0.05,
        stt_model="stub",
        stt_timeout_s=1,
        stt_min_avg_logprob=-1.0,
        stt_max_no_speech_prob=0.6,
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


PHONE_URL = "/v1/ws/phone?session_id=3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01"
DASHBOARD_URL = "/v1/ws/dashboard"


def receive(socket) -> dict:
    """Like socket.receive_json(), but a message that never comes fails instead of hanging."""
    received: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: received.put(socket.receive_json()), daemon=True).start()
    return received.get(timeout=5)


def ping(socket) -> None:
    """A pong also proves the server has finished registering this client."""
    socket.send_json({"type": "ping"})
    assert receive(socket) == {"type": "pong"}


def test_events_published_from_a_thread_reach_the_right_clients(client: TestClient) -> None:
    examples = json.loads((ROOT / "contracts" / "events.examples.json").read_text(encoding="utf-8"))
    by_type = {example["type"]: example for example in examples}
    events = [by_type[kind] for kind in ("frame_stored", "query_answered", "alert", "status")]

    with (
        client.websocket_connect(PHONE_URL) as phone,
        client.websocket_connect(DASHBOARD_URL) as dashboard,
    ):
        ping(phone)
        ping(dashboard)

        def publish_all() -> None:
            for event in events:
                client.app.state.hub.publish(event)

        publisher = threading.Thread(target=publish_all)
        publisher.start()
        publisher.join()

        assert [receive(dashboard) for _ in events] == events
        assert [receive(phone) for _ in range(2)] == [by_type["alert"], by_type["status"]]


@pytest.fixture
def short_intervals(settings: Settings) -> None:
    settings.status_interval_s = 0.01
    settings.ws_ping_interval_s = 0.01


def test_status_and_ping_are_sent_on_their_intervals(
    short_intervals: None, client: TestClient
) -> None:
    with client.websocket_connect(PHONE_URL) as phone:
        kinds = {receive(phone)["type"] for _ in range(20)}

    assert kinds == {"status", "ping"}


def test_invalid_messages_are_ignored_and_the_client_stays_connected(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with client.websocket_connect(DASHBOARD_URL) as dashboard:
        dashboard.send_text("not json, private text")
        dashboard.send_bytes(b"binary")
        dashboard.send_json({"type": "made_up"})
        dashboard.send_json({"type": "hello", "session_id": "not-a-uuid", "app_lang": "en"})
        ping(dashboard)

        client.app.state.hub.publish({"type": "alert", "message": "private text"})
        client.app.state.hub.publish({"type": "ping"})
        assert receive(dashboard) == {"type": "ping"}  # the invalid alert was never sent

    assert "private text" not in caplog.text


def test_hello_is_accepted(client: TestClient) -> None:
    with client.websocket_connect(PHONE_URL) as phone:
        phone.send_json({"type": "hello", "session_id": PHONE_URL.split("=")[1], "app_lang": "ar"})
        ping(phone)


@pytest.mark.parametrize("url", ["/v1/ws/phone", "/v1/ws/phone?session_id=not-a-uuid"])
def test_phone_needs_a_valid_session_id(client: TestClient, url: str) -> None:
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(url):
        pass


def test_clients_are_removed_when_they_disconnect(client: TestClient) -> None:
    hub = client.app.state.hub
    with client.websocket_connect(DASHBOARD_URL) as dashboard:
        ping(dashboard)
        assert len(hub.dashboards) == 1

    deadline = time.monotonic() + 2
    while hub.dashboards and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not hub.dashboards


class FakeClient:
    def __init__(self, behaviour: str = "healthy") -> None:
        self.behaviour = behaviour
        self.received: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send_json(self, body: dict) -> None:
        if self.behaviour == "stalled":
            await asyncio.Event().wait()
        if self.behaviour == "broken":
            raise RuntimeError("connection lost")
        self.received.put_nowait(body)

    async def close(self) -> None:
        self.closed = True


def test_stalled_and_broken_clients_are_dropped_and_others_still_receive() -> None:
    async def run() -> None:
        hub = EventHub(asyncio.get_running_loop(), send_timeout_s=0.01)
        healthy, stalled, broken = FakeClient(), FakeClient("stalled"), FakeClient("broken")
        hub.dashboards.update([healthy, stalled])
        hub.phones.add(broken)
        broadcast = asyncio.create_task(hub.run())

        for _ in range(2):  # events go one at a time, so the second means the first has finished
            hub.publish({"type": "ping"})
            await asyncio.wait_for(healthy.received.get(), timeout=2)
        broadcast.cancel()

        assert hub.dashboards == {healthy}
        assert not hub.phones
        assert stalled.closed and broken.closed and not healthy.closed

    asyncio.run(run())


def test_connection_events_change_only_stream_connected(settings: Settings) -> None:
    app = create_app(settings)
    app.state.hub = None
    app.state.live_buffer = LiveBuffer()
    app.state.pool = FakePool()
    app.state.stream_connected = False
    status = asyncio.run(main.read_status(app))
    app.state.last_status = status.model_copy(update={"frames_in_db": 7})

    event = main.create_ingest_worker(app)._connection_event_factory(True)

    message = MESSAGES.validate_python(event)  # the hub drops events that fail this
    assert message.stream_connected is True
    assert message.frames_in_db == 7
    assert asyncio.run(main.read_status(app)).stream_connected is True
