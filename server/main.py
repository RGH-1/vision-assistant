# """FastAPI app: startup, routes and error format (design §5.3, §6.4, §8.1)."""

# import asyncio
# import logging
# from collections.abc import AsyncIterator
# from contextlib import asynccontextmanager
# from typing import Any

# from fastapi import APIRouter, FastAPI, Request
# from fastapi.exceptions import RequestValidationError
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from starlette.exceptions import HTTPException

# from server.common.db import create_web_pool
# from server.common.embeddings import ClipTextEncoder
# from server.common.schemas import Error, ErrorCode, ErrorDetail, Status
# from server.events import EventHub
# from server.ingest.live_buffer import LiveBuffer
# from server.ingest.worker import IngestSource, IngestWorker
# from server.routes import sessions, ws
# from server.settings import Settings

# logger = logging.getLogger(__name__)

# MEDIA_KINDS = ("frames", "thumbs", "originals")
# # shortcut: English only, because these handlers don't know the user's language; VA-64 can localize.
# SPEAK_ERROR = "Sorry, something went wrong. Please try again."

# router = APIRouter()


# def load_clip_text_encoder(model_name: str) -> Any:
#     return ClipTextEncoder(model_name)  # the GPU belongs to ingest (§5.3)


# async def read_status(app: FastAPI) -> Status:
#     async with app.state.pool.connection() as conn:
#         cursor = await conn.execute(
#             """
#             SELECT
#                 (SELECT id FROM sessions WHERE ended_at IS NULL
#                  ORDER BY started_at DESC LIMIT 1) AS session_id,
#                 (SELECT count(*) FROM frames) AS frames_in_db,
#                 (SELECT max(asked_at) FROM questions) AS last_answer_at
#             """
#         )
#         row = await cursor.fetchone()

#     # shortcut: fps and GPU numbers stay empty until M2's ingest reports them, and groq_ok and
#     # gemini_ok stay false until VA-63 and VA-64 set them.
#     return Status(
#         session_id=row["session_id"],
#         stream_connected=app.state.stream_connected,
#         input_fps=0,
#         analysed_fps=0,
#         saved_per_min=0,
#         frames_in_db=row["frames_in_db"],
#         retention_min=app.state.settings.retention_minutes,
#         gpu_mem_mb=None,
#         last_answer_at=row["last_answer_at"],
#         groq_ok=False,
#         gemini_ok=False,
#     )


# async def publish_status(app: FastAPI) -> None:
#     while True:
#         await asyncio.sleep(app.state.settings.status_interval_s)
#         try:
#             status = await read_status(app)
#         except Exception:
#             # Logged and retried, so updates resume by themselves when the database is back.
#             logger.exception("Could not read status")
#             continue
#         app.state.last_status = status
#         app.state.hub.publish({"type": "status", **status.model_dump()})


# async def publish_pings(hub: EventHub, interval_s: float) -> None:
#     while True:
#         await asyncio.sleep(interval_s)
#         hub.publish({"type": "ping"})


# def create_ingest_worker(app: FastAPI) -> IngestWorker:
#     """Build the worker after the asyncio hub and live buffer exist (§5.3)."""
#     settings: Settings = app.state.settings

#     def connection_event(connected: bool) -> dict[str, Any]:
#         app.state.stream_connected = connected
#         # Runs on the ingest thread, which can't use the web pool, so it reuses the last numbers
#         # instead of blanking them on the dashboard. §6.3.1 wants this event right away.
#         status = app.state.last_status.model_dump()
#         return {"type": "status", **status, "stream_connected": connected}

#     return IngestWorker(
#         IngestSource(settings.rtsp_url),
#         app.state.live_buffer.add,
#         app.state.hub,
#         connection_event_factory=connection_event,
#         sample_hz=settings.sample_hz,
#         no_frame_timeout_s=settings.ingest_no_frame_timeout_s,
#         reconnect_delay_s=settings.ingest_reconnect_delay_s,
#     )


# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncIterator[None]:
#     settings: Settings = app.state.settings
#     hub = EventHub(asyncio.get_running_loop(), settings.ws_send_timeout_s)
#     app.state.hub = hub
#     app.state.live_buffer = LiveBuffer()
#     app.state.stream_connected = False

#     async with create_web_pool() as pool:
#         app.state.pool = pool
#         # In a thread so the pool's background tasks keep running while the model loads.
#         app.state.clip_text = await asyncio.to_thread(
#             load_clip_text_encoder, settings.clip_text_model
#         )

#         app.state.last_status = await read_status(app)
#         ingest = create_ingest_worker(app)
#         app.state.ingest = ingest
#         ingest.start()
#         background = [
#             asyncio.create_task(hub.run()),
#             asyncio.create_task(publish_status(app)),
#             asyncio.create_task(publish_pings(hub, settings.ws_ping_interval_s)),
#         ]
#         logger.info("Server ready")
#         try:
#             yield
#         finally:
#             ingest.stop()
#             for task in background:
#                 task.cancel()
#             await asyncio.gather(*background, return_exceptions=True)


# @router.get("/v1/status")
# async def get_status(request: Request) -> Status:
#     return await read_status(request.app)


# def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
#     body = Error(error=ErrorDetail(code=code, message=message, speak=SPEAK_ERROR))
#     return JSONResponse(body.model_dump(), status_code=status_code)


# async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
#     # Only field names and reasons, so the rejected input is never echoed back or logged.
#     problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
#     return error_response(400, "BAD_REQUEST", "; ".join(problems))


# async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
#     if exc.status_code == 404:
#         return error_response(404, "NOT_FOUND", "Not found.")
#     if exc.status_code < 500:
#         return error_response(400, "BAD_REQUEST", str(exc.detail))
#     return error_response(500, "INTERNAL", "Unexpected server error.")


# async def handle_uncaught_error(_: Request, __: Exception) -> JSONResponse:
#     # Starlette re-raises the exception after this response, so uvicorn logs the traceback.
#     # shortcut: runs outside CORSMiddleware, so a browser on the Vite dev server sees a CORS error
#     # instead of this body; the phone and the built dashboard aren't affected.
#     return error_response(500, "INTERNAL", "Unexpected server error.")


# def create_app(settings: Settings) -> FastAPI:
#     app = FastAPI(title="Vision Assistant", lifespan=lifespan)
#     app.state.settings = settings

#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=settings.cors_origins,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )
#     app.add_exception_handler(RequestValidationError, handle_validation_error)
#     app.add_exception_handler(HTTPException, handle_http_error)
#     app.add_exception_handler(Exception, handle_uncaught_error)
#     app.include_router(router)
#     app.include_router(sessions.router)
#     app.include_router(ws.router)

#     # One mount per kind, not all of DATA_DIR, which also holds videos and model files.
#     for kind in MEDIA_KINDS:
#         directory = settings.data_dir / kind
#         directory.mkdir(parents=True, exist_ok=True)
#         app.mount(f"/media/{kind}", StaticFiles(directory=directory), name=kind)

#     return app



"""FastAPI app: startup, routes and error format (design §5.3, §6.4, §8.1)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from server.common.db import create_web_pool
from server.common.embeddings import ClipTextEncoder
from server.common.schemas import Error, ErrorCode, ErrorDetail, Status
from server.events import EventHub
from server.ingest.live_buffer import LiveBuffer
from server.ingest.worker import IngestSource, IngestWorker
from server.routes import sessions, ws
from server.settings import Settings

logger = logging.getLogger(__name__)

MEDIA_KINDS = ("frames", "thumbs", "originals")
SPEAK_ERROR = "Sorry, something went wrong. Please try again."

router = APIRouter()


def load_clip_text_encoder(model_name: str) -> Any:
    return ClipTextEncoder(model_name)


async def read_status(app: FastAPI) -> Status:
    async with app.state.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT
                (SELECT id FROM sessions WHERE ended_at IS NULL
                 ORDER BY started_at DESC LIMIT 1) AS session_id,
                (SELECT count(*) FROM frames) AS frames_in_db,
                (SELECT max(asked_at) FROM questions) AS last_answer_at
            """
        )
        row = await cursor.fetchone()

    return Status(
        session_id=row["session_id"],
        stream_connected=app.state.stream_connected,
        input_fps=0,
        analysed_fps=0,
        saved_per_min=0,
        frames_in_db=row["frames_in_db"],
        retention_min=app.state.settings.retention_minutes,
        gpu_mem_mb=None,
        last_answer_at=row["last_answer_at"],
        groq_ok=False,
        gemini_ok=False,
    )


async def publish_status(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(app.state.settings.status_interval_s)
        try:
            status = await read_status(app)
        except Exception:
            logger.exception("Could not read status")
            continue
        app.state.last_status = status
        app.state.hub.publish({"type": "status", **status.model_dump()})


async def publish_pings(hub: EventHub, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        hub.publish({"type": "ping"})


def create_ingest_worker(app: FastAPI) -> IngestWorker:
    settings: Settings = app.state.settings

    def connection_event(connected: bool) -> dict[str, Any]:
        app.state.stream_connected = connected
        status = app.state.last_status.model_dump()
        return {"type": "status", **status, "stream_connected": connected}

    return IngestWorker(
        IngestSource(settings.rtsp_url),
        app.state.live_buffer.add,
        app.state.hub,
        connection_event_factory=connection_event,
        sample_hz=settings.sample_hz,
        no_frame_timeout_s=settings.ingest_no_frame_timeout_s,
        reconnect_delay_s=settings.ingest_reconnect_delay_s,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    hub = EventHub(asyncio.get_running_loop(), settings.ws_send_timeout_s)
    app.state.hub = hub
    app.state.live_buffer = LiveBuffer()
    app.state.stream_connected = False

    async with create_web_pool() as pool:
        app.state.pool = pool
        app.state.clip_text = await asyncio.to_thread(
            load_clip_text_encoder, settings.clip_text_model
        )

        app.state.last_status = await read_status(app)
        ingest = create_ingest_worker(app)
        app.state.ingest = ingest
        ingest.start()
        background = [
            asyncio.create_task(hub.run()),
            asyncio.create_task(publish_status(app)),
            asyncio.create_task(publish_pings(hub, settings.ws_ping_interval_s)),
        ]
        logger.info("Server ready")
        try:
            yield
        finally:
            ingest.stop()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)


@router.get("/v1/status")
async def get_status(request: Request) -> Status:
    return await read_status(request.app)


@router.post("/whip/{stream_name}")
async def proxy_whip(stream_name: str, request: Request) -> Response:
    """Proxy WHIP SDP offer from the mobile client to MediaMTX internally."""
    sdp_body = await request.body()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:8889/{stream_name}/whip",
                content=sdp_body,
                headers={"Content-Type": "application/sdp"},
                timeout=15.0,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type="application/sdp",
            )
        except Exception as e:
            logger.error(f"Failed to proxy WHIP request to MediaMTX: {e}")
            return Response(content=str(e), status_code=502)


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    body = Error(error=ErrorDetail(code=code, message=message, speak=SPEAK_ERROR))
    return JSONResponse(body.model_dump(), status_code=status_code)


async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return error_response(400, "BAD_REQUEST", "; ".join(problems))


async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 404:
        return error_response(404, "NOT_FOUND", "Not found.")
    if exc.status_code < 500:
        return error_response(400, "BAD_REQUEST", str(exc.detail))
    return error_response(500, "INTERNAL", "Unexpected server error.")


async def handle_uncaught_error(_: Request, __: Exception) -> JSONResponse:
    return error_response(500, "INTERNAL", "Unexpected server error.")


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Vision Assistant", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_uncaught_error)
    app.include_router(router)
    app.include_router(sessions.router)
    app.include_router(ws.router)

    for kind in MEDIA_KINDS:
        directory = settings.data_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(f"/media/{kind}", StaticFiles(directory=directory), name=kind)

    return app