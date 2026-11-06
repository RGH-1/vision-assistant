"""FastAPI app: startup, routes and error format (design §5.3, §6.4, §8.1)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from server.common.db import create_web_pool
from server.common.schemas import Error, ErrorCode, ErrorDetail, Status
from server.events import EventHub

from server.ingest.live_buffer import LiveBuffer
from server.ingest.worker import IngestSource, IngestWorker

from server.routes import sessions
from server.settings import Settings

logger = logging.getLogger(__name__)

MEDIA_KINDS = ("frames", "thumbs", "originals")
# shortcut: English only, because these handlers don't know the user's language; VA-64 can localize.
SPEAK_ERROR = "Sorry, something went wrong. Please try again."

router = APIRouter()


def load_clip_text_encoder(model_name: str) -> Any:
    # Imported here because it pulls in torch, which takes long and isn't needed by the tests.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")  # the GPU belongs to ingest (§5.3)


def create_ingest_worker(app: FastAPI) -> IngestWorker:
    """Build the worker after the asyncio hub and live buffer exist (§5.3)."""
    settings: Settings = app.state.settings

    def connection_event(connected: bool) -> dict[str, Any]:
        app.state.stream_connected = connected
        return {
            "type": "status",
            "session_id": None,
            "stream_connected": connected,
            "input_fps": 0.0,
            "analysed_fps": 0.0,
            "saved_per_min": 0.0,
            "frames_in_db": 0,
            "retention_min": settings.retention_minutes,
            "gpu_mem_mb": None,
            "last_answer_at": None,
            "groq_ok": False,
            "gemini_ok": False,
        }

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
    app.state.hub = EventHub(asyncio.get_running_loop())
    app.state.live_buffer = LiveBuffer()
    app.state.stream_connected = False

    async with create_web_pool() as pool:
        app.state.pool = pool
        # In a thread so the pool's background tasks keep running while the model loads.
        app.state.clip_text = await asyncio.to_thread(
            load_clip_text_encoder, settings.clip_text_model
        )

        ingest = create_ingest_worker(app)
        app.state.ingest = ingest
        ingest.start()
        logger.info("Server ready")
        try:
            yield
        finally:
            ingest.stop()


@router.get("/v1/status")
async def get_status(request: Request) -> Status:
    # shortcut: placeholder values until ingest and the assistant report real ones (VA-61).
    return Status(
        session_id=None,
        stream_connected=request.app.state.stream_connected,
        input_fps=0,
        analysed_fps=0,
        saved_per_min=0,
        frames_in_db=0,
        retention_min=request.app.state.settings.retention_minutes,
        gpu_mem_mb=None,
        last_answer_at=None,
        groq_ok=False,
        gemini_ok=False,
    )


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    body = Error(error=ErrorDetail(code=code, message=message, speak=SPEAK_ERROR))
    return JSONResponse(body.model_dump(), status_code=status_code)


async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Only field names and reasons, so the rejected input is never echoed back or logged.
    problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return error_response(400, "BAD_REQUEST", "; ".join(problems))


async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 404:
        return error_response(404, "NOT_FOUND", "Not found.")
    if exc.status_code < 500:
        return error_response(400, "BAD_REQUEST", str(exc.detail))
    return error_response(500, "INTERNAL", "Unexpected server error.")


async def handle_uncaught_error(_: Request, __: Exception) -> JSONResponse:
    # Starlette re-raises the exception after this response, so uvicorn logs the traceback.
    # shortcut: runs outside CORSMiddleware, so a browser on the Vite dev server sees a CORS error
    # instead of this body; the phone and the built dashboard aren't affected.
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

    # One mount per kind, not all of DATA_DIR, which also holds videos and model files.
    for kind in MEDIA_KINDS:
        directory = settings.data_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(f"/media/{kind}", StaticFiles(directory=directory), name=kind)

    return app
