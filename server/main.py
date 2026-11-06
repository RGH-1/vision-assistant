"""FastAPI app: startup, routes and error format (design §5.3, §6.4, §8.1)."""

import asyncio
import logging
import threading
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


def run_ingest_stub(stop: threading.Event, heartbeat_s: float) -> None:
    # shortcut: stand-in for M2's server/ingest/worker.py, which replaces this once it exists.
    while not stop.wait(heartbeat_s):
        logger.info("ingest stub heartbeat")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.hub = EventHub(asyncio.get_running_loop())

    async with create_web_pool() as pool:
        app.state.pool = pool
        # In a thread so the pool's background tasks keep running while the model loads.
        app.state.clip_text = await asyncio.to_thread(
            load_clip_text_encoder, settings.clip_text_model
        )

        stop = threading.Event()
        ingest = threading.Thread(
            target=run_ingest_stub,
            args=(stop, settings.ingest_stub_heartbeat_s),
            name="ingest",
            daemon=True,
        )
        ingest.start()
        logger.info("Server ready")
        try:
            yield
        finally:
            stop.set()
            ingest.join()


@router.get("/v1/status")
async def get_status(request: Request) -> Status:
    # shortcut: placeholder values until ingest and the assistant report real ones (VA-61).
    return Status(
        session_id=None,
        stream_connected=False,
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

    # One mount per kind, not all of DATA_DIR, which also holds videos and model files.
    for kind in MEDIA_KINDS:
        directory = settings.data_dir / kind
        directory.mkdir(parents=True, exist_ok=True)
        app.mount(f"/media/{kind}", StaticFiles(directory=directory), name=kind)

    return app
