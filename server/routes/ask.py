"""Ask routes: a question by voice or text, answered from the live camera or memory (§5.4, §8.1)."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from psycopg.types.json import Jsonb

from server.assistant import gate, gemini, retrieve, stt
from server.assistant.understand import understand
from server.common.schemas import (
    AskEvidence,
    AskResponse,
    AskTextRequest,
    Confidence,
    Error,
    ErrorCode,
    ErrorDetail,
    Lang,
    QuestionEvidence,
    TimingsMs,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

CANT_SEE: dict[Lang, str] = {
    "en": "I can't see right now. The camera is not connected.",
    "ar": "لا أستطيع الرؤية الآن. الكاميرا غير متصلة.",
    "fr": "Je ne vois rien pour le moment. La caméra n'est pas connectée.",
}
WENT_WRONG: dict[Lang, str] = {
    "en": "Sorry, something went wrong. Please try again.",
    "ar": "عذرًا، حدث خطأ ما. من فضلك حاول مرة أخرى.",
    "fr": "Désolé, une erreur s'est produite. Veuillez réessayer.",
}
NO_FRENCH = "Le français n'est pas encore disponible. Choisissez l'anglais ou l'arabe."
NOW_INTENTS = ("LIVE_SCENE", "READ_TEXT_NOW")  # answered from the live camera


def ask_error(status_code: int, code: ErrorCode, message: str, speak: str) -> JSONResponse:
    body = Error(error=ErrorDetail(code=code, message=message, speak=speak))
    return JSONResponse(body.model_dump(), status_code=status_code)


def ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


async def check_session(request: Request, session_id: UUID) -> None:
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,))
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404)


async def answer_question(
    request: Request,
    *,
    session_id: UUID,
    transcript: str,
    lang: Lang,
    input_mode: Literal["voice", "text"],
    stt_engine: Literal["groq", "local"] | None,
    asked_at: datetime,
    started: float,
    timings: TimingsMs,
) -> AskResponse | JSONResponse:
    app = request.app
    understand_started = time.perf_counter()
    try:
        question = understand(transcript, lang, asked_at, app.state.settings)
    except FileNotFoundError:
        # shortcut: French has no lexicon until VA-78, and without one identity questions
        # can't be refused in code, so French isn't answered at all.
        return ask_error(400, "BAD_REQUEST", "French is not supported yet.", NO_FRENCH)
    timings["understand"] = ms_since(understand_started)

    frame = app.state.live_buffer.get_sharpest() if app.state.stream_connected else None
    evidence: list[AskEvidence] = []
    stored: list[QuestionEvidence] = []  # evidence frames saved in Postgres
    confidence: Confidence
    if question.intent in gate.FIXED_REPLIES:
        answer, confidence, model = gate.FIXED_REPLIES[question.intent][lang], "none", "rule"
    elif question.intent in NOW_INTENTS and frame is None:
        return ask_error(503, "NO_STREAM", "The camera stream is not connected.", CANT_SEE[lang])
    elif question.intent in NOW_INTENTS or (question.use_live_frame and frame is not None):
        # shortcut: READ_TEXT_NOW and the rule-7 fallback use the sharpest live frame only;
        # sign-box frames (VA-72) and stored frames (VA-70) are added later.
        captured_at, jpeg, _, detections = frame
        seconds_ago = time.monotonic() - captured_at  # the ingest thread uses time.monotonic()
        gemini_started = time.perf_counter()
        try:
            reply = await gemini.describe_live_scene(
                jpeg, seconds_ago, detections, transcript, lang, app.state.settings
            )
        except Exception:
            # Whatever failed (quota, timeout, bad JSON), the phone still needs something to say.
            logger.exception("Gemini failed")
            return ask_error(502, "GEMINI_FAILED", "Gemini did not answer.", WENT_WRONG[lang])
        timings["gemini"] = ms_since(gemini_started)
        answer, confidence, model = reply.answer, reply.confidence, app.state.settings.gemini_model
        # shortcut: a single live frame has nothing to be ranked against; VA-70 adds scores.
        evidence = [
            AskEvidence(frame_id=None, seconds_ago=round(seconds_ago, 1), thumb_url=None, score=1)
        ]
    else:
        retrieve_started = time.perf_counter()
        # In a thread, because the CPU encoder would otherwise block every other request.
        clip_vector = await asyncio.to_thread(app.state.clip_text.encode, transcript)
        async with app.state.pool.connection() as conn:
            ranked = await retrieve.search_memory(conn, session_id, question, clip_vector, asked_at)
        timings["retrieve"] = ms_since(retrieve_started)
        stored = [
            QuestionEvidence(
                frame_id=hit.frame_id,
                rank=rank,
                score=round(hit.score, 3),
                parts=hit.parts,
                thumb_url=hit.thumb_url,
            )
            for rank, hit in enumerate(ranked, start=1)
        ]
        evidence = [
            AskEvidence(
                frame_id=hit.frame_id,
                seconds_ago=round(hit.seconds_ago, 1),
                thumb_url=hit.thumb_url,
                score=round(hit.score, 3),
            )
            for hit in ranked
        ]
        # shortcut: stored frames have no answer generator yet, so even good evidence gets
        # "I'm not sure"; VA-72 sends frames scoring above EVIDENCE_MIN to Gemini (§6.5.5).
        answer, confidence, model = gate.NOT_SURE[lang], "none", "template"

    timings["total"] = ms_since(started)
    # shortcut: only the prompt keeps "safe" out of answers until the safety filter (VA-73).
    abstained = confidence in ("low", "none")

    async with app.state.pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO questions (session_id, asked_at, input_mode, stt_engine, transcript, lang,
                                   intent, t_from, t_to, answer, confidence, abstained, model,
                                   timings_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                asked_at,
                input_mode,
                stt_engine,
                transcript,
                lang,
                question.intent,
                question.t_from,
                question.t_to,
                answer,
                confidence,
                abstained,
                model,
                Jsonb(timings),
            ),
        )
        question_id = (await cursor.fetchone())["id"]
        for item in stored:
            await conn.execute(
                """
                INSERT INTO question_evidence (question_id, frame_id, rank, score, parts)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (question_id, item.frame_id, item.rank, item.score, Jsonb(item.parts.model_dump())),
            )

    # Out here so the row is committed before the dashboard hears about it.
    app.state.hub.publish(
        {
            "type": "query_answered",
            "question_id": question_id,
            "session_id": session_id,
            "asked_at": asked_at,
            "input_mode": input_mode,
            "stt_engine": stt_engine,
            "transcript": transcript,
            "lang": lang,
            "intent": question.intent,
            "t_from": question.t_from,
            "t_to": question.t_to,
            "answer": answer,
            "confidence": confidence,
            "abstained": abstained,
            "safety_rewrite": False,  # nothing rewrites answers until the safety filter (VA-73)
            "model": model,
            "timings_ms": timings,
            # A live frame isn't stored, so only stored frames are evidence here.
            "evidence": [item.model_dump() for item in stored],
        }
    )

    return AskResponse(
        question_id=question_id,
        transcript=transcript,
        lang=lang,
        intent=question.intent,
        answer=answer,
        confidence=confidence,
        abstained=abstained,
        evidence=evidence,
        timings_ms=timings,
    )


@router.post("/ask_text", response_model=AskResponse)
async def ask_text(body: AskTextRequest, request: Request) -> AskResponse | JSONResponse:
    asked_at, started = datetime.now(UTC), time.perf_counter()
    await check_session(request, body.session_id)
    return await answer_question(
        request,
        session_id=body.session_id,
        transcript=body.text,
        lang=body.lang,
        input_mode="text",
        stt_engine=None,
        asked_at=asked_at,
        started=started,
        timings={},
    )


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    audio: Annotated[UploadFile, File()],
    session_id: Annotated[UUID, Form()],
    lang: Annotated[Lang, Form()],
) -> AskResponse | JSONResponse:
    asked_at, started = datetime.now(UTC), time.perf_counter()
    await check_session(request, session_id)
    try:
        # Read straight into transcribe, so the audio is never kept (§6.5.1, §9).
        # shortcut: Starlette keeps uploads under 1 MB in memory (10 s of audio is 320 KB), but
        # spools a larger, invalid one to a temp file until the request ends.
        transcript = await stt.transcribe(await audio.read(), lang, request.app.state.settings)
    except ValueError as error:
        return ask_error(400, "BAD_AUDIO", str(error), stt.DIDNT_CATCH[lang])
    except Exception:
        # Groq and local Whisper both failed, and the phone must still have something to say.
        logger.exception("Speech to text failed")
        return ask_error(502, "STT_FAILED", "Speech to text failed.", WENT_WRONG[lang])
    if not transcript.text:
        return ask_error(400, "BAD_AUDIO", "No speech was understood.", stt.DIDNT_CATCH[lang])

    return await answer_question(
        request,
        session_id=session_id,
        transcript=transcript.text,
        lang=lang,
        input_mode="voice",
        stt_engine=transcript.engine,
        asked_at=asked_at,
        started=started,
        timings={"stt": round(transcript.time_ms, 1)},
    )
