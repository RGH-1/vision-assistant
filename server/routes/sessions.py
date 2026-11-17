"""Session routes: create, end and delete (design §7.5, §8.1)."""

import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from server.common.schemas import SessionCreate, SessionCreated
from server.ingest.retention import delete_frame_files

router = APIRouter(prefix="/v1/sessions")


@router.post("", status_code=201)
async def create_session(body: SessionCreate, request: Request) -> SessionCreated:
    async with request.app.state.pool.connection() as conn:
        # One transaction, so a failed insert can't leave the old session ended with no new one.
        await conn.execute("UPDATE sessions SET ended_at = now() WHERE ended_at IS NULL")
        cursor = await conn.execute(
            """
            INSERT INTO sessions (app_lang, consent_at, consent_ver)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (body.app_lang, body.consent_at, body.consent_ver),
        )
        session_id = (await cursor.fetchone())["id"]
    request.app.state.live_buffer.clear()

    # The address the phone used to reach us is the laptop's current IP, so no setting is needed.
    host = request.url.hostname
    settings = request.app.state.settings
    return SessionCreated(
        session_id=session_id,
        whip_url=f"http://{host}:{settings.mediamtx_webrtc_port}/phone/whip",
        ws_url=f"ws://{host}:{settings.port}/v1/ws/phone?session_id={session_id}",
    )


@router.post("/{session_id}/end", status_code=204)
async def end_session(session_id: UUID, request: Request) -> None:
    async with request.app.state.pool.connection() as conn:
        # COALESCE keeps the first end time if the session was already ended.
        cursor = await conn.execute(
            "UPDATE sessions SET ended_at = COALESCE(ended_at, now()) WHERE id = %s RETURNING id",
            (session_id,),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404)
    request.app.state.live_buffer.clear()


async def purge_session(conn: object, data_dir: Path, session_id: UUID) -> bool:
    """Remove one session's media before cascading its rows, returning whether it existed."""
    cursor = await conn.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
    if await cursor.fetchone() is None:
        return False
    cursor = await conn.execute(
        """
        SELECT image_path, thumb_path, original_path FROM frames WHERE session_id = %s
        """,
        (session_id,),
    )
    frame_files = await cursor.fetchall()
    # If this fails, the surrounding transaction leaves all rows for a safe retry.
    await asyncio.to_thread(delete_frame_files, data_dir, frame_files)
    await conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    return True


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID, request: Request) -> None:
    async with request.app.state.pool.connection() as conn:
        if not await purge_session(conn, request.app.state.settings.data_dir, session_id):
            raise HTTPException(status_code=404)
    request.app.state.live_buffer.clear()
