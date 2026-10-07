"""WebSocket endpoint: authentication, admission control and the receive loop.

Security controls
-----------------
* Bearer token via ``Sec-WebSocket-Protocol`` (never in the URL).
* Origin allow-list (prevents cross-site WebSocket hijacking from other web pages).
* Per-process session cap (``VOICE__MAX_CONCURRENT_SESSIONS``) -> close 1013.
* Frame size limits, a start-message deadline and an idle timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING

from fastapi import APIRouter, Query, WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from maplo_voice.db.models import SessionMode, SessionStatus
from maplo_voice.observability import get_logger
from maplo_voice.services.openai_client import get_ai_services
from maplo_voice.voice.protocol import (
    BEARER_PREFIX,
    MAX_AUDIO_FRAME_BYTES,
    MAX_TEXT_FRAME_BYTES,
    PROTOCOL_VERSION,
    CloseCode,
    ErrorEvent,
    ServerEvent,
    SessionEnd,
    SessionStart,
    parse_client_message,
)
from maplo_voice.voice.session import VoiceAgentSession

if TYPE_CHECKING:
    from maplo_voice.config import Settings

log = get_logger(__name__)
router = APIRouter(tags=["voice"])

START_TIMEOUT_S = 10.0
TENANT_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
_TENANT_RE = re.compile(TENANT_PATTERN)


class WebSocketSender:
    """Serialises writes (turn + LLM tasks send concurrently) and tolerates disconnects."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._lock = asyncio.Lock()
        self.closed = False

    async def send_event(self, event: ServerEvent) -> None:
        await self._send(text=event.model_dump_json())

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send(data=pcm16)

    async def _send(self, *, text: str | None = None, data: bytes | None = None) -> None:
        if self.closed:
            return
        async with self._lock:
            try:
                if text is not None:
                    await self._ws.send_text(text)
                elif data is not None:
                    await self._ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                self.closed = True


def _bearer_token(websocket: WebSocket) -> str | None:
    for proto in websocket.scope.get("subprotocols", []):
        if proto.startswith(BEARER_PREFIX):
            return str(proto[len(BEARER_PREFIX) :])
    return None


def _origin_allowed(websocket: WebSocket, settings: Settings) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:  # non-browser clients (CLI, tests) send no Origin
        return True
    return "*" in settings.cors_origins or origin in settings.cors_origins


def _admission_error(websocket: WebSocket, tenant: str) -> tuple[CloseCode, str] | None:
    app = websocket.app
    settings: Settings = app.state.settings
    if not _TENANT_RE.match(tenant):
        return CloseCode.POLICY_VIOLATION, "invalid tenant"
    if not _origin_allowed(websocket, settings):
        return CloseCode.POLICY_VIOLATION, "origin not allowed"
    if not settings.auth.authorize(_bearer_token(websocket)):
        return CloseCode.POLICY_VIOLATION, "unauthorized"
    if app.state.voice_sessions_active >= settings.voice.max_concurrent_sessions:
        return CloseCode.TRY_AGAIN_LATER, "server at capacity"
    return None


async def _handshake(websocket: WebSocket, sender: WebSocketSender) -> SessionStart | None:
    """First message must be a valid ``session.start`` within ``START_TIMEOUT_S``."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=START_TIMEOUT_S)
        start = parse_client_message(raw)
        if not isinstance(start, SessionStart):
            raise TypeError("first message must be session.start")
    except (TimeoutError, ValidationError, TypeError, KeyError) as exc:
        await sender.send_event(
            ErrorEvent(code="invalid_start", message=f"Expected session.start: {exc}")
        )
        return None
    assessor = getattr(websocket.app.state, "assessor", None)
    if start.mode is SessionMode.ASSESSMENT and assessor is None:
        await sender.send_event(
            ErrorEvent(code="mode_unavailable", message="Assessment mode is not enabled.")
        )
        return None
    return start


async def _receive_loop(
    websocket: WebSocket, session: VoiceAgentSession, sender: WebSocketSender, idle_s: float
) -> None:
    while True:
        message = await asyncio.wait_for(websocket.receive(), timeout=idle_s)
        if message["type"] == "websocket.disconnect":
            return
        if (data := message.get("bytes")) is not None:
            if len(data) > MAX_AUDIO_FRAME_BYTES:
                await websocket.close(code=CloseCode.MESSAGE_TOO_BIG)
                return
            await session.on_audio(data)
            continue
        text = message.get("text")
        if text is None:
            continue
        if len(text) > MAX_TEXT_FRAME_BYTES:
            await websocket.close(code=CloseCode.MESSAGE_TOO_BIG)
            return
        try:
            control = parse_client_message(text)
        except ValidationError:
            await sender.send_event(
                ErrorEvent(code="invalid_message", message="Unrecognised message.")
            )
            continue
        if isinstance(control, SessionEnd):
            return
        await session.on_control(control)


@router.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    tenant: str = Query(default="default", pattern=TENANT_PATTERN),
) -> None:
    app = websocket.app
    settings: Settings = app.state.settings
    if (rejection := _admission_error(websocket, tenant)) is not None:
        code, reason = rejection
        log.info("voice_ws_rejected", code=int(code), reason=reason)
        await websocket.close(code=code, reason=reason)
        return

    offered = websocket.scope.get("subprotocols", [])
    await websocket.accept(subprotocol=PROTOCOL_VERSION if PROTOCOL_VERSION in offered else None)
    sender = WebSocketSender(websocket)
    app.state.voice_sessions_active += 1
    session: VoiceAgentSession | None = None
    status = SessionStatus.COMPLETED
    try:
        start = await _handshake(websocket, sender)
        if start is None:
            await websocket.close(code=CloseCode.POLICY_VIOLATION)
            return
        session = VoiceAgentSession(
            settings=settings,
            ai=get_ai_services(websocket),
            db=app.state.db,
            sender=sender,
            tenant_id=tenant,
            mode=start.mode,
            language=start.language,
            task_id=start.task_id,
            assessor=getattr(app.state, "assessor", None)
            if start.mode is SessionMode.ASSESSMENT
            else None,
            client_info={"user_agent": websocket.headers.get("user-agent", "")[:200]},
        )
        await session.start()
        await _receive_loop(websocket, session, sender, settings.voice.session_idle_timeout_s)
    except TimeoutError:
        await sender.send_event(ErrorEvent(code="idle_timeout", message="Session timed out."))
    except WebSocketDisconnect:
        pass
    except Exception:
        status = SessionStatus.FAILED
        log.exception("voice_ws_failed")
        await sender.send_event(ErrorEvent(code="internal_error", message="Session failed."))
    finally:
        app.state.voice_sessions_active -= 1
        if session is not None:
            await session.close(status)
        if websocket.client_state is WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await websocket.close(code=CloseCode.NORMAL)
