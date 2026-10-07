"""Integration tests: the WebSocket voice pipeline end to end (real app + DB, fake OpenAI)."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.conftest import (
    LONG_ANSWER,
    AppFactory,
    FakeOpenAI,
    fetch,
    fluent_answer,
    frames,
    make_settings,
    pcm,
    silence,
    speech,
)

from maplo_voice.db.models import SessionMode
from maplo_voice.db.session import Database
from maplo_voice.services.asr import Transcript
from maplo_voice.voice.session import VoiceAgentSession

pytestmark = pytest.mark.integration

PROTOCOLS = ["maplo.v1"]
ONE_TURN = pcm(silence(0.5), speech(1.0), silence(1.0))


def start(ws: Any, **kwargs: Any) -> dict[str, Any]:
    ws.send_text(json.dumps({"type": "session.start", **kwargs}))
    return json.loads(ws.receive_text())  # type: ignore[no-any-return]


def send_audio(ws: Any, audio: bytes) -> None:
    for frame in frames(audio):
        ws.send_bytes(frame)


def end_session(ws: Any) -> None:
    """Graceful end, as the UI does: session.end, then wait for the server's close frame."""
    ws.send_text(json.dumps({"type": "session.end"}))
    while ws.receive()["type"] != "websocket.close":
        pass


def collect_until(ws: Any, stop: str) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    audio_bytes = 0
    while True:
        message = ws.receive()
        if message.get("bytes"):
            audio_bytes += len(message["bytes"])
            continue
        event = json.loads(message["text"])
        events.append(event)
        if event["type"] in (stop, "error"):
            return events, audio_bytes


# --------------------------------------------------------------------------- conversation
def test_conversation_turn_end_to_end(app_factory: AppFactory, clean_db: str) -> None:
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice?tenant=acme", subprotocols=PROTOCOLS) as ws,
    ):
        assert ws.accepted_subprotocol == "maplo.v1"
        ready = start(ws, mode="conversation", language="en")
        assert ready["type"] == "session.ready"
        assert ready["task_prompt"] is None
        send_audio(ws, ONE_TURN)
        events, audio_bytes = collect_until(ws, "response.done")
        end_session(ws)

    types = [e["type"] for e in events]
    assert types[:2] == ["vad.speech_started", "vad.speech_stopped"]
    counts = Counter(types)
    assert counts["transcript.final"] == 1
    assert counts["response.audio.start"] == counts["response.audio.done"] == 1
    assert types.index("transcript.final") < types.index("response.text.delta")
    assert audio_bytes > 0
    assert audio_bytes % 2 == 0
    done = events[-1]
    assert done["metrics"]["time_to_first_audio_ms"] is not None

    [turn] = fetch("SELECT user_transcript, assistant_text, input_tokens, trace_id FROM turns")
    assert turn.user_transcript == "What are your opening hours?"
    assert turn.assistant_text.startswith("We are open")
    assert turn.input_tokens == 120
    [session] = fetch("SELECT tenant_id, status, ended_at FROM voice_sessions")
    assert (session.tenant_id, session.status) == ("acme", "completed")
    assert session.ended_at is not None


def test_store_transcripts_disabled_redacts(app_factory: AppFactory, clean_db: str) -> None:
    app = app_factory(voice={"store_transcripts": False})
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as ws,
    ):
        start(ws, mode="conversation")
        send_audio(ws, ONE_TURN)
        collect_until(ws, "response.done")
        end_session(ws)
    [turn] = fetch("SELECT user_transcript, assistant_text FROM turns")
    assert (turn.user_transcript, turn.assistant_text) == ("[redacted]", "[redacted]")


def test_control_messages(app_factory: AppFactory, clean_db: str) -> None:
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as ws,
    ):
        start(ws, mode="conversation")
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"
        ws.send_text("{not json")
        assert json.loads(ws.receive_text())["code"] == "invalid_message"
        # push-to-talk: commit before the silence timeout ends the turn immediately
        send_audio(ws, pcm(silence(0.5), speech(1.0)))
        ws.send_text(json.dumps({"type": "input.commit"}))
        events, _ = collect_until(ws, "response.done")
        assert events[-1]["type"] == "response.done"


def test_provider_failure_is_reported_not_fatal(
    app_factory: AppFactory, fake_openai: FakeOpenAI, clean_db: str
) -> None:
    fake_openai.fail["/responses"] = 429
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as ws,
    ):
        start(ws, mode="conversation")
        send_audio(ws, ONE_TURN)
        events, _ = collect_until(ws, "response.done")
        error = events[-1]
        assert error == {
            "type": "error",
            "code": "llm_failed",
            "message": "The service is busy, please try again.",
            "retryable": True,
        }
        ws.send_text(json.dumps({"type": "ping"}))  # session survives
        assert json.loads(ws.receive_text())["type"] == "pong"
        end_session(ws)
    [turn] = fetch("SELECT error FROM turns")
    assert turn.error.startswith("llm:")


# --------------------------------------------------------------------------- admission
@pytest.mark.parametrize(
    ("protocols", "headers", "query"),
    [
        (["maplo.v1", "bearer.wrong"], {}, ""),  # bad token
        (["maplo.v1"], {}, ""),  # missing token
        (["maplo.v1", "bearer.good-token"], {"origin": "https://evil.example"}, ""),  # origin
        (["maplo.v1", "bearer.good-token"], {}, "?tenant=Bad_Tenant"),  # tenant pattern
    ],
)
def test_rejected_connections(
    app_factory: AppFactory,
    clean_db: str,
    protocols: list[str],
    headers: dict[str, str],
    query: str,
) -> None:
    app = app_factory(auth={"api_tokens": ["good-token"]})
    url = f"/ws/voice{query}"
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(url, subprotocols=protocols, headers=headers) as ws,
    ):
        ws.receive_text()
    assert exc.value.code in (1008, 403)


def test_capacity_limit(app_factory: AppFactory, clean_db: str) -> None:
    app = app_factory(voice={"max_concurrent_sessions": 1})
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as first,
    ):
        start(first, mode="conversation")
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as second,
        ):
            second.receive_text()
        assert exc.value.code == 1013


def test_first_message_must_be_session_start(app_factory: AppFactory, clean_db: str) -> None:
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as ws,
    ):
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["code"] == "invalid_start"


# --------------------------------------------------------------------------- assessment
def test_assessment_end_to_end(
    app_factory: AppFactory, fake_openai: FakeOpenAI, clean_db: str
) -> None:
    fake_openai.transcript = LONG_ANSWER
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice?tenant=school", subprotocols=PROTOCOLS) as ws,
    ):
        ready = start(ws, mode="assessment", task_id="memorable-trip")
        assert ready["task_prompt"].startswith("Talk about a trip")
        send_audio(ws, fluent_answer())
        events, audio_bytes = collect_until(ws, "response.done")

    result = next(e for e in events if e["type"] == "assessment.result")["result"]
    assert result["status"] == "scored"
    assert result["cefr_level"] == "B2"
    llm_fluency = 68
    acoustic = result["feedback"]["acoustic_fluency"]
    assert result["scores"]["fluency"] == round(0.6 * llm_fluency + 0.4 * acoustic)
    feedback_text = "".join(e["delta"] for e in events if e["type"] == "response.text.delta")
    assert feedback_text.endswith("Your estimated level is B2.")
    assert audio_bytes > 0

    examiner_call = fake_openai.calls("/responses")[0]
    assert examiner_call["model"] == "gpt-6.1-sol"
    assert "ACOUSTIC MEASURES" in examiner_call["input"]
    [row] = fetch("SELECT tenant_id, cefr_level, overall_score FROM assessments")
    assert (row.tenant_id, row.cefr_level, row.overall_score) == (
        "school",
        "B2",
        result["overall_score"],
    )


def test_short_assessment_answer_skips_llm(
    app_factory: AppFactory, fake_openai: FakeOpenAI, clean_db: str
) -> None:
    with (
        TestClient(app_factory()) as client,
        client.websocket_connect("/ws/voice", subprotocols=PROTOCOLS) as ws,
    ):
        start(ws, mode="assessment")
        send_audio(ws, pcm(silence(0.5), speech(2.0), silence(1.2)))
        events, _ = collect_until(ws, "response.done")
    result = next(e for e in events if e["type"] == "assessment.result")["result"]
    assert result["status"] == "insufficient_sample"
    assert fake_openai.calls("/responses") == []
    assert fetch("SELECT count(*) AS n FROM assessments")[0].n == 0


# --------------------------------------------------------------------------- barge-in
class RecordingSender:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.audio = 0

    async def send_event(self, event: Any) -> None:
        self.events.append(event.type)

    async def send_audio(self, pcm16: bytes) -> None:
        self.audio += len(pcm16)


@dataclass
class SlowAI:
    """Fake services that stream slowly so a barge-in lands mid-response."""

    class _ASR:
        async def transcribe(self, pcm16: bytes, **_: Any) -> Transcript:
            return Transcript("tell me everything", "fake", 1000, 5)

    class _LLM:
        async def stream_reply(self, history: Any, **_: Any) -> Any:
            for i in range(100):
                await asyncio.sleep(0.01)
                yield f"Sentence number {i} is long enough to speak. "

    class _TTS:
        async def synthesize(self, text: str, **_: Any) -> Any:
            for _ in range(20):
                await asyncio.sleep(0.01)
                yield b"\x00\x00" * 240

    class _KB:
        async def search(self, *_: Any, **__: Any) -> list[Any]:
            return []

    asr: Any = field(default_factory=_ASR)
    llm: Any = field(default_factory=_LLM)
    tts: Any = field(default_factory=_TTS)
    knowledge_base: Any = field(default_factory=_KB)


async def test_barge_in_cancels_response_and_records_turn(clean_db: str) -> None:
    settings = make_settings()
    db = Database(settings)
    sender = RecordingSender()
    session = VoiceAgentSession(
        settings=settings,
        ai=SlowAI(),  # type: ignore[arg-type]
        db=db,
        sender=sender,
        tenant_id="acme",
        mode=SessionMode.CONVERSATION,
    )
    await session.start()
    for frame in frames(ONE_TURN):
        await session.on_audio(frame)
    await asyncio.sleep(0.4)  # response is streaming
    audio_before = sender.audio
    assert audio_before > 0
    for frame in frames(pcm(speech(0.5))):  # user starts talking again
        await session.on_audio(frame)
    await asyncio.sleep(0.2)

    assert "response.interrupted" in sender.events
    assert sender.events[-1] == "vad.speech_started"
    assert sender.audio - audio_before < 2_000  # playback stopped within ~one chunk
    await session.close()
    await db.dispose()

    [turn] = await asyncio.to_thread(
        fetch, "SELECT interrupted, length(assistant_text) AS n FROM turns"
    )
    assert turn.interrupted is True
    assert turn.n > 0  # partial reply kept for context/audit
