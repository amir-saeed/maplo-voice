"""One live voice conversation: VAD -> ASR -> (RAG + LLM | assessment) -> TTS.

Concurrency model (per WebSocket)
---------------------------------
* The receive loop (api/ws.py) feeds audio into ``on_audio`` synchronously; VAD is cheap.
* A completed utterance starts a *turn task*. Only one turn runs at a time.
* Inside a turn, LLM generation and speech synthesis run concurrently: the LLM producer
  pushes sentences into a queue while the speaker synthesises and streams the previous
  sentence. Time-to-first-audio is therefore ~ ASR + first sentence + TTS first byte.
* Barge-in: new user speech cancels the running turn task. Cancellation propagates into
  the OpenAI streams (closing HTTP connections, so we stop paying for unused tokens).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import sqlalchemy as sa
from opentelemetry import trace

from maplo_voice.db.models import SessionMode, SessionStatus, Turn, VoiceSession
from maplo_voice.observability import get_logger, pipeline_metrics, tracer
from maplo_voice.services.asr import pcm16_duration_ms
from maplo_voice.services.llm import ChatMessage, LLMUsage, SentenceChunker
from maplo_voice.services.openai_client import AIServiceError
from maplo_voice.services.tts import TTS_SAMPLE_RATE
from maplo_voice.voice.protocol import (
    AssessmentResult,
    ErrorEvent,
    InputCommit,
    Ping,
    Pong,
    ResponseAudioDone,
    ResponseAudioStart,
    ResponseCancel,
    ResponseDone,
    ResponseInterrupted,
    ResponseTextDelta,
    ServerEvent,
    SessionReady,
    SpeechStarted,
    SpeechStopped,
    TranscriptFinal,
    TranscriptPartial,
    TurnMetrics,
)
from maplo_voice.voice.vad import EnergyVAD, VadEvent, VadEventKind

if TYPE_CHECKING:
    from maplo_voice.config import Settings
    from maplo_voice.db.session import Database
    from maplo_voice.services.asr import Transcript
    from maplo_voice.services.openai_client import AIServices
    from maplo_voice.services.rag import RetrievedChunk
    from maplo_voice.voice.protocol import ClientMessage

log = get_logger(__name__)

REDACTED = "[redacted]"


class Sender(Protocol):
    async def send_event(self, event: ServerEvent) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class AssessmentReply:
    spoken_feedback: str
    result: dict[str, Any]


class Assessor(Protocol):
    """Implemented by the assessment module (batch 5); injected via ``app.state.assessor``."""

    def task_prompt(self, task_id: str | None) -> tuple[str, str]: ...

    async def assess(
        self,
        *,
        session_id: uuid.UUID,
        tenant_id: str,
        task_id: str,
        transcript: Transcript,
        pcm16: bytes,
        sample_rate: int,
    ) -> AssessmentReply: ...


@dataclass(slots=True)
class TurnRecord:
    turn_index: int
    audio_duration_ms: int
    user_transcript: str = ""
    assistant_text: str = ""
    interrupted: bool = False
    asr_ms: int | None = None
    llm_first_token_ms: int | None = None
    tts_first_byte_ms: int | None = None
    time_to_first_audio_ms: int | None = None
    total_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    trace_id: str | None = None
    error: str | None = None

    def metrics(self) -> TurnMetrics:
        return TurnMetrics(
            audio_duration_ms=self.audio_duration_ms,
            asr_ms=self.asr_ms,
            llm_first_token_ms=self.llm_first_token_ms,
            tts_first_byte_ms=self.tts_first_byte_ms,
            time_to_first_audio_ms=self.time_to_first_audio_ms,
            total_ms=self.total_ms,
        )


class VoiceAgentSession:
    def __init__(
        self,
        *,
        settings: Settings,
        ai: AIServices,
        db: Database,
        sender: Sender,
        tenant_id: str,
        mode: SessionMode,
        language: str | None = None,
        task_id: str | None = None,
        assessor: Assessor | None = None,
        client_info: dict[str, Any] | None = None,
    ) -> None:
        if mode is SessionMode.ASSESSMENT and assessor is None:
            raise ValueError("Assessment mode requires an assessor")
        self.id = uuid.uuid4()
        self._settings = settings
        self._cfg = settings.voice
        self._ai = ai
        self._db = db
        self._sender = sender
        self.tenant_id = tenant_id
        self.mode = mode
        self.language = language
        # The assessor is only relevant in assessment mode; ignore it otherwise.
        self._assessor = assessor if mode is SessionMode.ASSESSMENT else None
        self._task_id, self._task_prompt = (
            self._assessor.task_prompt(task_id) if self._assessor is not None else ("", "")
        )
        self._client_info = client_info or {}
        self._vad = EnergyVAD(
            sample_rate=self._cfg.input_sample_rate,
            frame_ms=self._cfg.frame_ms,
            threshold_dbfs=self._cfg.vad_threshold_dbfs,
            silence_ms=self._cfg.vad_silence_ms,
            min_speech_ms=self._cfg.vad_min_speech_ms,
            max_utterance_s=self._cfg.max_utterance_s,
        )
        self._history: list[ChatMessage] = []
        self._turn_task: asyncio.Task[None] | None = None
        self._next_turn = 0
        self._closed = False
        self._pending_writes: set[asyncio.Task[None]] = set()
        # Stable, non-reversible id for provider abuse monitoring (never send raw user ids).
        self._safety_id = hashlib.sha256(f"{tenant_id}:{self.id}".encode()).hexdigest()[:32]

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        cfg = self._settings.openai
        await self._db_write(
            VoiceSession(
                id=self.id,
                tenant_id=self.tenant_id,
                mode=self.mode,
                status=SessionStatus.ACTIVE,
                llm_model=cfg.assessment_model
                if self.mode is SessionMode.ASSESSMENT
                else cfg.llm_model,
                asr_model=cfg.asr_model,
                tts_model=cfg.tts_model,
                client_info=self._client_info,
            )
        )
        pipeline_metrics.active_sessions.add(1, {"mode": self.mode.value})
        await self._sender.send_event(
            SessionReady(
                session_id=str(self.id),
                mode=self.mode,
                input_sample_rate=self._cfg.input_sample_rate,
                output_sample_rate=TTS_SAMPLE_RATE,
                task_prompt=self._task_prompt or None,
            )
        )
        log.info("voice_session_started", session_id=str(self.id), mode=self.mode.value)

    async def close(self, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_turn()
        if self._pending_writes:  # turn rows must land before the session is marked ended
            await asyncio.gather(*self._pending_writes, return_exceptions=True)
        pipeline_metrics.active_sessions.add(-1, {"mode": self.mode.value})
        # Shielded: the socket task may be cancelled (client gone / shutdown) mid-write.
        await asyncio.shield(self._mark_ended(status))
        log.info("voice_session_closed", session_id=str(self.id), status=status.value)

    # ------------------------------------------------------------------ inputs
    async def on_audio(self, pcm16: bytes) -> None:
        if len(pcm16) % 2:
            await self._sender.send_event(
                ErrorEvent(code="invalid_audio", message="Audio must be 16-bit PCM.")
            )
            return
        for event in self._vad.process(pcm16):
            await self._on_vad_event(event)

    async def on_control(self, message: ClientMessage) -> None:
        if isinstance(message, InputCommit):
            if (event := self._vad.flush()) is not None:
                await self._on_vad_event(event)
        elif isinstance(message, ResponseCancel):
            await self._cancel_turn()
        elif isinstance(message, Ping):
            await self._sender.send_event(Pong())

    async def _on_vad_event(self, event: VadEvent) -> None:
        if event.kind is VadEventKind.SPEECH_START:
            await self._cancel_turn()  # barge-in
            await self._sender.send_event(SpeechStarted())
        elif event.kind is VadEventKind.SPEECH_DISCARDED:
            await self._sender.send_event(SpeechStopped(duration_ms=event.duration_ms))
        else:
            await self._sender.send_event(SpeechStopped(duration_ms=event.duration_ms))
            await self._cancel_turn()
            self._turn_task = asyncio.create_task(
                self._run_turn(event.audio, time.perf_counter()), name=f"turn-{self.id}"
            )

    async def _cancel_turn(self) -> None:
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ------------------------------------------------------------------ turn
    async def _run_turn(self, pcm16: bytes, speech_end: float) -> None:
        rec = TurnRecord(
            turn_index=self._next_turn,
            audio_duration_ms=pcm16_duration_ms(pcm16, self._cfg.input_sample_rate),
        )
        self._next_turn += 1
        with tracer.start_as_current_span(
            "voice.turn",
            attributes={
                "voice.session_id": str(self.id),
                "voice.turn_index": rec.turn_index,
                "voice.mode": self.mode.value,
                "voice.audio.duration_ms": rec.audio_duration_ms,
            },
        ) as span:
            rec.trace_id = format(span.get_span_context().trace_id, "032x")
            try:
                await self._execute_turn(rec, pcm16, speech_end)
            except asyncio.CancelledError:
                rec.interrupted = True
                span.set_attribute("voice.interrupted", True)
                await self._safe_send(ResponseInterrupted(turn_index=rec.turn_index))
                raise
            except AIServiceError as exc:
                rec.error = f"{exc.stage}: {exc.message}"
                span.set_status(trace.StatusCode.ERROR, exc.stage)
                await self._safe_send(
                    ErrorEvent(
                        code=f"{exc.stage}_failed", message=exc.message, retryable=exc.retryable
                    )
                )
            except Exception:
                log.exception("voice_turn_failed", session_id=str(self.id))
                rec.error = "internal_error"
                pipeline_metrics.errors.add(1, {"stage": "turn", "error.type": "internal"})
                await self._safe_send(
                    ErrorEvent(code="internal_error", message="Something went wrong.")
                )
            finally:
                rec.total_ms = int((time.perf_counter() - speech_end) * 1000)
                if rec.user_transcript:
                    # Run as its own task (tracked) and shield it, so a barge-in or a
                    # disconnect can't lose the record; close() waits for pending writes.
                    write = asyncio.create_task(self._persist_turn(rec))
                    self._pending_writes.add(write)
                    write.add_done_callback(self._pending_writes.discard)
                    await asyncio.shield(write)

    async def _execute_turn(self, rec: TurnRecord, pcm16: bytes, speech_end: float) -> None:
        async def on_partial(text: str) -> None:
            await self._sender.send_event(TranscriptPartial(turn_index=rec.turn_index, text=text))

        transcript = await self._ai.asr.transcribe(
            pcm16,
            sample_rate=self._cfg.input_sample_rate,
            language=self.language,
            on_partial=on_partial,
        )
        rec.asr_ms = transcript.latency_ms
        await self._sender.send_event(
            TranscriptFinal(turn_index=rec.turn_index, text=transcript.text)
        )
        if transcript.is_empty:
            return
        rec.user_transcript = transcript.text

        sources: list[str] = []
        if self.mode is SessionMode.ASSESSMENT:
            assert self._assessor is not None  # noqa: S101 - guaranteed by __init__
            reply = await self._assessor.assess(
                session_id=self.id,
                tenant_id=self.tenant_id,
                task_id=self._task_id,
                transcript=transcript,
                pcm16=pcm16,
                sample_rate=self._cfg.input_sample_rate,
            )
            await self._sender.send_event(
                AssessmentResult(turn_index=rec.turn_index, result=reply.result)
            )
            rec.assistant_text = reply.spoken_feedback
            await self._sender.send_event(
                ResponseTextDelta(turn_index=rec.turn_index, delta=reply.spoken_feedback)
            )
            await self._speak(_queue_from_text(reply.spoken_feedback), rec, speech_end)
        else:
            sources = await self._converse(transcript.text, rec, speech_end)

        pipeline_metrics.turns.add(1, {"mode": self.mode.value})
        rec.total_ms = int((time.perf_counter() - speech_end) * 1000)
        await self._sender.send_event(
            ResponseDone(turn_index=rec.turn_index, metrics=rec.metrics(), sources=sources)
        )

    async def _converse(self, text: str, rec: TurnRecord, speech_end: float) -> list[str]:
        chunks = await self._retrieve(text)
        rec.retrieved_chunk_ids = [str(c.id) for c in chunks]
        self._history.append(ChatMessage("user", text))
        self._trim_history()

        usage = LLMUsage()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        reply_parts: list[str] = []

        async def produce() -> None:
            chunker = SentenceChunker()
            try:
                async for delta in self._ai.llm.stream_reply(
                    self._history,
                    context=[c.content for c in chunks],
                    usage=usage,
                    safety_identifier=self._safety_id,
                ):
                    reply_parts.append(delta)
                    await self._sender.send_event(
                        ResponseTextDelta(turn_index=rec.turn_index, delta=delta)
                    )
                    for sentence in chunker.feed(delta):
                        await queue.put(sentence)
                if (rest := chunker.flush()) is not None:
                    await queue.put(rest)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce(), name=f"llm-{self.id}")
        try:
            await self._speak(queue, rec, speech_end)
            await producer  # surface LLM errors
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer
            reply = "".join(reply_parts).strip()
            rec.assistant_text = reply
            rec.llm_first_token_ms = usage.first_token_ms
            rec.input_tokens, rec.output_tokens = usage.input_tokens, usage.output_tokens
            if reply:
                self._history.append(ChatMessage("assistant", reply))
        return sorted({c.title for c in chunks})

    async def _speak(
        self, queue: asyncio.Queue[str | None], rec: TurnRecord, speech_end: float
    ) -> None:
        started = False

        def on_first_byte(ms: int) -> None:
            if rec.tts_first_byte_ms is None:
                rec.tts_first_byte_ms = ms

        while (sentence := await queue.get()) is not None:
            async for pcm in self._ai.tts.synthesize(sentence, on_first_byte=on_first_byte):
                if not started:
                    started = True
                    rec.time_to_first_audio_ms = int((time.perf_counter() - speech_end) * 1000)
                    pipeline_metrics.time_to_first_audio.record(
                        rec.time_to_first_audio_ms, {"mode": self.mode.value}
                    )
                    await self._sender.send_event(
                        ResponseAudioStart(turn_index=rec.turn_index, sample_rate=TTS_SAMPLE_RATE)
                    )
                await self._sender.send_audio(pcm)
        if started:
            await self._sender.send_event(ResponseAudioDone(turn_index=rec.turn_index))

    async def _retrieve(self, query: str) -> list[RetrievedChunk]:
        """RAG is best-effort: a retrieval failure degrades to an ungrounded answer."""
        try:
            async with self._db.session() as db:
                return await self._ai.knowledge_base.search(
                    db, tenant_id=self.tenant_id, query=query, top_k=self._cfg.rag_top_k
                )
        except Exception:
            log.warning("rag_retrieval_failed", session_id=str(self.id), exc_info=True)
            return []

    def _trim_history(self) -> None:
        max_messages = self._cfg.history_turns * 2
        if len(self._history) > max_messages:
            del self._history[: len(self._history) - max_messages]

    # ------------------------------------------------------------------ persistence
    async def _persist_turn(self, rec: TurnRecord) -> None:
        store = self._cfg.store_transcripts
        try:
            await self._db_write(
                Turn(
                    session_id=self.id,
                    turn_index=rec.turn_index,
                    user_transcript=rec.user_transcript if store else REDACTED,
                    assistant_text=rec.assistant_text if store else REDACTED,
                    interrupted=rec.interrupted,
                    audio_duration_ms=rec.audio_duration_ms,
                    asr_ms=rec.asr_ms,
                    llm_first_token_ms=rec.llm_first_token_ms,
                    tts_first_byte_ms=rec.tts_first_byte_ms,
                    time_to_first_audio_ms=rec.time_to_first_audio_ms,
                    total_ms=rec.total_ms,
                    input_tokens=rec.input_tokens,
                    output_tokens=rec.output_tokens,
                    retrieved_chunk_ids=rec.retrieved_chunk_ids,
                    trace_id=rec.trace_id,
                    error=rec.error,
                )
            )
        except Exception:
            log.exception("turn_persist_failed", session_id=str(self.id))

    async def _mark_ended(self, status: SessionStatus) -> None:
        try:
            async with self._db.session() as db:
                await db.execute(
                    sa.update(VoiceSession)
                    .where(VoiceSession.id == self.id)
                    .values(status=status, ended_at=datetime.now(UTC))
                )
        except Exception:
            log.exception("voice_session_close_persist_failed", session_id=str(self.id))

    async def _db_write(self, row: object) -> None:
        async with self._db.session() as db:
            db.add(row)

    async def _safe_send(self, event: ServerEvent) -> None:
        with contextlib.suppress(Exception):
            await self._sender.send_event(event)


def _queue_from_text(text: str) -> asyncio.Queue[str | None]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    chunker = SentenceChunker()
    for sentence in chunker.feed(text):
        queue.put_nowait(sentence)
    if (rest := chunker.flush()) is not None:
        queue.put_nowait(rest)
    queue.put_nowait(None)
    return queue
