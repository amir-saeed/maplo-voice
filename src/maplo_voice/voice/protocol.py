"""WebSocket wire protocol (v1).

Transport
---------
* Sub-protocols: the client offers ``["maplo.v1", "bearer.<token>"]``. Browsers cannot set
  headers on WebSockets; this keeps the token out of URLs (and therefore out of access logs).
* Binary frames, client -> server: mono PCM16-LE at ``input_sample_rate`` (16 kHz).
* Binary frames, server -> client: mono PCM16-LE at ``output_sample_rate`` (24 kHz).
* Text frames: JSON messages below, discriminated by ``type``.

Flow
----
client ``session.start`` -> server ``session.ready`` -> client streams audio ->
server ``vad.speech_started`` / ``vad.speech_stopped`` -> ``transcript.partial``* ->
``transcript.final`` -> ``response.text.delta``* + ``response.audio.start`` + binary audio ->
``response.audio.done`` -> ``response.done``.
User speech during a response cancels it (barge-in) -> ``response.interrupted``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from maplo_voice.db.models import SessionMode

PROTOCOL_VERSION = "maplo.v1"
BEARER_PREFIX = "bearer."
MAX_AUDIO_FRAME_BYTES = 64 * 1024  # ~2 s of 16 kHz PCM16 per frame is already generous
MAX_TEXT_FRAME_BYTES = 4 * 1024


class CloseCode(IntEnum):
    NORMAL = 1000
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    INTERNAL_ERROR = 1011
    TRY_AGAIN_LATER = 1013


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- client -> server
class SessionStart(_Message):
    type: Literal["session.start"]
    mode: SessionMode = SessionMode.CONVERSATION
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$", description="ISO-639-1")
    task_id: str | None = Field(default=None, max_length=64, pattern=r"^[a-z0-9_-]+$")


class InputCommit(_Message):
    """Push-to-talk release: end the current utterance now instead of waiting for silence."""

    type: Literal["input.commit"]


class ResponseCancel(_Message):
    type: Literal["response.cancel"]


class SessionEnd(_Message):
    type: Literal["session.end"]


class Ping(_Message):
    type: Literal["ping"]


ClientMessage = Annotated[
    SessionStart | InputCommit | ResponseCancel | SessionEnd | Ping,
    Field(discriminator="type"),
]
_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Raises ``pydantic.ValidationError`` on malformed input."""
    return _client_adapter.validate_json(raw)


# --------------------------------------------------------------------------- server -> client
class SessionReady(_Message):
    type: Literal["session.ready"] = "session.ready"
    session_id: str
    mode: SessionMode
    input_sample_rate: int
    output_sample_rate: int
    task_prompt: str | None = None


class SpeechStarted(_Message):
    type: Literal["vad.speech_started"] = "vad.speech_started"


class SpeechStopped(_Message):
    type: Literal["vad.speech_stopped"] = "vad.speech_stopped"
    duration_ms: int


class TranscriptPartial(_Message):
    type: Literal["transcript.partial"] = "transcript.partial"
    turn_index: int
    text: str


class TranscriptFinal(_Message):
    type: Literal["transcript.final"] = "transcript.final"
    turn_index: int
    text: str


class ResponseTextDelta(_Message):
    type: Literal["response.text.delta"] = "response.text.delta"
    turn_index: int
    delta: str


class ResponseAudioStart(_Message):
    type: Literal["response.audio.start"] = "response.audio.start"
    turn_index: int
    sample_rate: int


class ResponseAudioDone(_Message):
    type: Literal["response.audio.done"] = "response.audio.done"
    turn_index: int


class TurnMetrics(_Message):
    audio_duration_ms: int
    asr_ms: int | None = None
    llm_first_token_ms: int | None = None
    tts_first_byte_ms: int | None = None
    time_to_first_audio_ms: int | None = None
    total_ms: int | None = None


class ResponseDone(_Message):
    type: Literal["response.done"] = "response.done"
    turn_index: int
    metrics: TurnMetrics
    sources: list[str] = Field(default_factory=list)


class ResponseInterrupted(_Message):
    type: Literal["response.interrupted"] = "response.interrupted"
    turn_index: int


class AssessmentResult(_Message):
    type: Literal["assessment.result"] = "assessment.result"
    turn_index: int
    result: dict[str, Any]


class ErrorEvent(_Message):
    type: Literal["error"] = "error"
    code: str
    message: str
    retryable: bool = False


class Pong(_Message):
    type: Literal["pong"] = "pong"


ServerEvent = (
    SessionReady
    | SpeechStarted
    | SpeechStopped
    | TranscriptPartial
    | TranscriptFinal
    | ResponseTextDelta
    | ResponseAudioStart
    | ResponseAudioDone
    | ResponseDone
    | ResponseInterrupted
    | AssessmentResult
    | ErrorEvent
    | Pong
)
