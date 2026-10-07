"""Speech-to-text (ASR).

The voice pipeline segments speech with server-side VAD, then sends each complete
utterance here. The primary model streams partial transcripts (shown live in the UI);
if it fails, a fallback model is tried once before the error is surfaced.
"""

from __future__ import annotations

import io
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import openai
from openai import omit
from openai.types.audio import TranscriptionTextDeltaEvent, TranscriptionTextDoneEvent

from maplo_voice.observability import get_logger, pipeline_metrics
from maplo_voice.services.openai_client import genai_span, map_openai_error, record_usage

if TYPE_CHECKING:
    from openai import AsyncOpenAI

log = get_logger(__name__)

PartialCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    model: str
    audio_duration_ms: int
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SpeechToText(Protocol):
    async def transcribe(
        self,
        pcm16: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
        prompt: str | None = None,
        on_partial: PartialCallback | None = None,
    ) -> Transcript: ...


def pcm16_duration_ms(pcm16: bytes, sample_rate: int) -> int:
    return (len(pcm16) // 2) * 1000 // sample_rate


def pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono PCM16-LE in a WAV container (the API needs a file format)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return buf.getvalue()


class OpenAITranscriber:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        fallback_model: str | None = None,
        min_audio_ms: int = 200,
    ) -> None:
        self._client = client
        self.model = model
        self._fallback_model = fallback_model if fallback_model != model else None
        self._min_audio_ms = min_audio_ms

    async def transcribe(
        self,
        pcm16: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
        prompt: str | None = None,
        on_partial: PartialCallback | None = None,
    ) -> Transcript:
        duration_ms = pcm16_duration_ms(pcm16, sample_rate)
        if duration_ms < self._min_audio_ms:
            return Transcript("", self.model, duration_ms, 0)

        pipeline_metrics.audio_seconds_in.add(duration_ms / 1000, {"model": self.model})
        wav = pcm16_to_wav(pcm16, sample_rate)
        try:
            return await self._transcribe_stream(
                self.model,
                wav=wav,
                duration_ms=duration_ms,
                language=language,
                prompt=prompt,
                on_partial=on_partial,
            )
        except openai.OpenAIError as exc:
            if self._fallback_model is None:
                raise map_openai_error("asr", exc) from exc
            log.warning(
                "asr_fallback",
                primary=self.model,
                fallback=self._fallback_model,
                error_type=type(exc).__name__,
            )
            try:
                return await self._transcribe_once(
                    self._fallback_model,
                    wav=wav,
                    duration_ms=duration_ms,
                    language=language,
                    prompt=prompt,
                )
            except openai.OpenAIError as fallback_exc:
                raise map_openai_error("asr", fallback_exc) from fallback_exc

    async def _transcribe_stream(
        self,
        model: str,
        *,
        wav: bytes,
        duration_ms: int,
        language: str | None,
        prompt: str | None,
        on_partial: PartialCallback | None,
    ) -> Transcript:
        start = time.perf_counter()
        with genai_span(
            "asr", "transcription", model, **{"voice.audio.duration_ms": duration_ms}
        ) as span:
            stream = await self._client.audio.transcriptions.create(
                model=model,
                file=("utterance.wav", wav, "audio/wav"),
                stream=True,
                language=language or omit,
                prompt=prompt or omit,
            )
            parts: list[str] = []
            final: str | None = None
            in_tok = out_tok = None
            async for event in stream:
                if isinstance(event, TranscriptionTextDeltaEvent):
                    parts.append(event.delta)
                    if on_partial is not None:
                        await on_partial("".join(parts))
                elif isinstance(event, TranscriptionTextDoneEvent):
                    final = event.text
                    if event.usage is not None:
                        in_tok, out_tok = event.usage.input_tokens, event.usage.output_tokens
            text = (final if final is not None else "".join(parts)).strip()
            record_usage(span, model, in_tok, out_tok)
            span.set_attribute("voice.transcript.chars", len(text))
        return Transcript(
            text=text,
            model=model,
            audio_duration_ms=duration_ms,
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    async def _transcribe_once(
        self,
        model: str,
        *,
        wav: bytes,
        duration_ms: int,
        language: str | None,
        prompt: str | None,
    ) -> Transcript:
        start = time.perf_counter()
        with genai_span(
            "asr", "transcription", model, **{"voice.audio.duration_ms": duration_ms}
        ) as span:
            result = await self._client.audio.transcriptions.create(
                model=model,
                file=("utterance.wav", wav, "audio/wav"),
                language=language or omit,
                prompt=prompt or omit,
            )
            text = result.text.strip()
            span.set_attribute("voice.transcript.chars", len(text))
        return Transcript(
            text=text,
            model=model,
            audio_duration_ms=duration_ms,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
