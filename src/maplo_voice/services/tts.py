"""Text-to-speech (TTS).

Streams raw PCM16-LE mono audio at 24 kHz (the API's ``pcm`` format — the lowest-latency
option, no container to decode). Chunks are kept sample-aligned (even byte counts) so the
browser can play each one immediately.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Protocol

import openai
from openai import omit

from maplo_voice.services.openai_client import genai_span, map_openai_error

if TYPE_CHECKING:
    from openai import AsyncOpenAI

TTS_SAMPLE_RATE = 24_000
MAX_INPUT_CHARS = 4096  # API limit per request

FirstByteCallback = Callable[[int], None]


class TextToSpeech(Protocol):
    def synthesize(
        self, text: str, *, on_first_byte: FirstByteCallback | None = None
    ) -> AsyncIterator[bytes]: ...


_MARKDOWN = re.compile(r"[*_#`>|~]+")
_URL = re.compile(r"https?://\S+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_SPACES = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Strip formatting the model may emit despite instructions, so TTS doesn't read it."""
    text = _URL.sub("a link", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    return _SPACES.sub(" ", text).strip()


def split_for_tts(text: str, limit: int = MAX_INPUT_CHARS) -> list[str]:
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind(". ", 0, limit)
        cut = cut + 1 if cut > 0 else text.rfind(" ", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut].strip())
        text = text[cut:]
    if text.strip():
        parts.append(text.strip())
    return parts


class OpenAISpeechSynthesizer:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        voice: str,
        instructions: str | None = None,
        chunk_bytes: int = 4_800,  # 100 ms of 24 kHz PCM16
    ) -> None:
        self._client = client
        self.model = model
        self._voice = voice
        # Steerable "instructions" are only supported by the gpt-*-tts family.
        self._instructions = instructions if model.startswith("gpt-") else None
        self._chunk_bytes = chunk_bytes

    async def synthesize(
        self, text: str, *, on_first_byte: FirstByteCallback | None = None
    ) -> AsyncIterator[bytes]:
        text = clean_for_speech(text)
        if not text:
            return
        start = time.perf_counter()
        first = True
        for part in split_for_tts(text):
            with genai_span("tts", "speech", self.model, **{"voice.tts.chars": len(part)}) as span:
                try:
                    async with self._client.audio.speech.with_streaming_response.create(
                        model=self.model,
                        voice=self._voice,
                        input=part,
                        instructions=self._instructions or omit,
                        response_format="pcm",
                    ) as response:
                        carry = b""
                        async for chunk in response.iter_bytes(self._chunk_bytes):
                            if first:
                                first = False
                                elapsed = int((time.perf_counter() - start) * 1000)
                                span.set_attribute("voice.tts.first_byte_ms", elapsed)
                                if on_first_byte is not None:
                                    on_first_byte(elapsed)
                            data = carry + chunk
                            aligned = len(data) - (len(data) % 2)
                            carry = data[aligned:]
                            if aligned:
                                yield data[:aligned]
                except openai.OpenAIError as exc:
                    raise map_openai_error("tts", exc) from exc
