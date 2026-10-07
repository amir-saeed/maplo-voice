"""Shared fixtures.

* Unit tests need nothing external.
* Integration tests (``@pytest.mark.integration``) need PostgreSQL + pgvector. Set
  ``TEST_DATABASE_URL`` (e.g. from ``docker compose up db``); otherwise they are skipped.
  The schema is created once per run with Alembic, and tables are truncated per test.
* OpenAI is faked at the HTTP layer (``httpx2.MockTransport``), so the real SDK parsing,
  streaming and error handling code paths run — only the network is replaced.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from openai import AsyncOpenAI

from maplo_voice.config import Settings

ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- audio helpers
def speech(seconds: float, amplitude: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype("<i2")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype="<i2")


def noise(seconds: float, amplitude: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude * 32767, int(SAMPLE_RATE * seconds)).astype("<i2")


def pcm(*parts: np.ndarray) -> bytes:
    return np.concatenate(parts).tobytes()


def frames(data: bytes, ms: int = 20) -> list[bytes]:
    size = SAMPLE_RATE * ms // 1000 * 2
    return [data[i : i + size] for i in range(0, len(data), size)]


def fluent_answer(seconds_each: float = 2.0, repeats: int = 10) -> bytes:
    """~23 s of 'speech' with short pauses (shorter than the VAD end-of-turn silence)."""
    parts = [silence(0.5)]
    for _ in range(repeats):
        parts += [speech(seconds_each), silence(0.3)]
    return pcm(*parts, silence(1.2))


# --------------------------------------------------------------------------- fake OpenAI
USAGE = {
    "input_tokens": 120,
    "output_tokens": 18,
    "total_tokens": 138,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}

DEFAULT_JUDGEMENT: dict[str, Any] = {
    "fluency": {"score": 68, "evidence": "we stayed in a small flat"},
    "grammar": {"score": 64, "evidence": "when we took the old tram"},
    "vocabulary": {"score": 60, "evidence": "a difficult year"},
    "coherence": {"score": 72, "evidence": "The best moment was"},
    "strengths": ["Clear sequence of events"],
    "improvements": ["Use more varied linking words"],
    "corrections": [
        {"original": "we walked", "corrected": "we would walk", "explanation": "habitual past"}
    ],
    "off_topic": False,
    "insufficient_sample": False,
    "spoken_feedback": "Well done, your story was easy to follow.",
}

LONG_ANSWER = (
    "Last summer I went to Lisbon with my sister. We stayed in a small flat near the river "
    "and every morning we walked to a cafe. The best moment was when we took the old tram up "
    "the hill and watched the sunset. I remember it because it was a difficult year."
)


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def _response_obj(model: str, output: list[Any]) -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": USAGE,
    }


@dataclass
class FakeOpenAI:
    """Configurable fake of the OpenAI HTTP API used by this service."""

    transcript: str = "What are your opening hours?"
    reply_deltas: list[str] = field(
        default_factory=lambda: ["We are open ", "nine to five. ", "On Saturdays we open at ten."]
    )
    judgement: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_JUDGEMENT))
    tts_seconds: float = 0.3
    fail: dict[str, int] = field(default_factory=dict)  # path suffix -> HTTP status
    requests: list[tuple[str, Any]] = field(default_factory=list)

    def calls(self, suffix: str) -> list[Any]:
        return [body for path, body in self.requests if path.endswith(suffix)]

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        is_json = request.headers.get("content-type", "").startswith("application/json")
        body = json.loads(request.content) if is_json else request.content
        self.requests.append((path, body))
        for suffix, status in self.fail.items():
            if path.endswith(suffix):  # "#stream" keys never match a path; handled below
                return httpx2.Response(status, json={"error": {"message": "fake failure"}})

        if path.endswith("/audio/transcriptions"):
            if isinstance(body, bytes) and b'name="stream"' in body:
                if "/audio/transcriptions#stream" in self.fail:
                    return httpx2.Response(400, json={"error": {"message": "stream failed"}})
                half = len(self.transcript) // 2
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse(
                        [
                            {"type": "transcript.text.delta", "delta": self.transcript[:half]},
                            {"type": "transcript.text.delta", "delta": self.transcript[half:]},
                            {
                                "type": "transcript.text.done",
                                "text": self.transcript,
                                "usage": {
                                    "type": "tokens",
                                    "input_tokens": 50,
                                    "output_tokens": 8,
                                    "total_tokens": 58,
                                },
                            },
                        ]
                    ),
                )
            return httpx2.Response(200, json={"text": f"fallback: {self.transcript}"})

        if path.endswith("/responses"):
            if body.get("stream"):
                events = [
                    {
                        "type": "response.output_text.delta",
                        "delta": d,
                        "item_id": "item",
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": i,
                        "logprobs": [],
                    }
                    for i, d in enumerate(self.reply_deltas)
                ]
                events.append(
                    {
                        "type": "response.completed",
                        "sequence_number": len(events),
                        "response": _response_obj(body["model"], []),
                    }
                )
                return httpx2.Response(
                    200, headers={"content-type": "text/event-stream"}, content=_sse(events)
                )
            message = {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": json.dumps(self.judgement), "annotations": []}
                ],
            }
            return httpx2.Response(200, json=_response_obj(body["model"], [message]))

        if path.endswith("/audio/speech"):
            n = int(24_000 * self.tts_seconds)
            tone = b"".join(
                struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 24_000)))
                for i in range(n)
            )
            return httpx2.Response(200, content=tone + b"\x00")  # odd length on purpose

        if path.endswith("/embeddings"):

            def vector(text: str) -> list[float]:
                seed = sum(text.encode()) % 997
                rng = np.random.default_rng(seed)
                v = rng.normal(size=body["dimensions"])
                return list((v / np.linalg.norm(v)).astype(float))

            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vector(t)}
                        for i, t in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )
        return httpx2.Response(404, json={"error": {"message": "unexpected path"}})

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="sk-test-not-real",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )


@pytest.fixture
def fake_openai() -> FakeOpenAI:
    return FakeOpenAI()


# --------------------------------------------------------------------------- settings / app
def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "test",
        "log_level": "WARNING",
        "log_json": False,
        "telemetry": {"enabled": False},
        "openai": {"api_key": "sk-test-not-real"},
        "database": {"url": TEST_DATABASE_URL or "postgresql+asyncpg://localhost:1/unreachable"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


AppFactory = Callable[..., Any]


@pytest.fixture
def app_factory(fake_openai: FakeOpenAI) -> AppFactory:
    """Build a fully wired app with OpenAI replaced by ``fake_openai``."""
    from maplo_voice.assessment.service import register_assessment
    from maplo_voice.main import create_app
    from maplo_voice.services.openai_client import build_ai_services

    def factory(**overrides: Any) -> Any:
        s = make_settings(**overrides)
        app = create_app(s)
        app.state.ai = build_ai_services(s, fake_openai.client())
        register_assessment(app, s)
        return app

    return factory


# --------------------------------------------------------------------------- database
def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture(scope="session")
def migrated_database() -> str:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set (integration tests need PostgreSQL + pgvector)")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    return TEST_DATABASE_URL


@pytest.fixture
def clean_db(migrated_database: str) -> str:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def truncate() -> None:
        engine = create_async_engine(migrated_database)
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE documents, voice_sessions RESTART IDENTITY CASCADE"))
        await engine.dispose()

    _run(truncate())
    return migrated_database


def fetch(sql: str, **params: Any) -> list[Any]:
    """Run a read query against the test DB (sync helper for assertions)."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def go() -> list[Any]:
        engine = create_async_engine(TEST_DATABASE_URL or "")
        async with engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).all()
        await engine.dispose()
        return list(rows)

    return _run(go())  # type: ignore[no-any-return]
