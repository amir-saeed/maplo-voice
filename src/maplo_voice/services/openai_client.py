"""Shared OpenAI client, GenAI tracing helper, error mapping and the service container.

The OpenAI SDK (v3) uses ``httpx2``, which generic httpx auto-instrumentation does not cover,
so every model call is wrapped in an explicit span following the OpenTelemetry GenAI
semantic conventions (``gen_ai.*`` attributes) and recorded in the pipeline metrics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import openai
from openai import AsyncOpenAI
from opentelemetry.trace import Span, StatusCode

from maplo_voice.observability import get_logger, pipeline_metrics, tracer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.requests import HTTPConnection

    from maplo_voice.config import Settings
    from maplo_voice.services.asr import OpenAITranscriber
    from maplo_voice.services.llm import OpenAIResponder
    from maplo_voice.services.rag import KnowledgeBase
    from maplo_voice.services.tts import OpenAISpeechSynthesizer

log = get_logger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class AIServiceError(Exception):
    """Provider-agnostic failure raised by every AI service.

    ``message`` is safe to show to end users; provider details stay in logs/traces.
    """

    def __init__(self, stage: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.retryable = retryable


def map_openai_error(stage: str, exc: openai.OpenAIError) -> AIServiceError:
    retryable = isinstance(exc, _RETRYABLE)
    status = getattr(exc, "status_code", None)
    log.warning(
        "openai_error",
        stage=stage,
        error_type=type(exc).__name__,
        status_code=status,
        retryable=retryable,
    )
    message = (
        "The service is busy, please try again."
        if retryable
        else "The AI service could not process this request."
    )
    return AIServiceError(stage, message, retryable=retryable)


@contextmanager
def genai_span(stage: str, operation: str, model: str, **attributes: Any) -> Iterator[Span]:
    """Span + latency/error metrics for one model call.

    Uses ``start_span`` (not ``start_as_current_span``) so it is safe inside async generators
    that may be closed from a different context (e.g. barge-in cancels a TTS stream).
    """
    span = tracer.start_span(
        f"{operation} {model}",
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": operation,
            "gen_ai.request.model": model,
            "voice.stage": stage,
            **attributes,
        },
    )
    start = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        pipeline_metrics.errors.add(1, {"stage": stage, "error.type": type(exc).__name__})
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, type(exc).__name__)
        raise
    finally:
        pipeline_metrics.stage_latency.record(
            (time.perf_counter() - start) * 1000, {"stage": stage, "model": model}
        )
        span.end()


def record_usage(
    span: Span, model: str, input_tokens: int | None, output_tokens: int | None
) -> None:
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        pipeline_metrics.openai_tokens.add(input_tokens, {"model": model, "direction": "input"})
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        pipeline_metrics.openai_tokens.add(output_tokens, {"model": model, "direction": "output"})


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    cfg = settings.openai
    api_key = cfg.api_key.get_secret_value()
    if not api_key:
        log.warning("openai_api_key_missing", hint="set OPENAI__API_KEY")
    return AsyncOpenAI(
        # The SDK refuses to construct without a key; a placeholder lets the app boot
        # (health/readiness report the misconfiguration) instead of crash-looping.
        api_key=api_key or "not-configured",
        base_url=cfg.base_url,
        organization=cfg.organization,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,  # SDK retries 408/409/429/5xx with exponential backoff
    )


# --------------------------------------------------------------------------- container
@dataclass(slots=True)
class AIServices:
    client: AsyncOpenAI
    asr: OpenAITranscriber
    llm: OpenAIResponder
    tts: OpenAISpeechSynthesizer
    knowledge_base: KnowledgeBase


def build_ai_services(settings: Settings, client: AsyncOpenAI | None = None) -> AIServices:
    # Local imports: the service modules import helpers from this module.
    from maplo_voice.services.asr import OpenAITranscriber  # noqa: PLC0415
    from maplo_voice.services.llm import OpenAIResponder  # noqa: PLC0415
    from maplo_voice.services.rag import KnowledgeBase, OpenAIEmbedder  # noqa: PLC0415
    from maplo_voice.services.tts import OpenAISpeechSynthesizer  # noqa: PLC0415

    cfg = settings.openai
    client = client or build_openai_client(settings)
    return AIServices(
        client=client,
        asr=OpenAITranscriber(client, model=cfg.asr_model, fallback_model=cfg.asr_fallback_model),
        llm=OpenAIResponder(
            client, model=cfg.llm_model, max_output_tokens=cfg.llm_max_output_tokens
        ),
        tts=OpenAISpeechSynthesizer(
            client, model=cfg.tts_model, voice=cfg.tts_voice, instructions=cfg.tts_instructions
        ),
        knowledge_base=KnowledgeBase(
            OpenAIEmbedder(client, model=cfg.embedding_model, dimensions=cfg.embedding_dimensions)
        ),
    )


def register_ai_services(app: Any, settings: Settings) -> AIServices:
    services = build_ai_services(settings)
    app.state.ai = services

    async def _check_configured() -> None:
        # Deliberately no network call: probes run every few seconds and must not
        # burn rate limit or money. Real provider health is visible in traces/metrics.
        if not settings.openai.api_key.get_secret_value():
            raise RuntimeError("OPENAI__API_KEY not configured")

    async def _close() -> None:
        await services.client.close()

    app.state.readiness_checks["openai"] = _check_configured
    app.state.shutdown_hooks.append(_close)
    return services


def get_ai_services(conn: HTTPConnection) -> AIServices:
    """Dependency usable from both HTTP routes and WebSocket endpoints."""
    services: AIServices = conn.app.state.ai
    return services
