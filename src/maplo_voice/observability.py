"""Logging, tracing and metrics.

* structlog JSON logs, enriched with the active OpenTelemetry trace/span id so logs and
  traces correlate in Grafana/Jaeger.
* OTLP (gRPC) export of traces and metrics to an OpenTelemetry Collector.
* Auto-instrumentation for FastAPI and SQLAlchemy; OpenAI calls get explicit GenAI spans
  (services/openai_client.py) because the v3 SDK uses httpx2.
* Domain metrics for each voice-pipeline stage (ASR / LLM / TTS latency, time-to-first-audio).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from maplo_voice import __version__

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine
    from structlog.typing import EventDict, WrappedLogger

    from maplo_voice.config import Settings

TRACER_NAME = "maplo_voice"
tracer = trace.get_tracer(TRACER_NAME, __version__)
meter = metrics.get_meter(TRACER_NAME, __version__)


# --------------------------------------------------------------------------- logging
def _add_otel_context(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx.is_valid:
        event_dict["trace_id"] = format(span_ctx.trace_id, "032x")
        event_dict["span_id"] = format(span_ctx.span_id, "016x")
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Route both structlog and stdlib logging (uvicorn, sqlalchemy) through one pipeline."""
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_otel_context,
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    # The OpenAI SDK/httpx are noisy at INFO; spans already capture each call.
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- metrics
@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    stage_latency: metrics.Histogram
    time_to_first_audio: metrics.Histogram
    active_sessions: metrics.UpDownCounter
    turns: metrics.Counter
    errors: metrics.Counter
    audio_seconds_in: metrics.Counter
    openai_tokens: metrics.Counter


def _build_metrics() -> PipelineMetrics:
    return PipelineMetrics(
        stage_latency=meter.create_histogram(
            "voice.stage.duration", unit="ms", description="Latency per pipeline stage"
        ),
        time_to_first_audio=meter.create_histogram(
            "voice.time_to_first_audio",
            unit="ms",
            description="End of user speech to first TTS audio byte sent",
        ),
        active_sessions=meter.create_up_down_counter(
            "voice.sessions.active", description="Open WebSocket voice sessions"
        ),
        turns=meter.create_counter("voice.turns", description="Completed conversation turns"),
        errors=meter.create_counter("voice.errors", description="Pipeline errors by stage"),
        audio_seconds_in=meter.create_counter(
            "voice.audio.input", unit="s", description="Seconds of user audio processed"
        ),
        openai_tokens=meter.create_counter(
            "openai.tokens", description="OpenAI tokens consumed by model and direction"
        ),
    )


# Instruments bind lazily to whichever MeterProvider is global at record time.
pipeline_metrics = _build_metrics()


# --------------------------------------------------------------------------- telemetry
@dataclass(slots=True)
class Telemetry:
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None

    def shutdown(self) -> None:
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()


_telemetry: Telemetry | None = None


def setup_telemetry(settings: Settings) -> Telemetry:
    """Install global OTel providers once per process (idempotent; safe in tests)."""
    global _telemetry  # noqa: PLW0603
    if _telemetry is not None:
        return _telemetry
    if not settings.telemetry.enabled:
        _telemetry = Telemetry()
        return _telemetry

    cfg = settings.telemetry
    resource = Resource.create(
        {
            "service.name": cfg.service_name,
            "service.version": __version__,
            "deployment.environment.name": settings.environment,
        }
    )

    tracer_provider = TracerProvider(
        resource=resource, sampler=ParentBased(TraceIdRatioBased(cfg.trace_sample_ratio))
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=cfg.otlp_insecure))
    )
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=cfg.otlp_endpoint, insecure=cfg.otlp_insecure),
                export_interval_millis=cfg.metric_export_interval_ms,
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)

    _telemetry = Telemetry(tracer_provider, meter_provider)
    return _telemetry


def instrument_app(app: FastAPI, settings: Settings) -> None:
    if settings.telemetry.enabled:
        FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready")


def instrument_engine(engine: AsyncEngine, settings: Settings) -> None:
    if settings.telemetry.enabled:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger
