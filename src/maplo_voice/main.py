"""Application factory.

Run locally:  uvicorn maplo_voice.main:create_app --factory --reload
Or:           poetry run maplo-voice
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from maplo_voice import __version__
from maplo_voice.config import Settings, get_settings
from maplo_voice.db.session import register_database
from maplo_voice.observability import (
    configure_logging,
    get_logger,
    instrument_app,
    setup_telemetry,
)

log = get_logger(__name__)

ReadinessCheck = Callable[[], Awaitable[None]]
REQUEST_ID_HEADER = "X-Request-ID"


# --------------------------------------------------------------------------- lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    log.info("startup", environment=settings.environment, version=__version__)
    # Resources (DB engine, OpenAI client, ...) register startup/shutdown hooks here.
    for hook in app.state.startup_hooks:
        await hook()
    try:
        yield
    finally:
        for hook in reversed(app.state.shutdown_hooks):
            try:
                await hook()
            except Exception:
                log.exception("shutdown_hook_failed")
        app.state.telemetry.shutdown()
        log.info("shutdown")


# --------------------------------------------------------------------------- health
health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@health_router.get("/ready", summary="Readiness probe (checks dependencies)")
async def ready(request: Request, response: Response) -> dict[str, Any]:
    checks: dict[str, ReadinessCheck] = request.app.state.readiness_checks
    results: dict[str, str] = {}
    for name, check in checks.items():
        try:
            await check()
            results[name] = "ok"
        except Exception as exc:  # report, don't raise — this is a probe
            results[name] = f"error: {type(exc).__name__}"
    healthy = all(v == "ok" for v in results.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": results}


# --------------------------------------------------------------------------- factory
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    telemetry = setup_telemetry(settings)

    app = FastAPI(
        title="Maplo Voice",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.telemetry = telemetry
    app.state.readiness_checks = {}
    app.state.startup_hooks = []
    app.state.shutdown_hooks = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        if not request.url.path.startswith("/health"):
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "request_id": request.headers.get(REQUEST_ID_HEADER),
            },
        )

    register_database(app, settings)
    app.include_router(health_router)
    instrument_app(app, settings)
    return app


def run() -> None:
    """Console-script entrypoint."""
    import uvicorn  # noqa: PLC0415

    settings = get_settings()
    uvicorn.run(
        "maplo_voice.main:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 - container entrypoint
        port=8000,
        log_config=None,  # logging is owned by configure_logging()
        ws_ping_interval=20,
        ws_ping_timeout=20,
        reload=settings.environment == "local",
    )
