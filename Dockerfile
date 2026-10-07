# syntax=docker/dockerfile:1.7
# Multi-stage build: Poetry only exists in the builder; the runtime image is slim,
# non-root, read-only-filesystem friendly and contains no build tooling or secrets.

ARG PYTHON_VERSION=3.13

# ----------------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependencies first (cached layer unless pyproject/lock change).
COPY pyproject.toml poetry.lock* ./
RUN --mount=type=cache,target=/tmp/poetry-cache \
    poetry install --only main --no-root

# Then the project itself.
COPY src ./src
RUN poetry install --only main \
 && find .venv -type d -name "__pycache__" -prune -exec rm -rf {} +

# ----------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="maplo-voice" \
      org.opencontainers.image.description="Voice agent: FastAPI + OpenAI ASR/LLM/TTS + pgvector" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    ENVIRONMENT=production \
    LOG_JSON=true

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/src ./src
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).status == 200 else 1)"]

# Access logs are emitted by the app's own middleware (structured, with request ids).
CMD ["uvicorn", "maplo_voice.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--ws-ping-interval", "20", "--ws-ping-timeout", "20", \
     "--timeout-graceful-shutdown", "30"]
