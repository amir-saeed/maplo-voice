"""Typed, validated application configuration.

All settings come from environment variables (or a local ``.env`` file).
Nested groups use ``__`` as the delimiter, e.g. ``OPENAI__API_KEY`` or ``VOICE__VAD_SILENCE_MS``.
Model IDs are configuration, not code, so they can be upgraded without a deploy.
"""

from __future__ import annotations

import hmac
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class OpenAISettings(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    organization: str | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Conversation LLM — optimised for low latency in a voice loop.
    llm_model: str = "gpt-6-luna"
    llm_max_output_tokens: int = Field(default=400, gt=0)
    # Assessment LLM — more capable model for structured CEFR scoring.
    assessment_model: str = "gpt-6.1-sol"
    # Speech-to-text (streaming-capable) with a fallback model.
    asr_model: str = "gpt-transcribe"
    asr_fallback_model: str = "gpt-4o-mini-transcribe"
    # Text-to-speech.
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "marin"
    tts_instructions: str = "Speak in a warm, clear and concise British English tone."
    # Embeddings for RAG.
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)


class DatabaseSettings(BaseModel):
    # No credentials in code: set DATABASE__URL via environment / secrets manager.
    url: PostgresDsn = PostgresDsn("postgresql+asyncpg://localhost:5432/maplo_voice")
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout_s: float = Field(default=10.0, gt=0)
    echo: bool = False


class VoiceSettings(BaseModel):
    """Audio contract between browser and server: mono PCM16 little-endian."""

    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000  # OpenAI TTS PCM output is 24 kHz
    frame_ms: int = Field(default=20, ge=10, le=100)
    vad_threshold_dbfs: float = Field(default=-45.0, le=0)
    vad_silence_ms: int = Field(default=700, ge=200)
    vad_min_speech_ms: int = Field(default=250, ge=50)
    max_utterance_s: float = Field(default=30.0, gt=0)
    max_concurrent_sessions: int = Field(default=200, ge=1)
    session_idle_timeout_s: float = Field(default=120.0, gt=0)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    history_turns: int = Field(default=6, ge=1, le=50)
    # Privacy: set False to persist only metrics, never what was said.
    store_transcripts: bool = True


class TelemetrySettings(BaseModel):
    enabled: bool = True
    service_name: str = "maplo-voice"
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    metric_export_interval_ms: int = Field(default=15_000, ge=1_000)


class AuthSettings(BaseModel):
    """Static bearer tokens for API + WebSocket access (swap for Cognito/OIDC JWTs in prod)."""

    api_tokens: list[SecretStr] = Field(default_factory=list)

    def authorize(self, token: str | None) -> bool:
        """No tokens configured = open (local/test only; enforced by Settings validator)."""
        return self.is_valid(token) if self.api_tokens else True

    def is_valid(self, token: str | None) -> bool:
        if not token:
            return False
        # Constant-time comparison to avoid timing side channels.
        return any(
            hmac.compare_digest(t.get_secret_value().encode(), token.encode())
            for t in self.api_tokens
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "maplo-voice"
    environment: Environment = "local"
    log_level: LogLevel = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @model_validator(mode="after")
    def _validate_production(self) -> Settings:
        if self.environment in ("staging", "production"):
            if not self.openai.api_key.get_secret_value():
                raise ValueError("OPENAI__API_KEY is required outside local/test")
            if not self.auth.api_tokens:
                raise ValueError("AUTH__API_TOKENS is required outside local/test")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS is not allowed outside local/test")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (use as a FastAPI dependency)."""
    return Settings()
