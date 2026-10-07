"""SQLAlchemy 2.x ORM models.

Tables
------
documents / document_chunks  Knowledge base for RAG (pgvector HNSW cosine index).
voice_sessions               One row per WebSocket session (conversation or assessment).
turns                        One row per user→assistant exchange, with per-stage latency.
assessments                  CEFR speaking-assessment results (structured LLM output).

Multi-tenancy is row-level via ``tenant_id`` on every top-level table.
"""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003 - SQLAlchemy resolves Mapped[] at runtime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 1536  # must match OPENAI__EMBEDDING_DIMENSIONS and migration 0001

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONB}  # noqa: RUF012 - SQLAlchemy API


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


# --------------------------------------------------------------------------- enums
class SessionMode(StrEnum):
    CONVERSATION = "conversation"
    ASSESSMENT = "assessment"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class CEFRLevel(StrEnum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"


def _str_enum(enum: type[StrEnum], name: str) -> sa.Enum:
    # VARCHAR + CHECK constraint: portable and migration-friendly (no ALTER TYPE pain).
    return sa.Enum(
        enum,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda e: [m.value for m in e],
    )


# --------------------------------------------------------------------------- knowledge base
class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[str] = mapped_column(sa.String(64), index=True)
    title: Mapped[str] = mapped_column(sa.String(500))
    source_uri: Mapped[str | None] = mapped_column(sa.String(2000))
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", default=dict, server_default=sa.text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = _created_at()

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_idx"),
        sa.Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(sa.String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(sa.Integer)
    content: Mapped[str] = mapped_column(sa.Text)
    token_count: Mapped[int] = mapped_column(sa.Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = _created_at()

    document: Mapped[Document] = relationship(back_populates="chunks")


# --------------------------------------------------------------------------- voice sessions
class VoiceSession(Base):
    __tablename__ = "voice_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[str] = mapped_column(sa.String(64), index=True)
    mode: Mapped[SessionMode] = mapped_column(_str_enum(SessionMode, "session_mode"))
    status: Mapped[SessionStatus] = mapped_column(
        _str_enum(SessionStatus, "session_status"), default=SessionStatus.ACTIVE
    )
    llm_model: Mapped[str] = mapped_column(sa.String(100))
    asr_model: Mapped[str] = mapped_column(sa.String(100))
    tts_model: Mapped[str] = mapped_column(sa.String(100))
    client_info: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=sa.text("'{}'::jsonb")
    )
    started_at: Mapped[datetime] = _created_at()
    ended_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(sa.Text)

    turns: Mapped[list[Turn]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Turn.turn_index",
    )
    assessments: Mapped[list[Assessment]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (sa.UniqueConstraint("session_id", "turn_index", name="uq_turns_session_idx"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("voice_sessions.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(sa.Integer)
    user_transcript: Mapped[str] = mapped_column(sa.Text)
    assistant_text: Mapped[str] = mapped_column(sa.Text, default="")
    interrupted: Mapped[bool] = mapped_column(default=False, server_default=sa.false())

    # Latency breakdown (ms) — the core SLO data for a voice agent.
    audio_duration_ms: Mapped[int] = mapped_column(sa.Integer)
    asr_ms: Mapped[int | None] = mapped_column(sa.Integer)
    llm_first_token_ms: Mapped[int | None] = mapped_column(sa.Integer)
    tts_first_byte_ms: Mapped[int | None] = mapped_column(sa.Integer)
    time_to_first_audio_ms: Mapped[int | None] = mapped_column(sa.Integer)
    total_ms: Mapped[int | None] = mapped_column(sa.Integer)

    input_tokens: Mapped[int | None] = mapped_column(sa.Integer)
    output_tokens: Mapped[int | None] = mapped_column(sa.Integer)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(
        JSONB, default=list, server_default=sa.text("'[]'::jsonb")
    )
    trace_id: Mapped[str | None] = mapped_column(sa.String(32))
    error: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = _created_at()

    session: Mapped[VoiceSession] = relationship(back_populates="turns")


class Assessment(Base):
    __tablename__ = "assessments"
    __table_args__ = tuple(
        sa.CheckConstraint(f"{c} BETWEEN 0 AND 100", name=f"{c}_range")
        for c in ("overall_score", "fluency", "grammar", "vocabulary", "coherence")
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("voice_sessions.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(sa.String(64), index=True)
    task_id: Mapped[str] = mapped_column(sa.String(64))
    task_prompt: Mapped[str] = mapped_column(sa.Text)
    transcript: Mapped[str] = mapped_column(sa.Text)
    speech_duration_ms: Mapped[int] = mapped_column(sa.Integer)
    words_per_minute: Mapped[float] = mapped_column(sa.Float)

    cefr_level: Mapped[CEFRLevel] = mapped_column(_str_enum(CEFRLevel, "cefr_level"), index=True)
    overall_score: Mapped[int] = mapped_column(sa.SmallInteger)
    fluency: Mapped[int] = mapped_column(sa.SmallInteger)
    grammar: Mapped[int] = mapped_column(sa.SmallInteger)
    vocabulary: Mapped[int] = mapped_column(sa.SmallInteger)
    coherence: Mapped[int] = mapped_column(sa.SmallInteger)
    feedback: Mapped[dict[str, Any]] = mapped_column(default=dict)

    model: Mapped[str] = mapped_column(sa.String(100))
    created_at: Mapped[datetime] = _created_at()

    session: Mapped[VoiceSession] = relationship(back_populates="assessments")
