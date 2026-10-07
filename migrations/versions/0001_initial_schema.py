"""Initial schema: pgvector knowledge base, voice sessions, turns, assessments.

Revision ID: 0001
Revises:
Create Date: 2026-10-07
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 1536
UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def _pk() -> sa.Column[object]:
    return sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()"))


def _created(name: str = "created_at") -> sa.Column[object]:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def _enum_check(column: str, values: Sequence[str], name: str) -> sa.CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in values)
    return sa.CheckConstraint(f"{column} IN ({allowed})", name=name)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # gen_random_uuid() on PG < 13

    # ------------------------------------------------------------- knowledge base
    op.create_table(
        "documents",
        _pk(),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("source_uri", sa.String(2000)),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        _created(),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "document_chunks",
        _pk(),
        sa.Column(
            "document_id",
            UUID,
            sa.ForeignKey(
                "documents.id",
                ondelete="CASCADE",
                name="fk_document_chunks_document_id_documents",
            ),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS), nullable=False),
        _created(),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_idx"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_tenant_id", "document_chunks", ["tenant_id"])
    op.create_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # ------------------------------------------------------------- voice sessions
    op.create_table(
        "voice_sessions",
        _pk(),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("llm_model", sa.String(100), nullable=False),
        sa.Column("asr_model", sa.String(100), nullable=False),
        sa.Column("tts_model", sa.String(100), nullable=False),
        sa.Column("client_info", JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        _created("started_at"),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text),
        _enum_check("mode", ["conversation", "assessment"], "ck_voice_sessions_session_mode"),
        _enum_check(
            "status", ["active", "completed", "failed"], "ck_voice_sessions_session_status"
        ),
    )
    op.create_index("ix_voice_sessions_tenant_id", "voice_sessions", ["tenant_id"])

    op.create_table(
        "turns",
        _pk(),
        sa.Column(
            "session_id",
            UUID,
            sa.ForeignKey(
                "voice_sessions.id", ondelete="CASCADE", name="fk_turns_session_id_voice_sessions"
            ),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer, nullable=False),
        sa.Column("user_transcript", sa.Text, nullable=False),
        sa.Column("assistant_text", sa.Text, nullable=False),
        sa.Column("interrupted", sa.Boolean, server_default=sa.false(), nullable=False),
        sa.Column("audio_duration_ms", sa.Integer, nullable=False),
        sa.Column("asr_ms", sa.Integer),
        sa.Column("llm_first_token_ms", sa.Integer),
        sa.Column("tts_first_byte_ms", sa.Integer),
        sa.Column("time_to_first_audio_ms", sa.Integer),
        sa.Column("total_ms", sa.Integer),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column(
            "retrieved_chunk_ids", JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column("trace_id", sa.String(32)),
        sa.Column("error", sa.Text),
        _created(),
        sa.UniqueConstraint("session_id", "turn_index", name="uq_turns_session_idx"),
    )
    op.create_index("ix_turns_session_id", "turns", ["session_id"])

    # ------------------------------------------------------------- assessments
    score_cols = ("overall_score", "fluency", "grammar", "vocabulary", "coherence")
    op.create_table(
        "assessments",
        _pk(),
        sa.Column(
            "session_id",
            UUID,
            sa.ForeignKey(
                "voice_sessions.id",
                ondelete="CASCADE",
                name="fk_assessments_session_id_voice_sessions",
            ),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("task_prompt", sa.Text, nullable=False),
        sa.Column("transcript", sa.Text, nullable=False),
        sa.Column("speech_duration_ms", sa.Integer, nullable=False),
        sa.Column("words_per_minute", sa.Float, nullable=False),
        sa.Column("cefr_level", sa.String(20), nullable=False),
        *(sa.Column(c, sa.SmallInteger, nullable=False) for c in score_cols),
        sa.Column("feedback", JSONB, nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        _created(),
        _enum_check(
            "cefr_level", ["A1", "A2", "B1", "B2", "C1", "C2"], "ck_assessments_cefr_level"
        ),
        *(
            sa.CheckConstraint(f"{c} BETWEEN 0 AND 100", name=f"ck_assessments_{c}_range")
            for c in score_cols
        ),
    )
    op.create_index("ix_assessments_session_id", "assessments", ["session_id"])
    op.create_index("ix_assessments_tenant_id", "assessments", ["tenant_id"])
    op.create_index("ix_assessments_cefr_level", "assessments", ["cefr_level"])


def downgrade() -> None:
    op.drop_table("assessments")
    op.drop_table("turns")
    op.drop_table("voice_sessions")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    # Extensions are left installed: other schemas may depend on them.
