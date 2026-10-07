"""Retrieval-augmented generation: chunking, embeddings and pgvector search.

Ingestion:  text -> paragraph-aware chunks (with overlap) -> batched embeddings -> rows.
Retrieval:  query embedding -> HNSW cosine search scoped to the tenant -> similarity floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import openai
import sqlalchemy as sa

from maplo_voice.db.models import Document, DocumentChunk
from maplo_voice.observability import get_logger
from maplo_voice.services.openai_client import genai_span, map_openai_error, record_usage

if TYPE_CHECKING:
    import uuid

    from openai import AsyncOpenAI
    from sqlalchemy.ext.asyncio import AsyncSession

log = get_logger(__name__)

_PARAGRAPHS = re.compile(r"\n\s*\n")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"[ \t\r\f\v]+")


# --------------------------------------------------------------------------- chunking
def estimate_tokens(text: str) -> int:
    """~4 chars/token for English; good enough for budgeting without a tokenizer dep."""
    return max(1, len(text) // 4)


def chunk_text(text: str, *, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """Greedy paragraph packing; oversize paragraphs are split on sentences.

    Each chunk after the first is prefixed with the tail of the previous chunk (cut at a
    word boundary) so facts spanning a boundary remain retrievable.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    units: list[str] = []
    for raw in _PARAGRAPHS.split(text):
        para = _WS.sub(" ", raw).strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCES.split(para):
            # Pathological input with no punctuation: hard-split at max_chars.
            units.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap_chars:]
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
        current = f"{tail} {unit}" if len(tail) + len(unit) + 1 <= max_chars else unit
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- embeddings
class Embedder(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(
        self, client: AsyncOpenAI, *, model: str, dimensions: int, batch_size: int = 128
    ) -> None:
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            with genai_span(
                "embedding", "embeddings", self.model, **{"voice.embedding.batch": len(batch)}
            ) as span:
                try:
                    response = await self._client.embeddings.create(
                        model=self.model, input=batch, dimensions=self.dimensions
                    )
                except openai.OpenAIError as exc:
                    raise map_openai_error("embedding", exc) from exc
                record_usage(span, self.model, response.usage.prompt_tokens, None)
            vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
        return vectors


# --------------------------------------------------------------------------- knowledge base
@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: uuid.UUID
    document_id: uuid.UUID
    title: str
    content: str
    similarity: float


class KnowledgeBase:
    def __init__(
        self, embedder: Embedder, *, min_similarity: float = 0.25, ef_search: int = 100
    ) -> None:
        self._embedder = embedder
        self._min_similarity = min_similarity
        self._ef_search = ef_search

    async def ingest(
        self,
        session: AsyncSession,
        *,
        tenant_id: str,
        title: str,
        text: str,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document has no extractable text")
        vectors = await self._embedder.embed(chunks)
        document = Document(
            tenant_id=tenant_id, title=title, source_uri=source_uri, metadata_=metadata or {}
        )
        document.chunks = [
            DocumentChunk(
                tenant_id=tenant_id,
                chunk_index=i,
                content=content,
                token_count=estimate_tokens(content),
                embedding=vector,
            )
            for i, (content, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        session.add(document)
        await session.flush()
        log.info("document_ingested", tenant_id=tenant_id, chunks=len(chunks))
        return document

    async def search(
        self, session: AsyncSession, *, tenant_id: str, query: str, top_k: int = 4
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        [vector] = await self._embedder.embed([query])
        distance = DocumentChunk.embedding.cosine_distance(vector)
        # Wider HNSW candidate list so the tenant filter still leaves top_k results.
        await session.execute(sa.text(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}"))
        stmt = (
            sa.select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                DocumentChunk.content,
                (1 - distance).label("similarity"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.tenant_id == tenant_id)
            .order_by(distance)
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).all()
        return [
            RetrievedChunk(r.id, r.document_id, r.title, r.content, float(r.similarity))
            for r in rows
            if float(r.similarity) >= self._min_similarity
        ]

    async def delete_document(
        self, session: AsyncSession, *, tenant_id: str, document_id: uuid.UUID
    ) -> bool:
        result = await session.execute(
            sa.delete(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
        )
        return bool(getattr(result, "rowcount", 0))
