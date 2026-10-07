"""Knowledge-base REST API: ingest, list, delete and search documents (per tenant)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl

from maplo_voice.api.ws import TENANT_PATTERN
from maplo_voice.db.models import Document, DocumentChunk
from maplo_voice.db.session import DbSession  # noqa: TC001 - FastAPI resolves at runtime
from maplo_voice.services.openai_client import AIServiceError, AIServices, get_ai_services

router = APIRouter(prefix="/v1", tags=["knowledge-base"])
_bearer = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- auth
@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str


async def get_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_tenant_id: Annotated[str, Header(pattern=TENANT_PATTERN)] = "default",
) -> Principal:
    token = credentials.credentials if credentials else None
    if not request.app.state.settings.auth.authorize(token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(tenant_id=x_tenant_id)


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
AI = Annotated[AIServices, Depends(get_ai_services)]


def _ai_error(exc: AIServiceError) -> HTTPException:
    code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.retryable else status.HTTP_502_BAD_GATEWAY
    return HTTPException(code, detail=exc.message)


# --------------------------------------------------------------------------- schemas
class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=200_000)
    source_uri: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=50)


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    source_uri: str | None
    chunks: int
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    content: str
    similarity: float


# --------------------------------------------------------------------------- routes
@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def create_document(
    body: DocumentCreate, principal: CurrentPrincipal, db: DbSession, ai: AI
) -> DocumentOut:
    try:
        document = await ai.knowledge_base.ingest(
            db,
            tenant_id=principal.tenant_id,
            title=body.title,
            text=body.text,
            source_uri=str(body.source_uri) if body.source_uri else None,
            metadata=body.metadata,
        )
    except AIServiceError as exc:
        raise _ai_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return DocumentOut(
        id=document.id,
        title=document.title,
        source_uri=document.source_uri,
        chunks=len(document.chunks),
        created_at=document.created_at,
    )


@router.get("/documents")
async def list_documents(
    principal: CurrentPrincipal,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    chunk_count = (
        sa.select(sa.func.count(DocumentChunk.id))
        .where(DocumentChunk.document_id == Document.id)
        .scalar_subquery()
    )
    tenant_filter = Document.tenant_id == principal.tenant_id
    rows = (
        await db.execute(
            sa.select(Document, chunk_count.label("chunks"))
            .where(tenant_filter)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    total = await db.scalar(sa.select(sa.func.count(Document.id)).where(tenant_filter)) or 0
    return DocumentList(
        items=[
            DocumentOut(
                id=doc.id,
                title=doc.title,
                source_uri=doc.source_uri,
                chunks=chunks,
                created_at=doc.created_at,
            )
            for doc, chunks in rows
        ],
        total=total,
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, principal: CurrentPrincipal, db: DbSession, ai: AI
) -> Response:
    deleted = await ai.knowledge_base.delete_document(
        db, tenant_id=principal.tenant_id, document_id=document_id
    )
    if not deleted:  # same response for "missing" and "other tenant's" — no enumeration
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/search")
async def search(
    body: SearchRequest, principal: CurrentPrincipal, db: DbSession, ai: AI
) -> list[SearchHit]:
    try:
        hits = await ai.knowledge_base.search(
            db, tenant_id=principal.tenant_id, query=body.query, top_k=body.top_k
        )
    except AIServiceError as exc:
        raise _ai_error(exc) from exc
    return [
        SearchHit(
            chunk_id=h.id,
            document_id=h.document_id,
            title=h.title,
            content=h.content,
            similarity=round(h.similarity, 4),
        )
        for h in hits
    ]
