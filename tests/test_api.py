"""HTTP API tests: health, security headers, knowledge base and assessments."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from tests.conftest import LONG_ANSWER, AppFactory, FakeOpenAI, fluent_answer, frames

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Tenant-ID": "acme"}


# --------------------------------------------------------------------------- unit (no DB)
def test_liveness_and_degraded_readiness(app_factory: AppFactory) -> None:
    app = app_factory(
        openai={"api_key": ""},
        database={"url": "postgresql+asyncpg://localhost:1/unreachable", "pool_timeout_s": 1},
    )
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["checks"]["openai"].startswith("error")
        assert ready.json()["checks"]["database"].startswith("error")


def test_security_headers_and_csp_scope(app_factory: AppFactory) -> None:
    with TestClient(app_factory()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Maplo Voice" in page.text
        assert "default-src 'self'" in page.headers["content-security-policy"]
        assert page.headers["x-frame-options"] == "DENY"
        assert page.headers["x-content-type-options"] == "nosniff"
        assert page.headers["x-request-id"]
        assert "content-security-policy" not in client.get("/docs").headers  # Swagger CDN
        assert client.get("/static/app.js").status_code == 200


def test_request_id_is_propagated(app_factory: AppFactory) -> None:
    with TestClient(app_factory()) as client:
        assert (
            client.get("/health/live", headers={"X-Request-ID": "abc123"}).headers["x-request-id"]
            == "abc123"
        )


def test_auth_required_when_tokens_configured(app_factory: AppFactory) -> None:
    app = app_factory(auth={"api_tokens": [TOKEN]})
    with TestClient(app) as client:
        response = client.get("/v1/documents")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert (
            client.get("/v1/assessment/tasks", headers={"Authorization": "Bearer nope"}).status_code
            == 401
        )
        assert client.get("/v1/assessment/tasks", headers=AUTH).status_code == 200


def test_invalid_tenant_header_rejected(app_factory: AppFactory) -> None:
    with TestClient(app_factory()) as client:
        assert (
            client.get("/v1/assessment/tasks", headers={"X-Tenant-ID": "../etc"}).status_code == 422
        )


# --------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_document_lifecycle_and_tenant_isolation(app_factory: AppFactory, clean_db: str) -> None:
    app = app_factory(auth={"api_tokens": [TOKEN]})
    other = {**AUTH, "X-Tenant-ID": "globex"}
    body = {"title": "Opening hours", "text": "We open nine to five.\n\nSaturday ten to two."}
    with TestClient(app) as client:
        created = client.post("/v1/documents", headers=AUTH, json=body)
        assert created.status_code == 201
        doc = created.json()
        assert doc["chunks"] == 1

        listing = client.get("/v1/documents", headers=AUTH).json()
        assert listing["total"] == 1
        assert listing["items"][0]["chunks"] == 1
        assert client.get("/v1/documents", headers=other).json()["total"] == 0

        hits = client.post("/v1/search", headers=AUTH, json={"query": body["text"]}).json()
        assert hits[0]["document_id"] == doc["id"]
        assert client.post("/v1/search", headers=other, json={"query": body["text"]}).json() == []

        assert client.delete(f"/v1/documents/{doc['id']}", headers=other).status_code == 404
        assert client.delete(f"/v1/documents/{doc['id']}", headers=AUTH).status_code == 204
        assert client.delete(f"/v1/documents/{uuid.uuid4()}", headers=AUTH).status_code == 404


@pytest.mark.integration
@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "text": "x"},
        {"title": "t", "text": ""},
        {"title": "t", "text": "x", "source_uri": "not-a-url"},
    ],
)
def test_document_validation(app_factory: AppFactory, clean_db: str, payload: dict) -> None:  # type: ignore[type-arg]
    with TestClient(app_factory()) as client:
        assert client.post("/v1/documents", headers=AUTH, json=payload).status_code == 422


@pytest.mark.integration
def test_document_ingest_provider_outage(
    app_factory: AppFactory, fake_openai: FakeOpenAI, clean_db: str
) -> None:
    fake_openai.fail["/embeddings"] = 503
    with TestClient(app_factory()) as client:
        response = client.post("/v1/documents", headers=AUTH, json={"title": "t", "text": "hello"})
        assert response.status_code == 503
        assert client.get("/v1/documents", headers=AUTH).json()["total"] == 0  # rolled back


@pytest.mark.integration
def test_assessment_history_and_stats(
    app_factory: AppFactory, fake_openai: FakeOpenAI, clean_db: str
) -> None:
    fake_openai.transcript = LONG_ANSWER
    with TestClient(app_factory()) as client:
        for _ in range(2):
            with client.websocket_connect("/ws/voice?tenant=acme", subprotocols=["maplo.v1"]) as ws:
                ws.send_text(json.dumps({"type": "session.start", "mode": "assessment"}))
                ws.receive_text()
                for frame in frames(fluent_answer()):
                    ws.send_bytes(frame)
                while True:
                    message = ws.receive()
                    if (
                        message.get("text")
                        and json.loads(message["text"])["type"] == "response.done"
                    ):
                        break

        listing = client.get("/v1/assessments", headers=AUTH).json()
        assert listing["total"] == 2
        item = listing["items"][0]
        assert item["cefr_level"] == "B2"
        assert set(item["scores"]) == {"fluency", "grammar", "vocabulary", "coherence"}

        assert client.get(f"/v1/assessments/{item['id']}", headers=AUTH).status_code == 200
        other = {**AUTH, "X-Tenant-ID": "globex"}
        assert client.get(f"/v1/assessments/{item['id']}", headers=other).status_code == 404
        assert client.get("/v1/assessments?cefr_level=C2", headers=AUTH).json()["total"] == 0

        stats = client.get("/v1/assessments/stats", headers=AUTH).json()
        assert stats["total"] == 2
        assert {d["cefr_level"]: d["count"] for d in stats["distribution"]}["B2"] == 2
        assert client.get("/v1/assessments/stats", headers=other).json()["average_overall"] is None

        tasks = client.get("/v1/assessment/tasks", headers=AUTH).json()
        assert {t["id"] for t in tasks} >= {"memorable-trip", "remote-work"}
