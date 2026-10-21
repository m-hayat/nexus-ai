"""
Nexus-AI Platform — API integration tests.

FastAPI TestClient diye endpoint gulo test kore.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture — lifespan run kore (startup/shutdown).
    `with` block must, karon lifespan events trigger hoy.
    """
    get_settings.cache_clear()
    from main import app
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════
# HEALTH & READINESS
# ═══════════════════════════════════════════════════════════════

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "Nexus-AI Platform"
    assert "timestamp" in data


def test_ready_returns_ready_true(client):
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is True
    assert "checks" in data
    assert "mcp" in data["checks"]


def test_request_id_header_returned(client):
    """Request ID middleware prottek response e X-Request-ID pathay."""
    r = client.get("/health")
    assert "X-Request-ID" in r.headers


def test_request_id_respected_when_provided(client):
    """Client je request_id pathay, seta preserve hoy."""
    custom_id = "test-request-abc-123"
    r = client.get("/health", headers={"X-Request-ID": custom_id})
    assert r.headers["X-Request-ID"] == custom_id


# ═══════════════════════════════════════════════════════════════
# CHAT (NON-STREAMING)
# ═══════════════════════════════════════════════════════════════

def test_chat_valid_request(client):
    payload = {
        "message": "What is the deposit limit?",
        "session_id": "sess-12345678",
        "stream": False,
    }
    r = client.post("/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["session_id"] == "sess-12345678"
    assert data["routing"] == "policy"
    assert "request_id" in data


def test_chat_empty_message_rejected(client):
    """Empty message → 422."""
    payload = {"message": "", "session_id": "sess-12345678"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 422


def test_chat_message_too_long_rejected(client):
    """Message 4000+ chars → 422."""
    payload = {"message": "x" * 5000, "session_id": "sess-12345678"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 422


def test_chat_invalid_session_id_rejected(client):
    """Session ID pattern violate → 422."""
    payload = {"message": "hi", "session_id": "sess with spaces!"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 422


def test_chat_missing_fields_rejected(client):
    """Missing required fields → 422."""
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════
# CHAT STREAMING (SSE)
# ═══════════════════════════════════════════════════════════════

def test_stream_returns_event_stream(client):
    payload = {
        "message": "hi",
        "session_id": "sess-12345678",
        "stream": True,
    }
    r = client.post("/chat/stream", json=payload)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_stream_contains_expected_events(client):
    """
    SSE stream e expected sequence: validation, citation, token, done.

    Policy question use kori — citation thakbe.
    """
    payload = {
        "message": "What is the self-exclusion period?",
        "session_id": "sess-12345678",
        "stream": True,
    }
    r = client.post("/chat/stream", json=payload)
    body = r.text
    assert "event: validation" in body
    assert "event: citation" in body
    assert "event: token" in body
    assert "event: done" in body


def test_stream_validation_before_tokens(client):
    """Buffer-then-stream: validation event prothome, token er age."""
    payload = {
        "message": "What is the deposit limit?",
        "session_id": "sess-12345678",
        "stream": True,
    }
    r = client.post("/chat/stream", json=payload)
    body = r.text
    validation_pos = body.find("event: validation")
    token_pos = body.find("event: token")
    done_pos = body.find("event: done")
    assert validation_pos < token_pos < done_pos


# ═══════════════════════════════════════════════════════════════
# COMPLIANCE
# ═══════════════════════════════════════════════════════════════

def test_compliance_endpoint(client):
    r = client.get("/compliance")
    assert r.status_code == 200
    data = r.json()
    assert "framework" in data
    assert "articles" in data
    assert "Article 13 - Transparency" in data["articles"]
    assert "Article 14 - Human Oversight" in data["articles"]
    assert "Article 15 - Accuracy, Robustness, and Cybersecurity" in data["articles"]


# ═══════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════

def test_404_for_unknown_endpoint(client):
    r = client.get("/nonexistent-endpoint")
    assert r.status_code == 404


def test_method_not_allowed(client):
    """GET /chat (POST only) → 405."""
    r = client.get("/chat")
    assert r.status_code == 405
