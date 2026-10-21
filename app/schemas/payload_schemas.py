"""
Nexus-AI Platform — API payload schemas + agent state.

Pydantic models for API request/response.
TypedDict for LangGraph state (LangGraph prefers TypedDict).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, StringConstraints

# ═══════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════

UserMessage = Annotated[
    str,
    StringConstraints(min_length=1, max_length=4000, strip_whitespace=True),
]

SessionId = Annotated[
    str,
    StringConstraints(
        min_length=8,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    ),
]


class ChatRequest(BaseModel):
    """Client theke asha chat request."""
    message: UserMessage = Field(
        ...,
        description="User er prosno. Max 4000 chars.",
        examples=["What is the self-exclusion period?"],
    )
    session_id: SessionId = Field(
        ...,
        description="Session ID (client generate kore).",
        examples=["sess-abc12345"],
    )
    stream: bool = Field(
        default=True,
        description="True = SSE streaming, False = single JSON response.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "message": "What is the self-exclusion period?",
                "session_id": "sess-abc12345",
                "stream": True,
            }]
        }
    }


# ═══════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Liveness probe."""
    status: Literal["ok", "degraded", "unhealthy"] = "ok"
    app_name: str
    version: str
    environment: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReadinessResponse(BaseModel):
    """Readiness probe."""
    ready: bool
    checks: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Citation(BaseModel):
    """Citation metadata — response reference."""
    source_id: int = Field(..., ge=1, description="Source number")
    snippet: str = Field(..., max_length=500, description="Text chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class ChatResponse(BaseModel):
    """Non-streaming response."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    routing: Literal[
        "policy", "database", "escalation", "chitchat", "fallback"
    ] = "policy"
    session_id: str
    request_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: str | None = None
    request_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SSEEvent(BaseModel):
    """SSE event payload."""
    event: Literal["token", "citation", "validation", "error", "done"]
    data: dict | str | list


# ═══════════════════════════════════════════════════════════════
# AGENT STATE (LangGraph — TypedDict)
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """
    LangGraph state — nodes ei dict e read/write kore.

    Ki: Whole pipeline er shared state.
    Kno: LangGraph TypedDict prefer kore — immutable-ish, channel-based.
    How: Prottek node state update kore, final node response build kore.

    Fields:
        query             → user er input
        session_id        → session tracking
        request_id        → distributed tracing
        intent            → classify result (policy/database/escalation/chitchat)
        retrieved_docs    → RAG output (LangChain Document objects)
        context           → formatted context string with [Source N]
        answer            → generated text (with citations)
        citations         → parsed citation metadata
        invalid_citations → citations that failed validation
        faithfulness_score → 0.0 - 1.0
        validation_passed  → True/False based on thresholds
        error             → error message if any
    """
    # Input
    query: str
    session_id: str
    request_id: str

    # Routing
    intent: Literal["policy", "database", "escalation", "chitchat"]

    # RAG
    retrieved_docs: list[Any]      # list[Document] — Any use korchi circular import avoid korte
    context: str

    # Generation
    answer: str

    # Validation
    citations: list[dict]
    invalid_citations: list[dict]
    faithfulness_score: float | None
    validation_passed: bool

    # Error
    error: str | None
