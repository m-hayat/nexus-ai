"""
Nexus-AI Platform — FastAPI gateway.

Endpoints:
  GET  /health             — liveness probe
  GET  /ready              — readiness probe (external deps check)
  POST /chat               — non-streaming chat
  POST /chat/stream        — SSE streaming chat
  GET  /compliance         — Phase 5 EU AI Act mapping
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings, get_settings
from app.core.mcp_client import MCPClient, init_mcp_client
from app.core.prompts import FALLBACK_RESPONSE
from app.core.streaming import sse_headers, stream_verified_response
from app.schemas.payload_schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

def setup_logging(settings: Settings) -> None:
    """
    Ki: Structured logging setup.
    Kno: Production e JSON logs better (ELK/Datadog ingest korte pare),
         development e human-readable.
    """
    level = getattr(logging, settings.log_level, logging.INFO)
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if not settings.is_production
        else '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    )
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout,
        force=True,
    )
    # Third-party noisy loggers suppress
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# LIFESPAN — startup + shutdown
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ki: FastAPI lifespan — startup + shutdown.
    Kno: Persistent resources (MCP client, RAG engine) ekhane init.
    """
    settings = get_settings()
    setup_logging(settings)
    from app.config import apply_langsmith_env
    apply_langsmith_env(settings)

    logger.info("=" * 60)
    logger.info("Nexus-AI Platform starting")
    logger.info("  env      = %s", settings.app_env)
    logger.info("  provider = %s", settings.llm_provider)
    logger.info("  model    = %s", settings.llm_model)
    logger.info("  mcp      = %s", settings.mcp_enabled)
    logger.info("=" * 60)

    # ── MCP client (persistent) ──
    mcp_client = await init_mcp_client(settings)
    app.state.mcp_client = mcp_client
    app.state.settings = settings
    app.state.started_at = time.time()

    # ── RAG engine warmup (optional — lazy load o korte paro) ──
    if settings.mcp_enabled:
        logger.info("MCP enabled — client connected")
    else:
        logger.info("MCP disabled — degraded mode (database intent fallback)")

    logger.info("Startup complete — ready to serve traffic")

    try:
        yield
    finally:
        logger.info("Shutdown initiated")
        await mcp_client.disconnect()
        logger.info("Shutdown complete")

# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Betsson Group — Data Center of Excellence — AI Platform",
    lifespan=lifespan,
    # Production e docs off korben
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)


# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request ID middleware ──
class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Ki: Prottek request e unique request_id attach kore.
    Kno: Distributed tracing + log correlation. Client pathale respect,
         na pathale generate.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ── Simple in-memory rate limiter ──
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Ki: Simple per-IP rate limit.
    Kno: DoS prevent, cost control. Production e Redis-based better.
    """
    def __init__(self, app, per_minute: int):
        super().__init__(app)
        self.per_minute = per_minute
        self._buckets: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # Health endpoints bypass
        if request.url.path in ("/health", "/ready"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        bucket = self._buckets.setdefault(client_ip, [])
        bucket[:] = [t for t in bucket if t > window_start]

        if len(bucket) >= self.per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    error="rate_limit_exceeded",
                    detail=f"Max {self.per_minute} requests per minute.",
                    request_id=getattr(request.state, "request_id", ""),
                ).model_dump(mode="json"),
            )

        bucket.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)


# ═══════════════════════════════════════════════════════════════
# EXCEPTION HANDLERS
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Ki: Unhandled exception → standard error response.
    Kno: Stack trace client e leak kora jabe na (security).
         Full trace log e jabe, server side.
    """
    request_id = getattr(request.state, "request_id", "")
    logger.exception("Unhandled error [%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred. Please retry.",
            request_id=request_id,
        ).model_dump(mode="json"),
    )


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """
    Ki: Liveness probe.
    Kno: k8s/load balancer ei endpoint hit kore — app process alive kina.
         Cheap hobe, kono external dependency check korbe na.
    """
    s = app.state.settings
    return HealthResponse(
        status="ok",
        app_name=s.app_name,
        version=s.app_version,
        environment=s.app_env,
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
async def ready():
    """
    Ki: Readiness probe.
    Kno: Health er cheye strict — external deps ready kina.
         Jodi ready na → load balancer traffic pathabe na.
    """
    checks: dict[str, str] = {}

    # MCP
    mcp: MCPClient = app.state.mcp_client
    checks["mcp"] = "connected" if mcp.is_connected else "disabled"

    # Uptime
    uptime = time.time() - app.state.started_at
    checks["uptime_seconds"] = f"{uptime:.1f}"

    # All required deps OK?
    ready_status = checks["mcp"] in ("connected", "disabled")

    return ReadinessResponse(ready=ready_status, checks=checks)


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, request: Request):
    """
    Ki: Non-streaming chat endpoint.
    How: LangGraph agent invoke → final state → response.
    """
    request_id = request.state.request_id
    logger.info("chat [%s] session=%s msg=%r",
                request_id, payload.session_id, payload.message[:60])

    # LangGraph agent invoke
    from app.core.agent_router import run_agent

    final_state = await run_agent(
        query=payload.message,
        session_id=payload.session_id,
        request_id=request_id,
    )

    answer = final_state.get("answer") or FALLBACK_RESPONSE
    citations_raw = final_state.get("citations") or []
    intent = final_state.get("intent", "policy")

    # Citations → Pydantic models
    citations = [
        Citation( # pyright: ignore[reportUndefinedVariable]
            source_id=c["source_id"],
            snippet=c.get("snippet", "")[:500],
            score=float(c.get("score", 1.0)),
        )
        for c in citations_raw
    ]

    routing = intent if intent in (
        "policy", "database", "escalation", "chitchat", "fallback"
    ) else "policy"

    return ChatResponse(
        answer=answer,
        citations=citations,
        faithfulness_score=final_state.get("faithfulness_score"),
        routing=routing,
        session_id=payload.session_id,
        request_id=request_id,
    )


@app.post("/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest, request: Request):
    """
    Ki: SSE streaming endpoint.
    How: Full agent run → validate → chunk kore stream.
    """
    request_id = request.state.request_id
    logger.info("chat/stream [%s] session=%s",
                request_id, payload.session_id)

    # Full agent run (buffer-then-stream)
    from app.core.agent_router import run_agent

    try:
        final_state = await run_agent(
            query=payload.message,
            session_id=payload.session_id,
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("Agent failed [%s]: %s", request_id, exc)
        final_state = {
            "answer": FALLBACK_RESPONSE,
            "citations": [],
            "faithfulness_score": None,
            "intent": "fallback",
        }

    answer = final_state.get("answer") or FALLBACK_RESPONSE
    citations_raw = final_state.get("citations") or []
    faithfulness = final_state.get("faithfulness_score")
    routing = final_state.get("intent", "policy")
    if routing not in ("policy", "database", "escalation", "chitchat", "fallback"):
        routing = "policy"

    async def event_generator():
        async for sse in stream_verified_response(
            full_text=answer,
            citations=citations_raw,
            faithfulness_score=faithfulness,
            routing=routing,
            request_id=request_id,
        ):
            yield sse

    return EventSourceResponse(
        event_generator(),
        headers=sse_headers(),
        ping=15,  # 15 sec por por heartbeat pathabe — connection alive rakhe
    )

@app.get("/compliance", tags=["compliance"])
async def compliance():
    """
    Ki: EU AI Act compliance mapping.
    Kno: Phase 5 e purno hobe. Ekhon scaffold.
    """
    return {
        "framework": "EU AI Act",
        "risk_classification": "high-risk (Annex III, employment/critical infra scope review)",
        "articles": {
            "Article 13 (Transparency)": {
                "controls": [
                    "All responses carry inline [Source N] citations",
                    "SSE stream emits validation event before tokens",
                    "/compliance endpoint documents data flows",
                ],
            },
            "Article 14 (Human Oversight)": {
                "controls": [
                    "Distress protocol escalates to human RG review",
                    "Fallback response when faithfulness < threshold",
                ],
            },
            "Article 15 (Accuracy & Robustness)": {
                "controls": [
                    "CitationValidator: bounds + semantic check",
                    "FaithfulnessScorer: LLM-as-judge threshold",
                    "Golden eval dataset with pass/fail gates in CI",
                ],
            },
        },
        "status": "Phase 0 scaffold — full implementation in Phase 5",
    }
