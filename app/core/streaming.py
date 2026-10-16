"""
Nexus-AI Platform — SSE streaming utilities.

Buffer-then-stream strategy:
- Full LLM response generate hoy
- Citation validator + faithfulness scorer chole
- Jodi pass hoy → chunk kore SSE events pathai
- Jodi fail hoy → fallback message pathai, raw output kokhono show korbo na
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Literal

logger = logging.getLogger(__name__)

EventType = Literal["token", "citation", "validation", "error", "done"]


def format_sse(event: EventType, data) -> str:
    """
    Ki: Ekta SSE event ke string e convert kore.
    Kno: SSE protocol er format: 'event: <name>\\ndata: <json>\\n\\n'.
         Sesh e blank line (\\n\\n) must — nahole client parse korte parbe na.
    How: sse-starlette EventSourceResponse ei string yield korbe.
    """
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_text_chunks(
    text: str,
    chunk_size: int = 8,
    delay: float = 0.0,
) -> AsyncIterator[str]:
    """
    Ki: Text ke chunk kore yield kore (generator).
    Kno: Instant puro text pathale "stream" feel hoy na. Chunk kore
         pathale user ke typing effect dekhay (better UX).
    How: SSE token event hisebe client e jabe.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        if delay > 0:
            await asyncio.sleep(delay)


async def stream_verified_response(
    full_text: str,
    citations: list[dict] | None = None,
    faithfulness_score: float | None = None,
    routing: str = "policy",
    request_id: str = "",
    chunk_size: int = 20,   # ← 8 theke 20
) -> AsyncIterator[dict]:
    """
    Ki: Full pipeline — validation, citation, token, done.
    Kno: Client ke ekta well-defined sequence dite hobe.
    How: sse-starlette EventSourceResponse ei generator ke use korbe.
         Sob dict yield korchi — sse-starlette nijei SSE format korbe.
         format_sse() use korle double-wrapping hoy.
    """
    try:
        # Step 1: validation event
        yield {
            "event": "validation",
            "data": json.dumps({
                "faithfulness_score": faithfulness_score,
                "routing": routing,
                "request_id": request_id,
            }),
        }

        # Step 2: citation events
        for citation in (citations or []):
            yield {
                "event": "citation",
                "data": json.dumps(citation),
            }

        # Step 3: token chunks
        async for chunk in stream_text_chunks(full_text, chunk_size=chunk_size):
            yield {"event": "token", "data": chunk}

        # Step 4: done event
        yield {
            "event": "done",
            "data": json.dumps({
                "request_id": request_id,
                "routing": routing,
                "citation_count": len(citations or []),
            }),
        }

    except Exception as exc:
        logger.exception("Streaming error: %s", exc)
        yield {
            "event": "error",
            "data": json.dumps({
                "error": "streaming_failed",
                "detail": "Response generation interrupted. Please retry.",
                "request_id": request_id,
            }),
        }
        yield {
            "event": "done",
            "data": json.dumps({"request_id": request_id, "routing": "error"}),
        }


def sse_headers() -> dict[str, str]:
    """
    Ki: SSE response er jonno proper headers.
    Kno: Reverse proxy (nginx, cloudflare) buffering off korte hobe,
         nahole events guchhiye chole ashe.
    """
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",   # nginx specific
    }