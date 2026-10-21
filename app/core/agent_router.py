"""
Nexus-AI Platform — LangGraph agent router.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from app.core.ai_harness import get_harness
from app.core.llm_factory import get_llm
from app.core.prompts import (
    CHITCHAT_RESPONSE,
    CLASSIFY_PROMPT,
    DISTRESS_RESPONSE,
    FALLBACK_RESPONSE,
    POLICY_ANSWER_PROMPT,
)
from app.core.rag_engine import get_rag_engine
from app.schemas.payload_schemas import AgentState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# OBSERVABILITY DECORATOR
# ═══════════════════════════════════════════════════════════════

def log_node_latency(node_name: str) -> Callable:
    """Prottek node er execution time log korar decorator."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[LATENCY] %s: %.3fs", node_name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception("[LATENCY] %s FAILED: %.3fs", node_name, elapsed)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("[LATENCY] %s: %.3fs", node_name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception("[LATENCY] %s FAILED: %.3fs", node_name, elapsed)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# NODE: CLASSIFY
# ═══════════════════════════════════════════════════════════════

@log_node_latency("classify")
def classify_node(state: AgentState) -> dict:
    """User query classify kore intent e."""
    start = time.time()
    query = state.get("query", "").strip()
    logger.info("[classify] query=%r", query[:80])

    if not query:
        return {"intent": "chitchat"}

    distress_keywords = [
        "suicide", "kill myself", "end my life", "self harm",
        "can't stop gambling", "cannot stop gambling",
        "losing everything", "lost everything",
    ]
    lowered = query.lower()
    if any(kw in lowered for kw in distress_keywords):
        logger.warning("[classify] distress keyword detected → escalation")
        return {"intent": "escalation"}

    try:
        llm = get_llm()
        prompt = CLASSIFY_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        
        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            raw = str(raw)
        
        intent = raw.strip().lower().split()[0].strip(".,!?\"'") 

        valid_intents = {"policy", "database", "escalation", "chitchat"}
        if intent not in valid_intents:
            logger.warning("[classify] invalid intent %r, defaulting to policy", intent)
            intent = "policy"

        logger.info("[classify] intent=%s (%.2fs)", intent, time.time() - start)
        return {"intent": intent}

    except Exception as exc:
        logger.exception("[classify] failed: %s", exc)
        return {"intent": "policy"}


# ═══════════════════════════════════════════════════════════════
# NODE: POLICY RETRIEVE
# ═══════════════════════════════════════════════════════════════

@log_node_latency("policy_retrieve")
def policy_retrieve_node(state: AgentState) -> dict:
    """Hybrid RAG diye relevant policy chunks ber kore."""
    start = time.time()
    query = state.get("query", "")

    try:
        engine = get_rag_engine()
        docs = engine.retrieve(query, top_k=3)
        context = engine.format_context(docs)

        logger.info("[policy_retrieve] %d docs in %.2fs", len(docs), time.time() - start)
        return {"retrieved_docs": docs, "context": context}
    except Exception as exc:
        logger.exception("[policy_retrieve] failed: %s", exc)
        return {
            "retrieved_docs": [],
            "context": "(no relevant context found)",
            "error": f"retrieval_failed: {exc}",
        }


# ═══════════════════════════════════════════════════════════════
# NODE: GENERATE
# ═══════════════════════════════════════════════════════════════

@log_node_latency("generate")
def generate_node(state: AgentState) -> dict:
    """LLM diye answer generate kori context theke."""
    start = time.time()
    query = state.get("query", "")
    context = state.get("context", "")

    if not context or context == "(no relevant context found)":
        logger.warning("[generate] no context → refusal")
        return {
            "answer": "I don't have that information in the Betsson policy documents.",
            "citations": [],
        }

    try:
        llm = get_llm()
        prompt = POLICY_ANSWER_PROMPT.format(context=context, query=query)
        response = llm.invoke(prompt)
        
        answer = response.content if hasattr(response, "content") else str(response)
        if isinstance(answer, list):
            answer = str(answer)
        answer = answer.strip()

        logger.info("[generate] answer=%d chars (%.2fs)", len(answer), time.time() - start)
        return {"answer": answer}

    except Exception as exc:
        logger.exception("[generate] failed: %s", exc)
        return {"answer": FALLBACK_RESPONSE, "error": f"generation_failed: {exc}"}


# ═══════════════════════════════════════════════════════════════
# NODE: VALIDATE
# ═══════════════════════════════════════════════════════════════

@log_node_latency("validate")
def validate_node(state: AgentState) -> dict:
    """Answer ke validate kori — citations + faithfulness."""
    start = time.time()
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    context = state.get("context", "")

    try:
        harness = get_harness()
        result = harness.validate(answer, docs, context)

        logger.info(
            "[validate] passed=%s score=%s (%.2fs)",
            result.validation_passed, result.faithfulness_score, time.time() - start,
        )

        if not result.validation_passed:
            logger.warning("[validate] FAILED: %s", result.failure_reason)
            return {
                "answer": FALLBACK_RESPONSE,
                "citations": [],
                "invalid_citations": result.invalid_citations,
                "faithfulness_score": result.faithfulness_score,
                "validation_passed": False,
            }

        return {
            "citations": result.valid_citations,
            "invalid_citations": result.invalid_citations,
            "faithfulness_score": result.faithfulness_score,
            "validation_passed": True,
        }

    except Exception as exc:
        logger.exception("[validate] failed: %s", exc)
        return {
            "answer": FALLBACK_RESPONSE,
            "citations": [],
            "validation_passed": False,
            "error": f"validation_failed: {exc}",
        }


# ═══════════════════════════════════════════════════════════════
# NODE: ESCALATION & CHITCHAT
# ═══════════════════════════════════════════════════════════════

def escalation_node(state: AgentState) -> dict:
    logger.warning("[escalation] distress response issued")
    return {
        "answer": DISTRESS_RESPONSE,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }

def chitchat_node(state: AgentState) -> dict:
    return {
        "answer": CHITCHAT_RESPONSE,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }


# ═══════════════════════════════════════════════════════════════
# NODE: DATABASE
# ═══════════════════════════════════════════════════════════════

@log_node_latency("database")
async def database_node(state: AgentState) -> dict:
    """MCP tool call — player limits / self-exclusion info."""
    start = time.time()
    query = state.get("query", "")
    logger.info("[database] query=%r", query[:80])

    from app.core.mcp_client import extract_player_id, get_mcp_client

    player_id = extract_player_id(query)
    if not player_id:
        return {
            "answer": "I need a player ID to look up account details. Please provide a player ID (e.g., player_001).",
            "citations": [],
            "faithfulness_score": 1.0,
            "validation_passed": True,
        }

    lowered = query.lower()
    if "exclud" in lowered or "self-exclusion" in lowered:
        tool_name = "check_self_exclusion"
    elif "limit" in lowered or "deposit" in lowered:
        tool_name = "get_player_limits"
    else:
        tool_name = "get_player_limits"

    client = get_mcp_client()
    result = await client.call_tool(tool_name, {"player_id": player_id})

    logger.info(
        "[database] tool=%s player=%s status=%s (%.2fs)",
        tool_name, player_id, result["status"], time.time() - start,
    )

    if result["status"] != "ok":
        return {
            "answer": f"I couldn't retrieve the information for {player_id}. Please try again or contact support.",
            "citations": [],
            "faithfulness_score": 1.0,
            "validation_passed": True,
        }

    data = result["data"]

    if tool_name == "get_player_limits":
        if "error" in data:
            answer = f"No player found with ID '{player_id}'."
        else:
            daily = data.get("daily_limit")
            weekly = data.get("weekly_limit")
            monthly = data.get("monthly_limit")
            currency = data.get("currency", "EUR")

            if daily is None and weekly is None and monthly is None:
                answer = (
                    f"Player {player_id} has no deposit limits configured. "
                    f"They can set daily, weekly, or monthly limits in their account settings at any time."
                )
            else:
                answer = (
                    f"Player {player_id} has the following deposit limits: "
                    f"Daily: {currency} {daily}, Weekly: {currency} {weekly}, Monthly: {currency} {monthly}."
                )
    elif tool_name == "check_self_exclusion":
        if "error" in data:
            answer = f"No player found with ID '{player_id}'."
        else:
            excluded = data.get("excluded", False)
            until = data.get("exclusion_until")
            if excluded:
                answer = f"Player {player_id} is currently self-excluded" + (f" until {until}." if until else ".")
            else:
                answer = f"Player {player_id} is not currently self-excluded."
    else:
        answer = f"Retrieved data for {player_id}."

    return {
        "answer": answer,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }


# ═══════════════════════════════════════════════════════════════
# ROUTING & GRAPH
# ═══════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> Literal["policy_retrieve", "escalation", "chitchat", "database"]:
    intent = state.get("intent", "policy")
    if intent == "escalation": return "escalation"
    if intent == "chitchat": return "chitchat"
    if intent == "database": return "database"
    return "policy_retrieve"

def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("policy_retrieve", policy_retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("database", database_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "policy_retrieve": "policy_retrieve",
            "escalation": "escalation",
            "chitchat": "chitchat",
            "database": "database",
        },
    )
    graph.add_edge("policy_retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    graph.add_edge("escalation", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("database", END)
    return graph.compile()

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        logger.info("Building LangGraph agent...")
        _agent = build_agent_graph()
        logger.info("LangGraph agent ready")
    return _agent

async def run_agent(query: str, session_id: str, request_id: str) -> AgentState:
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "request_id": request_id,
        "error": None,
    }
    agent = get_agent()
    try:
        final_state = await agent.ainvoke(initial_state)
        return cast(AgentState, final_state)
    except Exception as exc:
        logger.exception("Agent invoke failed: %s", exc)
        return {
            **initial_state,
            "answer": FALLBACK_RESPONSE,
            "citations": [],
            "validation_passed": False,
            "error": str(exc),
        }