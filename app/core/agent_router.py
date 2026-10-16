"""
Nexus-AI Platform — LangGraph agent router.

StateGraph based agent pipeline:
    START → classify → [policy|escalation|chitchat] → validate → END

Design principles:
- Safety first: distress → escalation (hardcoded, no LLM)
- Citation-verified: every policy answer validated before return
- Graceful degradation: any node error → fallback response
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.documents import Document
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
# NODE: CLASSIFY
# ═══════════════════════════════════════════════════════════════

def classify_node(state: AgentState) -> dict:
    """
    Ki: User query classify kore intent e.
    Kno: Safety first — distress detect hole escalation e pathai.
    How: LLM diye classify kori, robust fallback ache.
    """
    start = time.time()
    query = state.get("query", "").strip()
    logger.info("[classify] query=%r", query[:80])

    # Fast-path: empty query
    if not query:
        return {"intent": "chitchat"}

    # Fast-path: obvious distress keywords (LLM call bachai)
    distress_keywords = [
        "suicide", "kill myself", "end my life", "self harm",
        "can't stop gambling", "cannot stop gambling",
        "losing everything", "lost everything",
    ]
    lowered = query.lower()
    if any(kw in lowered for kw in distress_keywords):
        logger.warning("[classify] distress keyword detected → escalation")
        return {"intent": "escalation"}

    # LLM classification
    try:
        llm = get_llm()
        prompt = CLASSIFY_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        intent = raw.strip().lower().split()[0].strip(".,!?\"'") # type: ignore

        # Validate
        valid_intents = {"policy", "database", "escalation", "chitchat"}
        if intent not in valid_intents:
            logger.warning("[classify] invalid intent %r, defaulting to policy", intent)
            intent = "policy"

        logger.info("[classify] intent=%s (%.2fs)", intent, time.time() - start)
        return {"intent": intent}

    except Exception as exc:
        logger.exception("[classify] failed: %s", exc)
        # Safe default — assume policy question
        return {"intent": "policy"}


# ═══════════════════════════════════════════════════════════════
# NODE: POLICY RETRIEVE
# ═══════════════════════════════════════════════════════════════

def policy_retrieve_node(state: AgentState) -> dict:
    """
    Ki: Hybrid RAG diye relevant policy chunks ber kore.
    Kno: LLM ke context hisebe ei chunks jabe.
    How: get_rag_engine() singleton use kori.
    """
    start = time.time()
    query = state.get("query", "")

    try:
        engine = get_rag_engine()
        docs = engine.retrieve(query, top_k=3)
        context = engine.format_context(docs)

        logger.info(
            "[policy_retrieve] %d docs in %.2fs",
            len(docs), time.time() - start,
        )
        return {
            "retrieved_docs": docs,
            "context": context,
        }
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

def generate_node(state: AgentState) -> dict:
    """
    Ki: LLM diye answer generate kori context theke.
    Kno: Mandatory [Source N] citation — prompt e strict rules.
    How: POLICY_ANSWER_PROMPT template use kori.
    """
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
        answer = answer.strip() # pyright: ignore[reportAttributeAccessIssue]

        logger.info("[generate] answer=%d chars (%.2fs)", len(answer), time.time() - start)
        return {"answer": answer}

    except Exception as exc:
        logger.exception("[generate] failed: %s", exc)
        return {
            "answer": FALLBACK_RESPONSE,
            "error": f"generation_failed: {exc}",
        }


# ═══════════════════════════════════════════════════════════════
# NODE: VALIDATE
# ═══════════════════════════════════════════════════════════════

def validate_node(state: AgentState) -> dict:
    """
    Ki: Answer ke validate kori — citations + faithfulness.
    Kno: Fail hole fallback response boshiye dei, raw output client e jabe na.
    """
    start = time.time()
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])
    context = state.get("context", "")

    try:
        harness = get_harness()
        result = harness.validate(answer, docs, context)

        logger.info(
            "[validate] passed=%s score=%s (%.2fs)",
            result.validation_passed,
            result.faithfulness_score,
            time.time() - start,
        )

        if not result.validation_passed:
            # Fallback response — raw unverified output replace
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
# NODE: ESCALATION (hardcoded, no LLM)
# ═══════════════════════════════════════════════════════════════

def escalation_node(state: AgentState) -> dict:
    """
    Ki: Distress response.
    Kno: Hardcoded — no LLM, karon safety critical.
    """
    logger.warning("[escalation] distress response issued")
    return {
        "answer": DISTRESS_RESPONSE,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }


# ═══════════════════════════════════════════════════════════════
# NODE: CHITCHAT (hardcoded)
# ═══════════════════════════════════════════════════════════════

def chitchat_node(state: AgentState) -> dict:
    """Ki: Greeting/off-topic response."""
    return {
        "answer": CHITCHAT_RESPONSE,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }


# ═══════════════════════════════════════════════════════════════
# NODE: DATABASE (MCP tool call)
# ═══════════════════════════════════════════════════════════════

async def database_node(state: AgentState) -> dict:
    """
    Ki: MCP tool call — player limits / self-exclusion info.
    Kno: Real DB data — LLM er bodole deterministic tool call.
    How:
      1. Query theke player_id extract
      2. Tool name decide (limits vs exclusion)
      3. MCP call
      4. Result ke natural language e format
    """
    start = time.time()
    query = state.get("query", "")
    logger.info("[database] query=%r", query[:80])

    # ── Step 1: Extract player_id ──
    from app.core.mcp_client import extract_player_id, get_mcp_client

    player_id = extract_player_id(query)
    if not player_id:
        return {
            "answer": (
                "I need a player ID to look up account details. "
                "Please provide a player ID (e.g., player_001)."
            ),
            "citations": [],
            "faithfulness_score": 1.0,
            "validation_passed": True,
        }

    # ── Step 2: Decide tool ──
    lowered = query.lower()
    if "exclud" in lowered or "self-exclusion" in lowered:
        tool_name = "check_self_exclusion"
    elif "limit" in lowered or "deposit" in lowered:
        tool_name = "get_player_limits"
    else:
        # Default — limits most common
        tool_name = "get_player_limits"

    # ── Step 3: MCP call ──
    client = get_mcp_client()
    result = await client.call_tool(tool_name, {"player_id": player_id})

    logger.info(
        "[database] tool=%s player=%s status=%s (%.2fs)",
        tool_name, player_id, result["status"], time.time() - start,
    )

    if result["status"] != "ok":
        return {
            "answer": (
                f"I couldn't retrieve the information for {player_id}. "
                f"Please try again or contact support."
            ),
            "citations": [],
            "faithfulness_score": 1.0,
            "validation_passed": True,
        }

    # ── Step 4: Format response ──
    data = result["data"]

    if tool_name == "get_player_limits":
        if "error" in data:
            answer = f"No player found with ID '{player_id}'."
        else:
            daily = data.get("daily_limit")
            weekly = data.get("weekly_limit")
            monthly = data.get("monthly_limit")
            currency = data.get("currency", "EUR")

            # Handle None limits (player with no limits set)
            if daily is None and weekly is None and monthly is None:
                answer = (
                    f"Player {player_id} has no deposit limits configured. "
                    f"They can set daily, weekly, or monthly limits "
                    f"in their account settings at any time."
                )
            else:
                answer = (
                    f"Player {player_id} has the following deposit limits: "
                    f"Daily: {currency} {daily}, "
                    f"Weekly: {currency} {weekly}, "
                    f"Monthly: {currency} {monthly}."
                )
    elif tool_name == "check_self_exclusion":
        if "error" in data:
            answer = f"No player found with ID '{player_id}'."
        else:
            excluded = data.get("excluded", False)
            until = data.get("exclusion_until")
            if excluded:
                answer = (
                    f"Player {player_id} is currently self-excluded"
                    + (f" until {until}." if until else ".")
                )
            else:
                answer = (
                    f"Player {player_id} is not currently self-excluded."
                )
    else:
        answer = f"Retrieved data for {player_id}."

    return {
        "answer": answer,
        "citations": [],
        "faithfulness_score": 1.0,
        "validation_passed": True,
    }
# ═══════════════════════════════════════════════════════════════
# ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> Literal[
    "policy_retrieve", "escalation", "chitchat", "database"
]:
    """
    Ki: Intent onujayi next node decide.
    Kno: Conditional edge — LangGraph ei function use kore.
    """
    intent = state.get("intent", "policy")
    if intent == "escalation":
        return "escalation"
    if intent == "chitchat":
        return "chitchat"
    if intent == "database":
        return "database"
    return "policy_retrieve"

# ═══════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════

def build_agent_graph():
    """
    Ki: LangGraph StateGraph build kori.
    Kno: Node + edge define — compile korle callable agent.
    How: Singleton pattern (get_agent()).
    """
    graph = StateGraph(AgentState)

    # Nodes add
    graph.add_node("classify", classify_node)
    graph.add_node("policy_retrieve", policy_retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("database", database_node)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "policy_retrieve": "policy_retrieve",
            "escalation": "escalation",
            "chitchat": "chitchat",
            "database": "database",       # ← changed: real node
        },
    )

    # Policy path: retrieve → generate → validate → END
    graph.add_edge("policy_retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)

    # Hardcoded paths: → END
    graph.add_edge("escalation", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("database", END)

    return graph.compile()


# ── Singleton ──
_agent = None


def get_agent():
    """Singleton LangGraph agent."""
    global _agent
    if _agent is None:
        logger.info("Building LangGraph agent...")
        _agent = build_agent_graph()
        logger.info("LangGraph agent ready")
    return _agent


async def run_agent(
    query: str,
    session_id: str,
    request_id: str,
) -> AgentState:
    """
    Ki: Agent invoke kori, full state return.
    Kno: main.py ei function call korbe.
    How: initial state → ainvoke → final state.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "request_id": request_id,
        "error": None,
    }

    agent = get_agent()
    try:
        final_state = await agent.ainvoke(initial_state)
        return final_state # pyright: ignore[reportReturnType]
    except Exception as exc:
        logger.exception("Agent invoke failed: %s", exc)
        return {
            **initial_state,
            "answer": FALLBACK_RESPONSE,
            "citations": [],
            "validation_passed": False,
            "error": str(exc),
        }