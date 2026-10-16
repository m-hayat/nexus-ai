"""
Nexus-AI Platform — LLM provider factory.

Version-safe: langchain-openai / langchain-groq er version jai hok,
ei file kaj korbe. Parameter names runtime e adapt hoy.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# HELPER — filter kwargs based on what the class accepts
# ═══════════════════════════════════════════════════════════════

def _safe_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Ki: Class er __init__ signature check kore, shudhu supported kwargs rakhe.
    Kno: langchain-openai version er upor `max_tokens` vs `max_completion_tokens`
         depend kore. Ei function version drift handle kore.
    How: inspect.signature diye allowed params ber kori.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
    except (ValueError, TypeError):
        return kwargs

    # `**kwargs` accept kore mane sob allowed
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
        return kwargs

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        logger.debug("Dropped unsupported kwargs for %s: %s", cls.__name__, dropped)
    return filtered


# ═══════════════════════════════════════════════════════════════
# OLLAMA
# ═══════════════════════════════════════════════════════════════

def _build_ollama(settings: Settings) -> BaseChatModel:
    """Ollama (local) — default provider."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama install kora nei. "
            "Run: pip install langchain-ollama"
        ) from exc

    logger.info(
        "Initializing Ollama: model=%s base_url=%s",
        settings.llm_model, settings.ollama_base_url,
    )
    # ChatOllama uses num_predict for max tokens
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
    )


# ═══════════════════════════════════════════════════════════════
# OPENAI-COMPATIBLE (DeepSeek + OpenAI)
# ═══════════════════════════════════════════════════════════════

def _build_openai_compatible(
    settings: Settings,
    base_url: str | None,
    api_key: Any,
    default_model: str,
) -> BaseChatModel:
    """
    Ki: ChatOpenAI init with version-safe kwargs.
    Kno: langchain-openai different version e different param name.
         `max_tokens` (old) vs `max_completion_tokens` (new).
    How: _safe_kwargs diye filter kori.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai install kora nei. "
            "Run: pip install langchain-openai"
        ) from exc

    candidate_kwargs: dict[str, Any] = {
        "model": settings.llm_model or default_model,
        "api_key": api_key,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,              # old param name
        "max_completion_tokens": settings.llm_max_tokens,   # new param name
    }
    if base_url:
        candidate_kwargs["base_url"] = base_url

    safe = _safe_kwargs(ChatOpenAI, candidate_kwargs)

    # Ensure at least one max-token param survived
    if "max_tokens" not in safe and "max_completion_tokens" not in safe:
        safe["max_tokens"] = settings.llm_max_tokens

    logger.info(
        "Initializing ChatOpenAI: model=%s base_url=%s",
        safe.get("model"), base_url or "default",
    )
    return ChatOpenAI(**safe)


def _build_deepseek(settings: Settings) -> BaseChatModel:
    """DeepSeek API — OpenAI-compatible endpoint."""
    logger.info("Initializing DeepSeek: model=%s", settings.llm_model)
    return _build_openai_compatible(
        settings,
        base_url="https://api.deepseek.com/v1",
        api_key=settings.deepseek_api_key,
        default_model="deepseek-chat",
    )


def _build_openai(settings: Settings) -> BaseChatModel:
    """OpenAI GPT models."""
    logger.info("Initializing OpenAI: model=%s", settings.llm_model)
    return _build_openai_compatible(
        settings,
        base_url=None,  # default OpenAI endpoint
        api_key=settings.openai_api_key,
        default_model="gpt-4o-mini",
    )


# ═══════════════════════════════════════════════════════════════
# GROQ (optional — silent fallback if not installed)
# ═══════════════════════════════════════════════════════════════

def _build_groq(settings: Settings) -> BaseChatModel:
    """
    Groq — fast inference, free tier.
    Note: langchain-groq install na thakle clear error.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "langchain-groq install kora nei. "
            "Run: pip install langchain-groq"
        ) from exc

    logger.info("Initializing Groq: model=%s", settings.llm_model)

    candidate_kwargs: dict[str, Any] = {
        "model": settings.llm_model or "llama-3.3-70b-versatile",
        "api_key": settings.groq_api_key,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    safe = _safe_kwargs(ChatGroq, candidate_kwargs)

    return ChatGroq(**safe)


# ═══════════════════════════════════════════════════════════════
# REGISTRY + SINGLETON
# ═══════════════════════════════════════════════════════════════

_BUILDERS = {
    "ollama": _build_ollama,
    "deepseek": _build_deepseek,
    "openai": _build_openai,
    "groq": _build_groq,
}


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Singleton LLM factory."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider not in _BUILDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: {list(_BUILDERS.keys())}"
        )

    builder = _BUILDERS[provider]
    try:
        llm = builder(settings)
        logger.info("LLM ready: provider=%s model=%s", provider, settings.llm_model)
        return llm
    except Exception as exc:
        logger.exception("LLM init failed for provider=%s: %s", provider, exc)
        raise