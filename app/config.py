"""
Nexus-AI Platform — Resilient configuration layer.

Ei module puro application er settings load kore .env file theke.

Design principles:
- Required values missing hole fail-fast (startup e error, runtime e na)
- Optional values graceful default niye chole
- Secrets use SecretStr (log e leak korbe na)
- LLM provider switchable (ollama/deepseek/openai/groq)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Ei module er logger — warning/error ekhane jabe
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings class.

    Pydantic BaseSettings .env file automatic porbe.
    Environment variable priority beshi than .env file (12-factor principle).
    """

    # ══════════════════════════════════════════════════════════════
    # 1) APP META
    # ══════════════════════════════════════════════════════════════
    app_name: str = "Nexus-AI Platform"
    app_version: str = "0.1.0"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False        # production e False rakhben (True hole trace leak)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ══════════════════════════════════════════════════════════════
    # 2) SERVER
    # ══════════════════════════════════════════════════════════════
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # ══════════════════════════════════════════════════════════════
    # 3) LLM PROVIDER (switchable)
    # ══════════════════════════════════════════════════════════════
    llm_provider: Literal["ollama", "deepseek", "openai", "groq"] = "ollama"
    llm_model: str = "qwen3.5:4b"
    llm_temperature: float = 0.1     # 0.1 = factual, creative hole barhaben
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 60

    # Provider-specific credentials — Optional (jekono ekta use hobe)
    deepseek_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ══════════════════════════════════════════════════════════════
    # 4) VECTOR STORE (ChromaDB)
    # ══════════════════════════════════════════════════════════════
    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "betsson_policy"
    chroma_embedding_model: str = "all-MiniLM-L6-v2"

    # ══════════════════════════════════════════════════════════════
    # 5) MCP (Phase 2 te enable korben)
    # ══════════════════════════════════════════════════════════════
    mcp_enabled: bool = False
    mcp_server_command: str = "python"
    mcp_server_args: Annotated[list[str], NoDecode] = Field(default_factory=list)
    mcp_health_timeout: float = 5.0
    mcp_call_timeout: float = 10.0

    # ══════════════════════════════════════════════════════════════
    # 6) OBSERVABILITY (LangSmith)
    # ══════════════════════════════════════════════════════════════
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "nexus-ai"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ══════════════════════════════════════════════════════════════
    # 7) SAFETY / HARNESS
    # ══════════════════════════════════════════════════════════════
    citation_min_faithfulness: float = 0.85
    max_user_message_chars: int = 4000
    rate_limit_per_minute: int = 30

    # ══════════════════════════════════════════════════════════════
    # PYDANTIC CONFIG
    # ══════════════════════════════════════════════════════════════
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,     # LLM_PROVIDER == llm_provider
        extra="ignore",           # .env e extra var thakle crash korbe na
    )

       # ══════════════════════════════════════════════════════════════
    # VALIDATORS
    # ══════════════════════════════════════════════════════════════

    @field_validator("cors_origins", "mcp_server_args", mode="before")
    @classmethod
    def _parse_csv_list(cls, v):
        """CSV string → list."""
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator(
        "deepseek_api_key",
        "openai_api_key",
        "groq_api_key",
        "langsmith_api_key",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, v):
        """
        Empty string ('') ke None e convert kore.
        Kno: .env e `OPENAI_API_KEY=` (khali) likha thake — pydantic
             seta SecretStr('') banay, None na. Ei validator fix kore.
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("llm_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """Temperature 0.0-2.0 er moddhe hote hobe."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"llm_temperature must be 0.0-2.0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> Settings:
        """Fail-fast: provider je API key chay, na thakle error."""
        required_map = {
            "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "groq": ("groq_api_key", "GROQ_API_KEY"),
        }
        if self.llm_provider in required_map:
            field_name, env_name = required_map[self.llm_provider]
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"LLM_PROVIDER={self.llm_provider} set, "
                    f"kintu {env_name} missing. .env file e add korben."
                )
        return self

    # ══════════════════════════════════════════════════════════════
    # HELPER PROPERTIES
    # ══════════════════════════════════════════════════════════════

    @property
    def is_production(self) -> bool:
        """Production e specific behavior enable/disable korar jonno."""
        return self.app_env == "production"

    @property
    def active_api_key(self) -> SecretStr | None:
        """Currently active provider er API key return kore."""
        return {
            "deepseek": self.deepseek_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "ollama": None,
        }.get(self.llm_provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Ki: Singleton — ekbar load hoy, puro app lifetime cached thake.
    Kno: Bar bar .env pora lagbe na, ar env mid-run change hoy na.
    How: FastAPI dependency injection e use hobe:
         `settings = Depends(get_settings)`
    """
    try:
        settings = Settings()
        logger.info(
            "Settings loaded: env=%s provider=%s model=%s debug=%s",
            settings.app_env,
            settings.llm_provider,
            settings.llm_model,
            settings.debug,
        )
        return settings
    except Exception as exc:
        logger.critical("Settings load failed: %s", exc)
        raise


import os

def apply_langsmith_env(settings: Settings) -> None:
    """
    Ki: LangSmith tracing enable korar jonno environment variables set kore.
    Kno: LangChain internally LANGCHAIN_* env vars pore.
    How: main.py lifespan e call korbo (LangChain init er age).
    """
    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled")
        return
    
    if settings.langsmith_api_key is None:
        logger.warning("LangSmith tracing enabled but API key missing — skipping")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    logger.info("LangSmith tracing enabled: project=%s", settings.langsmith_project)