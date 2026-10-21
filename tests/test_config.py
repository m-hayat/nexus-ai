"""
Nexus-AI Platform — config layer tests.

Verifies:
- Settings load correctly with defaults
- Fail-fast on missing API keys
- Empty string → None conversion
- CSV parsing
- Temperature bounds
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Prottek test er age lru_cache clear — fresh settings."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════
# TESTS: DEFAULTS
# ═══════════════════════════════════════════════════════════════

def test_default_provider_is_ollama():
    """Default provider ollama — kono API key lage na."""
    s = Settings(_env_file=None)  # ignore .env file
    assert s.llm_provider == "ollama"
    assert s.active_api_key is None


def test_default_app_env_is_development():
    s = Settings(_env_file=None)
    assert s.app_env == "development"
    assert s.is_production is False


def test_debug_defaults_to_false():
    """Security: debug default False — production e vul e True hoy na."""
    s = Settings(_env_file=None)
    assert s.debug is False


# ═══════════════════════════════════════════════════════════════
# TESTS: FAIL-FAST (CRITICAL)
# ═══════════════════════════════════════════════════════════════

def test_deepseek_without_key_fails():
    """LLM_PROVIDER=deepseek kintu key nei → ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            llm_provider="deepseek",
            deepseek_api_key=None,
        )
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)


def test_openai_without_key_fails():
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            llm_provider="openai",
            openai_api_key=None,
        )
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_groq_without_key_fails():
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            llm_provider="groq",
            groq_api_key=None,
        )
    assert "GROQ_API_KEY" in str(exc_info.value)


def test_deepseek_with_key_succeeds():
    s = Settings(
        _env_file=None,
        llm_provider="deepseek",
        deepseek_api_key="sk-test123",
    )
    assert s.llm_provider == "deepseek"
    assert s.active_api_key is not None


# ═══════════════════════════════════════════════════════════════
# TESTS: EMPTY STRING → NONE BUG FIX
# ═══════════════════════════════════════════════════════════════

def test_empty_api_key_becomes_none():
    """
    Empty string '' → None.
    Regression test: pydantic SecretStr('') banay, None na.
    Ei bug fix chara fail-fast miss hoy.
    """
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_provider="openai",
            openai_api_key="",  # empty string
        )


def test_whitespace_api_key_becomes_none():
    """Whitespace '   ' o None hoy."""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_provider="openai",
            openai_api_key="   ",
        )


# ═══════════════════════════════════════════════════════════════
# TESTS: VALIDATORS
# ═══════════════════════════════════════════════════════════════

def test_temperature_bounds_valid():
    s = Settings(_env_file=None, llm_temperature=0.0)
    assert s.llm_temperature == 0.0
    s = Settings(_env_file=None, llm_temperature=2.0)
    assert s.llm_temperature == 2.0


def test_temperature_above_max_fails():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_temperature=3.0)


def test_temperature_below_min_fails():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_temperature=-0.5)


def test_csv_origins_parsing():
    """'a,b,c' → ['a', 'b', 'c']"""
    s = Settings(_env_file=None, cors_origins="http://a.com,http://b.com")
    assert s.cors_origins == ["http://a.com", "http://b.com"]


def test_csv_origins_empty():
    s = Settings(_env_file=None, cors_origins="")
    assert s.cors_origins == []


# ═══════════════════════════════════════════════════════════════
# TESTS: SECRETS
# ═══════════════════════════════════════════════════════════════

def test_secret_str_not_leaked_in_repr():
    """SecretStr log/repr e value leak kore na."""
    s = Settings(
        _env_file=None,
        llm_provider="deepseek",
        deepseek_api_key="sk-super-secret",
    )
    assert "sk-super-secret" not in repr(s)


# ═══════════════════════════════════════════════════════════════
# TESTS: SINGLETON
# ═══════════════════════════════════════════════════════════════

def test_get_settings_returns_cached_instance():
    """lru_cache — same object return kore."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ═══════════════════════════════════════════════════════════════
# TESTS: ENV OVERRIDE (12-FACTOR)
# ═══════════════════════════════════════════════════════════════

def test_env_var_overrides_defaults():
    """Environment variable .env ke override kore."""
    with patch.dict(os.environ, {"LLM_MODEL": "qwen-test"}, clear=False):
        s = Settings(_env_file=None)
        assert s.llm_model == "qwen-test"
