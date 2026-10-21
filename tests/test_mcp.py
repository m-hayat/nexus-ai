"""
Nexus-AI Platform — MCP client tests.

Unit tests for input sanitization and player_id extraction.
Integration tests for MCP tool calls (need mock server).
"""
from __future__ import annotations

import pytest

from app.core.mcp_client import extract_player_id

# ═══════════════════════════════════════════════════════════════
# PLAYER ID EXTRACTION (no server needed)
# ═══════════════════════════════════════════════════════════════

def test_extract_player_id_standard():
    assert extract_player_id("What is player_001 limit?") == "player_001"


def test_extract_player_id_dash():
    assert extract_player_id("Check player-abc123") == "player-abc123"


def test_extract_player_id_no_underscore():
    assert extract_player_id("playerxyz123 status") == "playerxyz123"


def test_extract_player_id_case_insensitive():
    assert extract_player_id("PLAYER_001 balance") == "PLAYER_001"


def test_extract_player_id_not_found():
    assert extract_player_id("What is the deposit limit?") is None


def test_extract_player_id_empty():
    assert extract_player_id("") is None
    assert extract_player_id(None) is None  # defensive


def test_extract_player_id_multiple_picks_first():
    result = extract_player_id("Compare player_001 and player_002")
    assert result == "player_001"


# ═══════════════════════════════════════════════════════════════
# MCP CLIENT (no server — degraded mode)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_mcp_client_degraded_when_not_connected():
    """Not connected → call returns unavailable status."""
    from app.config import Settings
    from app.core.mcp_client import MCPClient

    settings = Settings(_env_file=None, mcp_enabled=False) # type: ignore[call-arg]
    client = MCPClient(settings)
    # Not connected — call should return degraded response
    result = await client.call_tool("get_player_limits", {"player_id": "player_001"})
    assert result["status"] == "unavailable"
    assert result["data"] is None


@pytest.mark.asyncio
async def test_mcp_client_sanitizes_invalid_player_id():
    """Invalid player_id → error before any network call."""
    from app.config import Settings
    from app.core.mcp_client import MCPClient

    settings = Settings(_env_file=None, mcp_enabled=False) # type: ignore[call-arg]
    client = MCPClient(settings)
    client._connected = True  # force connected flag
    client._session = None    # but no session

    result = await client.call_tool(
        "get_player_limits",
        {"player_id": "bad id with spaces!!"},
    )
    # Sanitization should catch this before session check
    assert result["status"] == "error"


# ═══════════════════════════════════════════════════════════════
# INTEGRATION (optional — needs running MCP server)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires running MCP server — enable manually")
async def test_mcp_integration_player_limits():
    """Integration test — actual MCP subprocess spawn."""
    from app.config import get_settings
    from app.core.mcp_client import MCPClient

    settings = get_settings()
    settings.mcp_enabled = True
    client = MCPClient(settings)
    await client.connect()

    try:
        result = await client.call_tool(
            "get_player_limits",
            {"player_id": "player_001"},
        )
        assert result["status"] == "ok"
        assert result["data"]["daily_limit"] == 500.0
    finally:
        await client.disconnect()
