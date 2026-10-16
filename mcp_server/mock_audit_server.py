"""
Nexus-AI Platform — Mock MCP Audit Server.

MCPServer-based (MCP 2.x compatible).
In-memory mock for player audit tools.

Run:
    python -m mcp_server.mock_audit_server

Tools:
    - get_player_limits(player_id) → deposit limits
    - check_self_exclusion(player_id) → exclusion status
"""
from __future__ import annotations

import json
import logging
import re
import sys

from mcp.server.mcpserver import MCPServer  # ← FIX 1

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

PLAYER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,32}$")


MOCK_PLAYERS: dict[str, dict] = {
    "player_001": {
        "daily_limit": 500.0,
        "weekly_limit": 2000.0,
        "monthly_limit": 5000.0,
        "currency": "EUR",
        "excluded": False,
        "exclusion_until": None,
    },
    "player_002": {
        "daily_limit": 100.0,
        "weekly_limit": 500.0,
        "monthly_limit": 1500.0,
        "currency": "EUR",
        "excluded": True,
        "exclusion_until": "2026-06-01T00:00:00Z",
    },
    "player_999": {
        "daily_limit": None,
        "weekly_limit": None,
        "monthly_limit": None,
        "currency": "EUR",
        "excluded": False,
        "exclusion_until": None,
    },
}


def _sanitize_player_id(player_id: str) -> str | None:
    """Input validation — injection prevent."""
    if not player_id or not isinstance(player_id, str):
        return None
    cleaned = player_id.strip()
    if not PLAYER_ID_PATTERN.match(cleaned):
        return None
    return cleaned


# ═══════════════════════════════════════════════════════════════
# MCPServer INSTANCE
# ═══════════════════════════════════════════════════════════════

mcp = MCPServer("nexus-audit-server")  # ← FIX 2


@mcp.tool()
def get_player_limits(player_id: str) -> str:
    """
    Get deposit limits for a specific Betsson player.

    Args:
        player_id: Betsson player ID (alphanumeric, 3-32 chars)
    """
    logger.info("get_player_limits: player_id=%r", player_id)

    sanitized = _sanitize_player_id(player_id)
    if sanitized is None:
        return json.dumps({
            "error": "invalid_player_id",
            "message": "player_id must be alphanumeric (3-32 chars)",
            "received": player_id,
        })

    player = MOCK_PLAYERS.get(sanitized)
    if player is None:
        return json.dumps({
            "error": "player_not_found",
            "player_id": sanitized,
        })

    return json.dumps({
        "player_id": sanitized,
        "daily_limit": player["daily_limit"],
        "weekly_limit": player["weekly_limit"],
        "monthly_limit": player["monthly_limit"],
        "currency": player["currency"],
    })


@mcp.tool()
def check_self_exclusion(player_id: str) -> str:
    """
    Check if a player is currently under self-exclusion.

    Args:
        player_id: Betsson player ID (alphanumeric, 3-32 chars)
    """
    logger.info("check_self_exclusion: player_id=%r", player_id)

    sanitized = _sanitize_player_id(player_id)
    if sanitized is None:
        return json.dumps({
            "error": "invalid_player_id",
            "message": "player_id must be alphanumeric (3-32 chars)",
            "received": player_id,
        })

    player = MOCK_PLAYERS.get(sanitized)
    if player is None:
        return json.dumps({
            "error": "player_not_found",
            "player_id": sanitized,
        })

    return json.dumps({
        "player_id": sanitized,
        "excluded": player["excluded"],
        "exclusion_until": player["exclusion_until"],
    })


if __name__ == "__main__":
    mcp.run()