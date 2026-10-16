"""
Nexus-AI Platform — Persistent MCP client.

MCP 2.x compatible. AsyncExitStack-based lifecycle.
Stdio subprocess ekbar spawn hoy, puro app lifetime reuse hoy.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Player ID extraction from user query
PLAYER_ID_QUERY_PATTERN = re.compile(
    r"\b(player[_-]?[a-zA-Z0-9]{2,32})\b",
    re.IGNORECASE,
)


class MCPClient:
    """
    Persistent MCP client wrapper.

    Lifecycle:
      - connect() → FastAPI startup, ekbar
      - call_tool() → per-request
      - disconnect() → FastAPI shutdown, ekbar

    Errors are handled gracefully — connect fail hole app cholbe
    degraded mode e (MCP unavailable).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stack: AsyncExitStack | None = None
        self._session: Any = None   # mcp.ClientSession
        self._connected = False

    # ═══════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    async def connect(self) -> None:
        """
        Ki: MCP server subprocess spawn + ClientSession initialize.
        Kno: Lifespan e ekbar — persistent connection reuse.
        How: AsyncExitStack diye stdio_client + ClientSession manage.
        """
        if not self.settings.mcp_enabled:
            logger.info("MCP disabled (MCP_ENABLED=false) — skipping connect")
            return

        try:
            # Late import — MCP disabled hole ei import o avoid kori
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            self._stack = AsyncExitStack()

            params = StdioServerParameters(
                command=self.settings.mcp_server_command,
                args=self.settings.mcp_server_args,
            )
            logger.info(
                "MCP connecting: command=%s args=%s",
                self.settings.mcp_server_command,
                self.settings.mcp_server_args,
            )

            read_stream, write_stream = await self._stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            self._connected = True
            logger.info("MCP client connected")

        except Exception as exc:
            logger.exception("MCP connect failed: %s", exc)
            self._connected = False
            # Cleanup partial state
            if self._stack is not None:
                try:
                    await self._stack.aclose()
                except Exception:
                    pass
                self._stack = None

    async def disconnect(self) -> None:
        """
        Ki: Gracefully close subprocess + session.
        Kno: Lifespan shutdown e ekbar — clean process exit.
        """
        if self._stack is not None:
            try:
                await self._stack.aclose()
                logger.info("MCP client disconnected")
            except Exception as exc:
                logger.warning("MCP disconnect error: %s", exc)
            finally:
                self._stack = None
                self._session = None
                self._connected = False

    # ═══════════════════════════════════════════════════════════
    # TOOL CALL
    # ═══════════════════════════════════════════════════════════

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        # ... docstring ...

        # ── Input sanitization first (always validate input) ──
        sanitized_args = self._sanitize_arguments(tool_name, arguments)
        if sanitized_args is None:
            return {
                "status": "error",
                "tool": tool_name,
                "data": None,
                "message": "Invalid arguments.",
            }

        # ── Connection check ──
        if not self._connected or self._session is None:
            return {
                "status": "unavailable",
                "tool": tool_name,
                "data": None,
                "message": "MCP service unavailable.",
            }

        # ── Input sanitization (defense in depth) ──
        sanitized_args = self._sanitize_arguments(tool_name, arguments)
        if sanitized_args is None:
            return {
                "status": "error",
                "tool": tool_name,
                "data": None,
                "message": "Invalid arguments.",
            }

        # ── Call with timeout ──
        import asyncio
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, sanitized_args),
                timeout=self.settings.mcp_call_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("MCP call timeout: tool=%s", tool_name)
            return {
                "status": "timeout",
                "tool": tool_name,
                "data": None,
                "message": f"Tool call timed out after {self.settings.mcp_call_timeout}s.",
            }
        except Exception as exc:
            logger.exception("MCP call failed: tool=%s err=%s", tool_name, exc)
            return {
                "status": "error",
                "tool": tool_name,
                "data": None,
                "message": "Tool call failed.",
            }

        # ── Parse result ──
        parsed = self._parse_result(result, tool_name)
        logger.info("MCP call done: tool=%s status=%s", tool_name, parsed["status"])
        return parsed

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    def _sanitize_arguments(self, tool_name: str, arguments: dict) -> dict | None:
        """
        Ki: Tool-specific input validation.
        Kno: LLM generate kora args world untrusted — always validate.
        """
        if tool_name in ("get_player_limits", "check_self_exclusion"):
            player_id = arguments.get("player_id")
            if not player_id or not isinstance(player_id, str):
                return None
            cleaned = player_id.strip()
            # Same pattern as server side
            if not re.match(r"^[a-zA-Z0-9_-]{3,32}$", cleaned):
                return None
            return {"player_id": cleaned}
        # Unknown tool — pass through
        return arguments

    def _parse_result(self, result: Any, tool_name: str) -> dict:
        """
        Ki: MCP CallToolResult theke structured data extract kore.
        Kno: MCP 2.x e content blocks list hisebe ashe (TextContent).
        How:
          1. is_error check (MCP 2.x snake_case)
          2. Text content extract
          3. JSON parse (amader tool gulo JSON string return kore)
        """
        # MCP 2.x field names (snake_case)
        is_error = getattr(result, "is_error", False) or getattr(result, "isError", False)

        content_blocks = getattr(result, "content", []) or []
        text_parts: list[str] = []
        for block in content_blocks:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)

        raw_text = "\n".join(text_parts) if text_parts else ""

        # Try JSON parse
        try:
            data = json.loads(raw_text) if raw_text else {}
        except (json.JSONDecodeError, TypeError):
            data = {"raw": raw_text}

        if is_error:
            return {
                "status": "error",
                "tool": tool_name,
                "data": data,
                "message": data.get("error", "Tool returned error"),
            }

        return {
            "status": "ok",
            "tool": tool_name,
            "data": data,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """
    Ki: Module-level singleton — app lifetime e same client.
    Kno: main.py lifespan init kore, agent_router get kore.
    """
    global _client
    if _client is None:
        _client = MCPClient(get_settings())
    return _client


async def init_mcp_client(settings: Settings) -> MCPClient:
    """
    Ki: Factory + connect — lifespan e ekbar call.
    Kno: Client create kore, connect kore return kore.
    """
    global _client
    _client = MCPClient(settings)
    await _client.connect()
    return _client


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def extract_player_id(query: str | None) -> str | None:
    """
    Ki: User query theke player ID extract kori.
    Kno: Regex pattern — "player_001", "player-abc", "playerxyz".
    How: Case-insensitive, word boundaries.
    Note: query None hole defensive return None.
    """
    if not query:
        return None
    match = PLAYER_ID_QUERY_PATTERN.search(query)
    if match:
        return match.group(1)
    return None