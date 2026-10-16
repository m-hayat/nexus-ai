"""
Nexus-AI Platform — Production MCP Audit Server.

Ki: Real audit server — ei file e real DB/API call hobe.
Kno: Mock theke real e migrate korar jonno ei placeholder rakchi.
Status: Phase 2 e scaffold. Real DB integration Phase 5+ e.

Run:
    python -m mcp_server.audit_server
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Real server entry point.

    Phase 5+ e ei jaygay:
      - Real DB connection (PostgreSQL, MySQL, etc.)
      - Auth layer (service account, API key)
      - Rate limiting per caller
      - Audit logging
    """
    logger.warning(
        "audit_server.py — production server scaffold. "
        "Real DB integration Phase 5+ e. "
        "Current setup mock_audit_server.py use kore."
    )
    raise NotImplementedError(
        "Production audit server not yet implemented. "
        "Use mcp_server.mock_audit_server for development."
    )


if __name__ == "__main__":
    main()