"""
Nexus-AI Platform — pytest root conftest.

Ei file er presence e pytest project root ke sys.path e add kore,
tai `from main import app` kaj kore.
"""
import sys
from pathlib import Path

# Project root ke sys.path e add kori
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))