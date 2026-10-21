"""
Nexus-AI Platform — Seed ChromaDB with policy documents.

Manual run:
    python scripts/seed_chroma.py

Verifies:
    - Policy file exists
    - Chunking works
    - Chroma persistence works
    - Retrieval works with sample queries
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root ke sys.path e add kori (direct script run er jonno)
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.core.rag_engine import HybridRAGEngine  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("Nexus-AI — Seeding ChromaDB")
    print("=" * 60)

    settings = get_settings()
    engine = HybridRAGEngine(settings)

    try:
        engine.initialize()
    except Exception as exc:
        print(f"[FAIL] Init failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"[OK] Indexed {engine.document_count} chunks")
    print(f"[OK] Persist dir: {settings.chroma_persist_dir}")
    print()

    # ── Sample queries for verification ──
    queries = [
        "What is the self-exclusion period?",
        "How much is the bonus wagering requirement?",
        "When do withdrawals process?",
        "What happens if a player shows distress signals?",
    ]

    for q in queries:
        print(f"Query: {q}")
        docs = engine.retrieve(q, top_k=2)
        if not docs:
            print("  -> (no results)")
            print()
            continue
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        print()

    print("=" * 60)
    print("[DONE] Seeding complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
