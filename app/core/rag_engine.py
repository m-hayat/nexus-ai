"""
Nexus-AI Platform — Hybrid RAG engine.

Custom hybrid retriever: BM25 (keyword) + Chroma (semantic) with RRF fusion.
No dependency on langchain.retrievers.EnsembleRetriever — version-safe.

Dependencies:
    pip install rank-bm25 langchain-community langchain-chroma langchain-text-splitters
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

# ── Text splitter ──
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Chroma ──
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# ── BM25Retriever ──
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain_community.retrievers.bm25 import BM25Retriever

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION (RRF)
# ═══════════════════════════════════════════════════════════════

def _rrf_fuse(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 3,
) -> list[Document]:
    """
    Ki: Reciprocal Rank Fusion — multiple ranked lists ke ek list e merge kore.

    Formula: score(doc) = Σ  1 / (k + rank_i)

    Kno: EnsembleRetriever er bodole nijer implementation — version-independent.
         RRF standard technique, Elasticsearch/OpenSearch/Vespa sob use kore.

    How:
      - Prottek list e doc er rank dekhi
      - RRF score calculate kori
      - Top-N highest score return kori
      - Duplicate doc (same content) merge — first occurrence er rank use

    Args:
        ranked_lists: list of ranked Document lists (BM25 result, Chroma result)
        k: RRF constant (usually 60) — lower k = top rank more weight
        top_n: koto doc return korbo

    Returns:
        Top-N fused Document list
    """
    # Content hash ke unique key hisebe use kori (Document object identity na)
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content  # content-based dedup
            rrf_score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score
            # First occurrence er doc object rakhi
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k_] for k_ in sorted_keys[:top_n]]


# ═══════════════════════════════════════════════════════════════
# HYBRID RAG ENGINE
# ═══════════════════════════════════════════════════════════════

class HybridRAGEngine:
    """
    Hybrid retriever: BM25 (keyword) + Chroma (semantic) with RRF fusion.

    Lifecycle:
      - initialize()  → load + chunk + index (lifespan e ekbar)
      - retrieve()    → per-request, top-k chunks
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._bm25: Any = None
        self._chroma: Any = None
        self._documents: list[Document] = []
        self._ready = False

    # ═══════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════

    def initialize(self) -> None:
        """
        Ki: Policy load + chunk + index (BM25 + Chroma).
        Kno: Slow operation — lifespan e ekbar.
        """
        policy_path = Path("data/betsson_policy.txt")
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Policy file not found: {policy_path.resolve()}"
            )

        raw_text = policy_path.read_text(encoding="utf-8")
        logger.info("Policy loaded: %d chars", len(raw_text))

        # ── Step 1: Chunk ──
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(raw_text)
        logger.info("Split into %d chunks", len(chunks))

        self._documents = [
            Document(
                page_content=chunk,
                metadata={"source": "betsson_policy", "chunk_id": i},
            )
            for i, chunk in enumerate(chunks)
        ]

        # ── Step 2: Chroma vector store ──
        try:
            vectorstore = Chroma.from_documents(
                documents=self._documents,
                collection_name=self.settings.chroma_collection,
                persist_directory=self.settings.chroma_persist_dir,
            )
            self._chroma = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            logger.info(
                "Chroma index ready: collection=%s dir=%s",
                self.settings.chroma_collection,
                self.settings.chroma_persist_dir,
            )
        except Exception as exc:
            logger.exception("Chroma init failed: %s", exc)
            raise

        # ── Step 3: BM25 ──
        try:
            self._bm25 = BM25Retriever.from_documents(self._documents)
            self._bm25.k = 5
            logger.info("BM25 index ready: %d docs", len(self._documents))
        except Exception as exc:
            logger.exception("BM25 init failed: %s", exc)
            raise

        self._ready = True
        logger.info("Hybrid RAG engine ready")

    # ═══════════════════════════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """
        Ki: Query er sathe relevant chunks return kore.
        How:
          1. BM25 e top-5
          2. Chroma e top-5
          3. RRF diye fuse
          4. Top-k return
        """
        if not self._ready:
            logger.warning("RAG engine not ready — returning empty")
            return []

        try:
            bm25_results = self._bm25.invoke(query) if self._bm25 else []
        except Exception as exc:
            logger.warning("BM25 retrieve failed: %s", exc)
            bm25_results = []

        try:
            chroma_results = self._chroma.invoke(query) if self._chroma else []
        except Exception as exc:
            logger.warning("Chroma retrieve failed: %s", exc)
            chroma_results = []

        if not bm25_results and not chroma_results:
            return []

        # RRF fusion
        fused = _rrf_fuse(
            ranked_lists=[bm25_results, chroma_results],
            k=60,
            top_n=top_k,
        )
        logger.debug(
            "Retrieved bm25=%d chroma=%d fused=%d for query=%r",
            len(bm25_results), len(chroma_results), len(fused), query[:60],
        )
        return fused

    def format_context(self, docs: list[Document]) -> str:
        """
        Ki: Documents → numbered context string for LLM.
        How: Prottek chunk er age '[Source N]' label.
        """
        if not docs:
            return "(no relevant context found)"
        parts = []
        for i, doc in enumerate(docs, start=1):
            parts.append(f"[Source {i}]\n{doc.page_content.strip()}\n")
        return "\n".join(parts)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def document_count(self) -> int:
        return len(self._documents)


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_rag_engine() -> HybridRAGEngine:
    """Singleton RAG engine."""
    settings = get_settings()
    engine = HybridRAGEngine(settings)
    engine.initialize()
    return engine
