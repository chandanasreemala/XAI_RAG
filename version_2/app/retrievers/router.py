from typing import Any, Dict, List, Optional
from app.config import settings
from app.retrievers import dense_faiss, bm25, hybrid

SUPPORTED = {"dense", "bm25", "hybrid", "multi_query"}


def retrieve(
    query: str,
    k: int = 5,
    retriever: Optional[str] = None,
    generator=None,          # GeneratorWrapper for LLM-based query rewriting
) -> List[Dict[str, Any]]:
    """
    Route retrieval to the appropriate backend.

    Retriever options:
      "dense"       – FAISS dense retrieval only
      "bm25"        – BM25 sparse retrieval only
      "hybrid"      – Dense + BM25 fused with RRF + cross-encoder rerank
      "multi_query" – Multi-query rewriting → hybrid per query → RRF → rerank
    """
    r = (retriever or settings.RETRIEVER_DEFAULT).lower().strip()
    if r not in SUPPORTED:
        raise ValueError(f"Unknown retriever '{r}'. Supported: {sorted(SUPPORTED)}")

    if r == "dense":
        return dense_faiss.retrieve(query, k)
    if r == "bm25":
        return bm25.retrieve(query, k)
    if r == "hybrid":
        return hybrid.retrieve(query, k)
    if r == "multi_query":
        from app.retrievers.query_rewriter import retrieve as mq_retrieve
        return mq_retrieve(query, k, generator=generator)

    raise ValueError(f"Unhandled retriever '{r}'")
