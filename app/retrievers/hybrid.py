# app/retrievers/hybrid.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np

from sentence_transformers import CrossEncoder
import torch
from app.config import settings
from app.retrievers import dense_faiss, bm25


# ----------------------------
# Lazy-loaded reranker
# ----------------------------
_RERANKER: Optional[CrossEncoder] = None


def _get_reranker(model_name: Optional[str] = None) -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        name = model_name or getattr(settings, "BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
        # Prefer GPU if available; fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # CrossEncoder supports device=... in sentence-transformers
        _RERANKER = CrossEncoder(name, device=device)
    return _RERANKER

def _minmax_scale(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn = float(min(scores))
    mx = float(max(scores))
    if mx - mn < 1e-12:
        return [1.0 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]


def _doc_id(item: Dict[str, Any]) -> str:
    # repo schema: {"doc": {"id": ... , "text": ..., "meta": ...}, "score": ...}
    d = item.get("doc", {}) if isinstance(item.get("doc"), dict) else {}
    return str(d.get("id", ""))


def _doc_text(item: Dict[str, Any]) -> str:
    d = item.get("doc", {}) if isinstance(item.get("doc"), dict) else {}
    return str(d.get("text", ""))


def retrieve(
    query: str,
    k: int = 5,
    *,
    dense_k: int = 50,
    sparse_k: int = 50,
    rerank_k: int = 100,
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
    reranker_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
      1) Get dense candidates (FAISS) and sparse candidates (BM25)
      2) Merge & dedupe by doc.id
      3) Rerank candidates with a BGE CrossEncoder reranker
      4) Return top-k docs

    Returns list[{"doc": {...}, "score": <reranker_score>}]
    and preserves original scores inside doc["meta"]["hybrid_scores"] for debugging.
    """
    dense_res = dense_faiss.retrieve(query, dense_k)
    sparse_res = bm25.retrieve(query, sparse_k)

    dense_scores = [float(x.get("score", 0.0)) for x in dense_res]
    sparse_scores = [float(x.get("score", 0.0)) for x in sparse_res]
    dense_scaled = _minmax_scale(dense_scores)
    sparse_scaled = _minmax_scale(sparse_scores)

    # Merge by doc.id
    merged: Dict[str, Dict[str, Any]] = {}

    for item, s_scaled in zip(dense_res, dense_scaled):
        did = _doc_id(item)
        if not did:
            continue
        if did not in merged:
            merged[did] = {
                "doc": copy.deepcopy(item["doc"]),
                "dense_score": float(item.get("score", 0.0)),
                "bm25_score": 0.0,
                "hybrid_score": w_dense * float(s_scaled),
            }
        else:
            merged[did]["dense_score"] = max(merged[did]["dense_score"], float(item.get("score", 0.0)))
            merged[did]["hybrid_score"] = max(merged[did]["hybrid_score"], w_dense * float(s_scaled))

    for item, s_scaled in zip(sparse_res, sparse_scaled):
        did = _doc_id(item)
        if not did:
            continue

        raw = float(item.get("score", 0.0))
        norm = float(s_scaled)

        if did not in merged:
            merged[did] = {
                "doc": copy.deepcopy(item["doc"]),
                "dense_score": 0.0,
                "bm25_raw": raw,
                "bm25_norm": norm,
                "hybrid_score": w_sparse * norm,
            }
        else:
            merged[did]["bm25_raw"] = max(merged[did].get("bm25_raw", 0.0), raw)
            merged[did]["bm25_norm"] = max(merged[did].get("bm25_norm", 0.0), norm)
            merged[did]["hybrid_score"] += w_sparse * norm

    candidates = list(merged.values())
    if not candidates:
        return []

    # Preselect top rerank_k by hybrid_score (cheap)
    candidates.sort(key=lambda x: float(x.get("hybrid_score", 0.0)), reverse=True)
    candidates = candidates[: max(1, int(rerank_k))]

    reranker = _get_reranker(reranker_model)
    pairs = [(query, str(c["doc"].get("text", ""))) for c in candidates]
    rr_scores = reranker.predict(pairs)

    # Attach reranker score
    out: List[Dict[str, Any]] = []
    for c, rr in zip(candidates, rr_scores):
        doc = c["doc"]
        meta = doc.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        meta["hybrid_scores"] = {
            "dense_norm": float(c.get("dense_score", 0.0)),     # already normalized
            "bm25_norm": float(c.get("bm25_norm", 0.0)),        # 0–1 ✅
            "hybrid_pre": float(c.get("hybrid_score", 0.0)),    # fused score
            "reranker": float(rr),
            # optional: keep raw for debugging
            "bm25_raw": float(c.get("bm25_raw", 0.0)),
        }
        doc["meta"] = meta
        out.append({"doc": doc, "score": float(rr)})

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:k]
