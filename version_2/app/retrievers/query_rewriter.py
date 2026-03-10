"""
app/retrievers/query_rewriter.py
Multi-query retrieval: rewrite the user query N times, retrieve for each,
pool by Reciprocal Rank Fusion (RRF), then cross-encoder rerank and return top-k.

Pipeline:
  user query
      → rewrite to Q1…QN           (LLM or heuristic fallback)
      → retrieve top-M for each    (hybrid: dense + BM25 + reranker)
      → RRF merge + deduplicate
      → cross-encoder rerank
      → top-k results

Ported from version_3 to version_2. No extra config keys required — all
tunable parameters have sensible defaults built-in.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Defaults (version_2 config doesn't expose these keys)
# ---------------------------------------------------------------------------
_DEFAULT_N_REWRITES: int = 4
_DEFAULT_USE_LLM: bool = True


# ---------------------------------------------------------------------------
# Heuristic rewriters (zero-cost; always available as fallback)
# ---------------------------------------------------------------------------

def _heuristic_rewrites(query: str, n: int) -> List[str]:
    """Generate n simple heuristic rewrites without using an LLM."""
    q = query.strip().rstrip("?")
    candidates: List[str] = [query]  # original always included

    # 1. Question → imperative / declarative
    for prefix in ("What is", "What are", "Who is", "Where is", "When did", "How does"):
        if query.lower().startswith(prefix.lower()):
            rest = query[len(prefix):].strip().rstrip("?")
            candidates.append(f"Tell me about{rest}")
            break

    # 2. Add qualifiers
    candidates.append(f"{q} in detail?")
    candidates.append(f"briefly explain: {q}")

    # 3. Entity-focused: keep NOUN-like tokens (≥4 chars, titlecase or all-caps)
    nouns = [t for t in query.split() if (len(t) >= 4 and (t[0].isupper() or t.isupper()))]
    if nouns:
        candidates.append(" ".join(nouns))

    # 4. Keywords only (drop common stopwords heuristically)
    _STOP = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "at", "for",
        "to", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "and", "or", "but", "not", "with", "by", "from",
    }
    kw = [t.lower() for t in re.split(r"\W+", query) if t.lower() not in _STOP and len(t) > 2]
    if kw:
        candidates.append(" ".join(kw))

    # 5. "Explain X" paraphrase
    candidates.append(f"Explain {q.lower()}")

    # Deduplicate while preserving order, normalise whitespace
    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        nc = " ".join(c.split())
        if nc.lower() not in seen:
            seen.add(nc.lower())
            unique.append(nc)

    return unique[:n]


# ---------------------------------------------------------------------------
# LLM-based rewriter
# ---------------------------------------------------------------------------

def _llm_rewrites(query: str, n: int, generator) -> List[str]:
    """Use the loaded GeneratorWrapper to produce n query paraphrases."""
    prompt = (
        f"Generate {n} different ways to ask the following question. "
        f"Each rewrite should be on its own line and cover a different angle "
        f"(paraphrase, entity-focused, keyword-only, etc.).\n\n"
        f"Original question: {query}\n\nRewrites:"
    )
    try:
        result = generator(
            prompt,
            max_length=256,
            do_sample=True,
            temperature=0.7,
        )
        raw = result[0].get("generated_text", "") if isinstance(result, list) else str(result)
        lines = [l.strip().lstrip("-•*1234567890. ") for l in raw.split("\n")]
        rewrites = [l for l in lines if len(l) > 5 and l.lower() != query.lower()]
        all_queries = [query] + rewrites[: n - 1]
        return all_queries[:n]
    except Exception:
        return _heuristic_rewrites(query, n)


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    results_per_query: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    import copy

    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}

    for ranked_list in results_per_query:
        for rank, item in enumerate(ranked_list, start=1):
            doc = item.get("doc", {})
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                doc_id = str(hash(doc.get("text", "")[:120]))

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = copy.deepcopy(doc)

    merged = [
        {"doc": doc_store[did], "score": float(rrf_scores[did])}
        for did in sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    ]
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
    *,
    n_rewrites: Optional[int] = None,
    use_llm: Optional[bool] = None,
    generator=None,
    per_query_k: int = 30,
    reranker_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-query retrieval pipeline.

    1. Rewrite `query` into n_rewrites distinct variants.
    2. Run hybrid retrieval for each variant.
    3. Pool via RRF.
    4. Cross-encoder rerank → top-k.
    """
    from app.retrievers import hybrid  # lazy import to avoid circular deps

    n = n_rewrites if n_rewrites is not None else _DEFAULT_N_REWRITES
    _use_llm = use_llm if use_llm is not None else _DEFAULT_USE_LLM

    # ── 1. Rewrite ──────────────────────────────────────────────────────────
    if _use_llm and generator is not None:
        queries = _llm_rewrites(query, n, generator)
    else:
        queries = _heuristic_rewrites(query, n)

    # Ensure original is always present
    if query not in queries:
        queries.insert(0, query)

    # ── 2. Retrieve for each rewritten query ────────────────────────────────
    results_per_query: List[List[Dict[str, Any]]] = []
    for q in queries:
        try:
            try:
                res = hybrid.retrieve(q, per_query_k, dense_k=per_query_k, sparse_k=per_query_k)
            except TypeError:
                res = hybrid.retrieve(q, per_query_k)
            results_per_query.append(res)
        except Exception:
            pass

    if not results_per_query:
        return hybrid.retrieve(query, k)

    # ── 3. RRF merge ────────────────────────────────────────────────────────
    merged = _rrf_merge(results_per_query)

    # ── 4. Cross-encoder rerank → top-k ─────────────────────────────────────
    try:
        reranker = hybrid._get_reranker(reranker_model)
        candidates = merged[: max(k * 5, 50)]
        pairs = [(query, str(c["doc"].get("text", ""))) for c in candidates]
        rr_scores = reranker.predict(pairs)

        import copy
        out: List[Dict[str, Any]] = []
        for c, rr in zip(candidates, rr_scores):
            doc = copy.deepcopy(c["doc"])
            meta = doc.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta = dict(meta)
            meta["multi_query_rrf_score"] = float(c["score"])
            meta["reranker_score"] = float(rr)
            meta["queries_used"] = queries
            doc["meta"] = meta
            out.append({"doc": doc, "score": float(rr)})

        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out[:k]
    except Exception:
        # Reranker unavailable — annotate with queries_used and return RRF order
        import copy
        out = []
        for c in merged[:k]:
            doc = copy.deepcopy(c["doc"])
            meta = doc.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            meta = dict(meta)
            meta["queries_used"] = queries
            doc["meta"] = meta
            out.append({"doc": doc, "score": c["score"]})
        return out
