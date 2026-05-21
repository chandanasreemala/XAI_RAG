"""Shared helpers for retriever implementations."""

from __future__ import annotations

from typing import Any, Dict, List


def dedupe_top_k(
    candidates: List[tuple],
    k: int,
    *,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """
    Deduplicate retrieval candidates by normalised document text and return top-k.

    Each candidate is (doc_dict, score).
    """
    results: List[Dict[str, Any]] = []
    seen_texts: set = set()
    for doc, score in candidates:
        text = doc.get(text_key, "") if isinstance(doc, dict) else ""
        norm = str(text).lower().strip()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        results.append({"doc": doc, "score": float(score)})
        if len(results) >= k:
            break
    return results
