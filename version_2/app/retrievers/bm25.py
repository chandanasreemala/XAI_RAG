from __future__ import annotations
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi
from app.config import settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]

@dataclass
class BM25Index:
    bm25: BM25Okapi
    docs: List[Dict[str, Any]]

_BM25_CACHE: BM25Index | None = None


def invalidate_cache() -> None:
    """Reset the in-memory BM25 index so the next retrieve() call reloads from disk.
    Call this whenever the active dataset changes."""
    global _BM25_CACHE
    _BM25_CACHE = None

def _load_docs_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"docs.jsonl line {line_no} missing 'text'")
            docs.append(obj)
    if not docs:
        raise ValueError(f"No docs in {path}")
    return docs

def build_index(
    docs_path: Optional[str] = None,
    bm25_path: Optional[str] = None,
) -> BM25Index:
    if docs_path is None:
        docs_path = settings.DOCUMENTS_PATH
    if bm25_path is None:
        bm25_path = settings.BM25_INDEX_PATH
    docs = _load_docs_jsonl(docs_path)
    corpus = [_tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus)
    idx = BM25Index(bm25=bm25, docs=docs)
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(idx, f)
    return idx

def load_index(bm25_path: Optional[str] = None) -> BM25Index:
    if bm25_path is None:
        bm25_path = settings.BM25_INDEX_PATH
    if not os.path.exists(bm25_path):
        # try building from docs.jsonl automatically
        return build_index()
    with open(bm25_path, "rb") as f:
        return pickle.load(f)

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        _BM25_CACHE = load_index()

    q = _tokenize(query)
    scores = _BM25_CACHE.bm25.get_scores(q)
    # Fetch extra candidates so we can deduplicate and still return k results
    top = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k * 4]

    results: List[Dict[str, Any]] = []
    seen_texts: set = set()
    for i in top:
        text = _BM25_CACHE.docs[i].get("text", "")
        norm = text.lower().strip()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        results.append({"doc": _BM25_CACHE.docs[i], "score": float(scores[i])})
        if len(results) == k:
            break
    return results
