from sentence_transformers import SentenceTransformer
import faiss
import json
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings
from app.retrievers._utils import dedupe_top_k

_EMB_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)

# In-memory FAISS cache — keyed by (index_path, docs_path) tuple.
# Invalidated explicitly when the active dataset changes.
_FAISS_CACHE: Optional[tuple] = None  # (index_path, docs_path, index, docs)


def invalidate_cache() -> None:
    """Reset the in-memory FAISS cache so the next query reloads from disk."""
    global _FAISS_CACHE
    _FAISS_CACHE = None


def build_index(documents: List[dict], index_path: str = settings.FAISS_INDEX_PATH):
    texts = [d["text"] for d in documents]
    embeddings = _EMB_MODEL.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # normalize -> cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    # NOTE: docs are NOT re-written here — they already live in their own
    # *_docs.jsonl file and are read from settings.DOCUMENTS_PATH at query time.
    invalidate_cache()  # clear stale cache after rebuilding
    return index


def load_index(index_path: Optional[str] = None):
    global _FAISS_CACHE
    # Always read settings at call time so dataset switches take effect immediately
    if index_path is None:
        index_path = settings.FAISS_INDEX_PATH
    docs_path = settings.DOCUMENTS_PATH

    # Return cached version if both paths are unchanged
    if (
        _FAISS_CACHE is not None
        and _FAISS_CACHE[0] == index_path
        and _FAISS_CACHE[1] == docs_path
    ):
        return _FAISS_CACHE[2], _FAISS_CACHE[3]

    index = faiss.read_index(index_path)
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    _FAISS_CACHE = (index_path, docs_path, index, docs)
    return index, docs


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    idx, docs = load_index()
    q_emb = _EMB_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    # Fetch extra candidates so we can deduplicate and still return k results
    D, I = idx.search(q_emb, min(k * 4, len(docs)))

    candidates = [
        (docs[i], float(score)) for i, score in zip(I[0], D[0]) if 0 <= i < len(docs)
    ]
    return dedupe_top_k(candidates, k)
