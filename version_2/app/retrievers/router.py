from typing import Any, Dict, List, Optional
from app.config import settings
from app.retrievers import dense_faiss, bm25, hybrid

SUPPORTED = {"dense", "bm25", "hybrid"}

def retrieve(query: str, k: int = 5, retriever: Optional[str] = None) -> List[Dict[str, Any]]:
    r = (retriever or settings.RETRIEVER_DEFAULT).lower().strip()
    if r not in SUPPORTED:
        raise ValueError(f"Unknown retriever '{r}'. Supported: {sorted(SUPPORTED)}")

    if r == "dense":
        return dense_faiss.retrieve(query, k)
    if r == "bm25":
        return bm25.retrieve(query, k)
    if r == "hybrid":
        return hybrid.retrieve(query, k)

    # unreachable
    raise ValueError(f"Unhandled retriever '{r}'")
