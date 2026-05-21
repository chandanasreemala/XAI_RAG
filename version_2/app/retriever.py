"""Facade for retrieval — delegates to app.retrievers.router."""

from typing import Any, Dict, List, Optional

from app.retrievers.router import retrieve as _retrieve


def retrieve(
    query: str,
    k: int = 5,
    retriever: Optional[str] = None,
    generator=None,
) -> List[Dict[str, Any]]:
    return _retrieve(query=query, k=k, retriever=retriever, generator=generator)
