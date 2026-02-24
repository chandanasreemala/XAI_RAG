# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from typing import List
# from app.config import settings
# import json

# _EMB_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)

# def build_index(documents: List[dict], index_path: str = settings.FAISS_INDEX_PATH):
#     texts = [d["text"] for d in documents]
#     embeddings = _EMB_MODEL.encode(texts, convert_to_numpy=True)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     faiss.normalize_L2(embeddings)
#     index.add(embeddings)
#     faiss.write_index(index, index_path)
#     with open(settings.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
#         for d in documents:
#             f.write(json.dumps(d, ensure_ascii=False) + "\n")
#     return index

# def load_index(index_path: str = settings.FAISS_INDEX_PATH):
#     index = faiss.read_index(index_path)
#     docs = []
#     with open(settings.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             docs.append(json.loads(line))
#     return index, docs

# def retrieve(query: str, k: int = 5):
#     idx, docs = load_index()
#     q_emb = _EMB_MODEL.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     D, I = idx.search(q_emb, k)
#     results = []
#     for i, score in zip(I[0], D[0]):
#         results.append({"doc": docs[i], "score": float(score)})
#     return results

# app/retriever.py
from typing import Any, Dict, List, Optional
from app.retrievers.router import retrieve as _retrieve

def retrieve(query: str, k: int = 5, retriever: Optional[str] = None) -> List[Dict[str, Any]]:
    return _retrieve(query=query, k=k, retriever=retriever)
