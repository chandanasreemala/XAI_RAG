# usage: python -m scripts.build_index data/docs.jsonl
# scripts/build_index.py

import json, sys
from app.retrievers.dense_faiss import build_index as build_dense
from app.retrievers.bm25 import build_index as build_bm25
from app.config import settings

def main(path: str):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    # Dense FAISS index
    build_dense(docs, index_path=settings.FAISS_INDEX_PATH)
    print(f"Dense FAISS index built at {settings.FAISS_INDEX_PATH}")

    # BM25 index (uses docs.jsonl already written by dense build, but we can rebuild directly too)
    build_bm25(docs_path=settings.DOCUMENTS_PATH, bm25_path=settings.BM25_INDEX_PATH)
    print(f"BM25 index built at {settings.BM25_INDEX_PATH}")

if __name__ == "__main__":
    main(sys.argv[1])
