"""Central registry of indexed datasets (paths + display labels)."""

from typing import Dict

DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "hotpot": {
        "DOCUMENTS_PATH": "data/hotpot/hotpot_docs.jsonl",
        "FAISS_INDEX_PATH": "data/hotpot/hotpot_faiss.index",
        "BM25_INDEX_PATH": "data/hotpot/hotpot_bm25.pkl",
        "label": "HotpotQA",
    },
    "trivia": {
        "DOCUMENTS_PATH": "data/trivia/trivia_docs.jsonl",
        "FAISS_INDEX_PATH": "data/trivia/trivia_faiss.index",
        "BM25_INDEX_PATH": "data/trivia/trivia_bm25.pkl",
        "label": "TriviaQA (Wikipedia)",
    },
}
