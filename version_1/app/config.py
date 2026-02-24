from pydantic import BaseSettings

class Settings(BaseSettings):
    HF_TOKEN: str
    HF_MODEL: str = "google/flan-t5-large"
    SBERT_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RETRIEVER_DEFAULT: str = "dense"  # "dense" or "bm25"
    BM25_INDEX_PATH: str = "data/hotpot/bm25.pkl"
    FAISS_INDEX_PATH: str = "data/hotpot/faiss.index"
    DOCUMENTS_PATH: str = "data/hotpot/hotpot_docs.jsonl"

    class Config:
        env_file = ".env"

settings = Settings()
