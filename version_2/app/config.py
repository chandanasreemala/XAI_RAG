import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file's location (version_2/app/ -> version_2/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)  # version_2/
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")

class Settings(BaseSettings):
    HF_TOKEN: str
    HF_MODEL: str = "google/flan-t5-small"
    SBERT_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RETRIEVER_DEFAULT: str = "dense"  # "dense" or "bm25"
    BM25_INDEX_PATH: str = "data/hotpot/bm25.pkl"
    FAISS_INDEX_PATH: str = "data/hotpot/faiss.index"
    DOCUMENTS_PATH: str = "data/hotpot/hotpot_docs.jsonl"

    model_config = SettingsConfigDict(env_file=_ENV_FILE)

settings = Settings()
