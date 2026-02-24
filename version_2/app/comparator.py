from rapidfuzz.distance import Levenshtein, JaroWinkler
from collections import Counter
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

sbert = SentenceTransformer(settings.SBERT_MODEL)

def levenshtein_score(a: str, b: str) -> float:
    dist = Levenshtein.normalized_similarity(a, b)
    return dist

def jaro_winkler_score(a: str, b: str) -> float:
    return JaroWinkler.similarity(a, b)

def ngram_overlap(a: str, b: str, n=3) -> float:
    def ngrams(s: str):
        toks = s.split()
        return [" ".join(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))]
    na = ngrams(a)
    nb = ngrams(b)
    if not na or not nb:
        return 0.0
    ca = Counter(na)
    cb = Counter(nb)
    common = sum((ca & cb).values())
    total = max(len(na), len(nb))
    return common / total

def semantic_cosine(a: str, b: str) -> float:
    emb = sbert.encode([a, b])
    a_v, b_v = emb[0], emb[1]
    cos = (a_v @ b_v) / (np.linalg.norm(a_v) * np.linalg.norm(b_v) + 1e-10)
    return float(cos)

COMPARATORS = {
    "levenshtein": levenshtein_score,
    "jaro_winkler": jaro_winkler_score,
    "n_gram": ngram_overlap,
    "semantic": semantic_cosine
}
