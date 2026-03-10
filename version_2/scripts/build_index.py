# usage: python -m scripts.build_index <docs.jsonl> [output_dir]
#   output_dir defaults to the same directory as docs.jsonl
# Examples:
#   python -m scripts.build_index data/trivia/trivia_docs.jsonl
#   python -m scripts.build_index data/hotpot/hotpot_docs.jsonl data/hotpot
# scripts/build_index.py

import json, os, sys
from typing import Optional
from app.retrievers.dense_faiss import build_index as build_dense
from app.retrievers.bm25 import build_index as build_bm25

def main(path: str, out_dir: Optional[str] = None):
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Docs file not found: '{path}'\n"
                                f"  Hint: did you mean one of these?\n" +
                                "\n".join(f"    {p}" for p in _suggest(path)))

    if out_dir is None:
        out_dir = os.path.dirname(path)

    os.makedirs(out_dir, exist_ok=True)

    # Derive a dataset prefix from the docs filename:
    #   hotpot_docs.jsonl  ->  hotpot
    #   trivia_docs.jsonl  ->  trivia
    #   anything_docs.jsonl -> anything
    #   other.jsonl        -> other  (fallback: full stem)
    basename = os.path.basename(path)                          # e.g. hotpot_docs.jsonl
    stem     = os.path.splitext(basename)[0]                   # e.g. hotpot_docs
    prefix   = stem[:-5] if stem.endswith("_docs") else stem   # e.g. hotpot

    faiss_path = os.path.join(out_dir, f"{prefix}_faiss.index")
    bm25_path  = os.path.join(out_dir, f"{prefix}_bm25.pkl")

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents from '{path}'")

    # Dense FAISS index
    build_dense(docs, index_path=faiss_path)
    print(f"Dense FAISS index built at '{faiss_path}'")

    # BM25 index
    build_bm25(docs_path=path, bm25_path=bm25_path)
    print(f"BM25 index built at '{bm25_path}'")

def _suggest(path: str):
    """Walk nearby directories and suggest similarly-named files."""
    import glob
    base = os.path.splitext(os.path.basename(path))[0]
    root = "data" if os.path.isdir("data") else "."
    return glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)[:8]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.build_index <docs.jsonl> [output_dir]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
