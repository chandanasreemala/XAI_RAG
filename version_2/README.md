## RAG-Ex v2 — Quick start

Minimal steps for colleagues to run and experiment.

1. Create `.env` with your Hugging Face token and models:

```
HF_TOKEN=hf_xxx
HF_MODEL=google/flan-t5-large
SBERT_MODEL=sentence-transformers/all-mpnet-base-v2
```

2. Install dependencies (from `version_2`):

```
pip install -r requirements.txt
```

3. Prepare data: create `data/docs.jsonl` (one JSON per line: {"id":"doc1","text":"...","meta":{}}).

4. Build indexes (dense + BM25):

```
python -m scripts.build_index data/docs.jsonl
```

5. Start API server:

```
CUDA_VISIBLE_DEVICES=0 uvicorn app.api:app --reload --port 8000
```

6. Example request (POST `/explain`):

```json
{
  "question": "What river runs through Paris?",
  "context": "",
  "retriever": "hybrid",
  "top_k_docs": 3,
  "perturber": "leave_one_out",
  "comparator": "semantic"
}
```

Notes:
- Use `retriever` = `dense`, `bm25`, or `hybrid`.
- `perturber` options: `leave_one_out`, `random_noise`, `entity_perturber`, `antonym_perturber`, `synonym_perturber`, `reorder_perturber`.
- `comparator` options: `levenshtein`, `jaro_winkler`, `n_gram`, `semantic`.


