## RAG-Ex v2 — Quick start


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
Cuda example (direct):
CUDA_VISIBLE_DEVICES=1 uvicorn app.api:app --reload --port 8000 --host 0.0.0.0

# Or, use the helper script (recommended):
./run_server_v2.sh 8000
```



> **Note:** `--host 0.0.0.0` binds to all network interfaces so colleagues on the same network can access the API at `http://<your-machine-IP>:8000`. Without it, the server is only reachable from localhost. Find your IP with `hostname -I`.

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
- "importance_mode": "modified_ragex", "ragex_core" aplha value: 0-1
- "k-values": "top-1", "top-3", "top-20%"


## Running v2 and v3 side-by-side (ports & shareable URLs)

Start `version_2` on port 8000 and `version_3` on port 8001 so they don't conflict.

```bash
# version_2
cd /home/csmala/XAI_RT_RAG/ragex_core_v2_main/version_2
./run_server_v2.sh 8000

OR 

CUDA_VISIBLE_DEVICES=1 uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
CUDA_VISIBLE_DEVICES=1 WATCHFILES_FORCE_POLLING=true uvicorn app.api:app --reload --reload-dir app --port 8000 --host 0.0.0.0


Shareable URLs for colleagues on the same LAN:

Version 2:
http://131.114.2.129:8000/static/index.html


Find your local IP with:

```bash
hostname -I | awk '{print $1}'
```

To expose either server publicly, use a tunnel like `ngrok`:

```bash
ngrok http 8000   # share version_2
```

The `ngrok` command prints a public https URL you can give to remote colleagues.



Demo: Gary Harrison, began his career in the 1970s and has written over how many major-label recorded songs including several number-one hits, another artist who have recorded his work include Bryan White, an American country music artist?

Who is the American internet entrepreneur who founded the company featured on 24 Hours on Craigslist?


Which American college that has sent students to Centre for Medieval and Renaissance Studies was founded in 1874?
compare dense and hybrid - 5 docs, flant5, cop fusion

Which airport is closer to Cleaveland, Ohio, Luis Muñoz Marín International Airport or Rickenbacker International Airport?