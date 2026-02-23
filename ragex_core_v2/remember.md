# RAG-Ex Core — Minimal, production-ready extraction

This repository contains a modular reimplementation of the core experiment code from
the explainable-lms RAG-Ex project. It extracts the experiment logic (perturbations,
RAG retriever, generation, comparators, metrics) and removes the LMS/service/UI dependency.

## Structure
- `app/` - main python package with modules:
  - `retrievers/`
     - `bm25.py`
     - `dense_faiss.py`
     - `router.py`
  - `perturb.py` - perturbation strategies, what units are perturbed and how
  - `generator.py` - Hugging Face generation wrapper
  - `comparator.py` - comparison functions (syntactic & semantic)
  - `retriever.py` - FAISS + SBERT retrieval
  - `evaluator.py` - metrics (F1, MRR, Response Match)
  - `api.py` - FastAPI endpoints to run experiments
- `data/`
   - `bm25.pkl`
   - `faiss.index`
   - `docs.jsonl`
- `scripts/build_index.py` - build FAISS index from `data/docs.jsonl`
- `requirements.txt` - pinned dependencies

## Quick start
1. Create `.env` with your HF token:
   ```
   HF_TOKEN=hf_xxx
   HF_MODEL=google/flan-t5-large
   SBERT_MODEL=sentence-transformers/all-mpnet-base-v2
   ```
2. Install:
   ```
   pip install -r requirements.txt
   conda activate ragex
   cd rag_ex_core/
   
   ```
3. Prepare `data/docs.jsonl` (one JSON per line, e.g. {"id":"doc1","text":"...","meta":{}})
4. Build index(dense and bm25):
   ```
   python -m scripts.build_index data/docs.jsonl
   ```
5. Start API:
   ```
   uvicorn app.api:app --reload --port 8000
   
   ```
6. Test the API:

- `With Context passed in the input` at POST/explain
{
  "question": "What river runs through Paris?",
  "context": "Paris is the capital and most populous city of France. The city is located along the Seine River in northern France.",
  "perturber": "leave_one_out",
  "comparator": "semantic",
  "top_k_docs": 3,
  "max_length": 256,
  "temperature": 0.0
}

- `Without Context passed, and switched to bm25 retriever in the input` at POST/explain

{
  "question": "What river runs through Paris?",
  "context": "",
  "retriever": "bm25",
  "top_k_docs": 3,
  "perturber": "leave_one_out",
  "comparator": "semantic"
}

In the explanation we can choose between : word | phrase | sentence | paragraph 
In the perturbation we can choose between : "leave_one_out" , "random_noise", "entity_perturber", "antonym_perturber", "synonym_perturber", "reorder_perturber"
In the comparator we can choose between : "levenshtein" ,"jaro_winkler", "n_gram", "semantic"

"importance_mode": "modified_ragex", "ragex_core" aplha value: 0-1
"k-values": "top-1", "top-3", "top-20%"

## Notes
- This implementation uses Hugging Face models directly 
- For evaluation at scale, adapt `app/api.py` or write a runner to iterate datasets and collect metrics.
- Existing retriever is standard dense retriever (SentenceTransformer embeddings + FAISS cosine)
- Embedding model: sentence-transformers/all-mpnet-base-v2, 
Index: FAISS IndexFlatIP (inner product), 
Similarity: cosine similarity (we normalize vectors to unit length before indexing/searching)

The new constructed (perturbed) prompts are shown in
details[<unit>]["samples"][i]["prompt"],
while the baseline prompt is in
details["_baseline"]["prompt"].


Conclusion: If changing a unit(based on explanation_level) breaks the answer, the unit matters.
If changing it doesn’t breaks the answer, it doesn’t matter and not of an interest.

# Demo:
Example 1: See how debug is showing the pertubations when entities are changing --- masks seine river with bar, bazz then the scores changes.  

 {
  "question": "What river runs through Paris?",
  "context": "",
  "retriever": "bm25",
  "top_k_docs": 3,
  "explanation_level": "sentence",
  "perturber": "entity_perturber",
  "comparator": "semantic",
  "max_length": 256,
  "temperature": 0,
  "debug": true,
  "debug_max_perturbations": 2
}


Example 2: Compare dense and hybrid retrievers effectiveness, where using dense, the answer is not correct but using the hybrid the answer is correct 

{
  "question": "What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?",
  "context": "",
  "retriever": "hybrid",
  "top_k_docs": 3,
  "explanation_level": "sentence",
  "perturber": "leave_one_out",
  "comparator": "semantic",
  "max_length": 256,
  "temperature": 0,
  "debug": false,
  "debug_max_perturbations": 5
}

Example 3: For understanding the token importances scores, 

{
  "question": "A Head Full of Dreams Tour is the seventh tour by Coldplay, and which had it's first show at a stadium that is known as Estadio Unico and is owned by who?",
  "context": "",
  "retriever": "dense",
  "top_k_docs": 3,
  "explanation_level": "sentence",
  "perturber": "leave_one_out",
  "comparator": "semantic",
  "max_length": 256,
  "temperature": 0,
  "debug": true,
  "debug_max_perturbations": 3
}