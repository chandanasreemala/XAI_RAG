Absolutely — here’s a **clean, handy, “future-you friendly” `README.md`** that explains **what this repo does, why it exists, and how to use each major capability**, including:

* Hybrid retrieval (dense + BM25 + BGE reranker)
* Token-importance explanations
* Case A / B / C faithfulness experiments
* The new `/interpretation` API
* How to run things correctly and what to look for

You can drop this directly as `README.md` at the repo root.

---

# XAI-RAG: Interpretable Retrieval-Augmented Generation

This repository implements an **explainable RAG (Retrieval-Augmented Generation)** system with:

* Dense + sparse + reranked retrieval
* Token / sentence importance explanations
* Faithfulness testing via **controlled deletion experiments**
* A clean API to test **whether explanations actually matter**

The goal is **not just to answer questions**, but to **understand *why* the model answered the way it did**, and to **verify that explanations are faithful**.

---

## High-Level Architecture

```
Question
   ↓
Retriever (dense / bm25 / hybrid)
   ↓
Context (retrieved docs)
   ↓
LLM Generator
   ↓
Answer
   ↓
Explainer (perturbation-based)
   ↓
Importance Ranking (most / important / least)
   ↓
Deletion Experiments (Case A/B/C)
   ↓
Faithfulness Diagnosis
```

---

## Retrieval Modes

### 1. Dense Retrieval

* Uses sentence-transformer embeddings + FAISS
* Good semantic recall
* Can miss keyword-exact matches

### 2. Sparse Retrieval (BM25)

* Classic lexical retrieval
* Strong keyword matching
* Weak semantic generalization

### 3. **Hybrid Retrieval (NEW)**

**Location:** `app/retrievers/hybrid.py`

Hybrid retrieval combines:

1. Dense retrieval (FAISS)
2. Sparse retrieval (BM25)
3. **BGE CrossEncoder reranking**

**Pipeline:**

```
query
 → dense_k docs (FAISS)
 → sparse_k docs (BM25)
 → merge + dedupe
 → BGE reranker (GPU if available)
 → top-k final docs
```

**Why this matters**

* Dense gives recall
* BM25 gives precision
* BGE reranker gives *ordering correctness*

This is the **recommended default retriever** for serious experiments.

---

## Explanation: Token / Sentence Importance

We explain model behavior using **perturbation-based importance**.

### How importance is computed

1. Generate baseline answer `y`
2. Split context into units (sentences or tokens)
3. Perturb one unit at a time
4. Re-generate answer `y'`
5. Measure similarity `sim(y, y')`
6. Importance = `1 − sim(y, y')`
7. Normalize importances to `[0,1]`

### Buckets

Units are grouped into:

* **most** (top ~20%)
* **important** (middle)
* **least** (bottom)

These buckets are used for deletion experiments.

---

## Faithfulness Testing: Case A / B / C

**Key idea:**

> If an explanation is correct, removing “important” units should hurt the answer more than removing unimportant ones.

We test this explicitly.

---

### Deletion Arms

For each example we generate multiple answers:

| Arm          | Description                                          |
| ------------ | ---------------------------------------------------- |
| `DEL_NONE`   | Original context                                     |
| `DEL_MOST`   | Remove top-K important units                         |
| `DEL_LEAST`  | Remove bottom-K units (control)                      |
| `DEL_RANDOM` | Remove K random units (control, repeated many times) |
| `DEL_Q`      | Remove question (sanity check)                       |

We test **multiple K values**:

* top-1
* top-3
* top-20%
* cumulative top-K

---

### Comparison Metrics

Answers are compared using:

* **Semantic similarity** (Sentence-BERT cosine)
* Exact match
* Token overlap (Jaccard)
* Optional edit distance / BLEU / ROUGE

Primary signal:

```
semantic_similarity(y, y')
```

A response is considered “unchanged” if:

```
similarity ≥ T   (default T = 0.85)
```

---

## Case Definitions (IMPORTANT)

Let:

* `sim_most`   = similarity after `DEL_MOST`
* `sim_least` = similarity after `DEL_LEAST`
* `sim_random`= mean similarity over random deletions

### ✅ Case A — Faithful Explanation

```
sim_most   ≪ sim_random
sim_least ≈ sim_random
```

Interpretation:

* Removing important units hurts
* Removing unimportant ones does not
* ✅ explanation is trustworthy

---

### ❌ Case B — Explainer or Model Failure

```
sim_most   ≈ sim_random
sim_least ≈ sim_random
```

Interpretation:

* Importance ranking has no effect
* Either:

  * explainer failed, or
  * model isn’t using the context meaningfully

---

### ⚠️ Case C — Redundancy / Partial Dependence

```
sim_most ≪ sim_random
BUT sim_most is still high (e.g. ≥ 0.9)
```

Interpretation:

* Important units matter
* But evidence is redundant
* Inspect examples individually

---

## Random Deletion & Statistics

* `DEL_RANDOM` is repeated **50× (configurable)**
* We report:

  * mean similarity
  * 95% confidence interval
* Enables paired significance tests:

  * McNemar (changed / unchanged)
  * Wilcoxon / paired t-test on similarities
* Effect size = mean difference vs random

This avoids false conclusions from single random deletions.

---

## API Endpoints

### `/interpretation` (NEW)

**Purpose:**
Run the *entire pipeline*:

* retrieval
* explanation
* deletion experiments
* Case A/B/C diagnosis

**Method:** `POST`

**Minimal payload**

```json
{
  "question": "What is the capital of France?",
  "retriever": "hybrid",
  "top_k_docs": 3
}
```

**Extended payload**

```json
{
  "question": "What is the capital of France?",
  "retriever": "hybrid",
  "top_k_docs": 3,
  "explanation_level": "sentence",
  "k_values": ["top-1", "top-3", "top-20%"],
  "random_repeats": 50,
  "semantic_threshold_t": 0.85,
  "debug": true
}
```

**Returns**

* Importance scores & buckets
* Per-K deletion results
* Similarity scores
* Case (A/B/C) per K
* Optional debug info (answers, removed units)

---

## GPU Usage

* **Generator**: uses GPU if configured
* **BGE reranker**: automatically runs on GPU if available
* CPU fallback is automatic

No manual flags required.

---

## Recommended Workflow

1. **Use `retriever = hybrid`**
2. Run `/interpretation` on a validation set
3. Aggregate Case A/B/C statistics
4. Investigate:

   * Case B → explanation or retrieval issues
   * Case C → redundancy or dataset artifacts
5. Only trust explanations that consistently produce **Case A**

---

## Philosophy

> *An explanation that cannot fail is not an explanation.*

This repo treats explainability as a **hypothesis to be tested**, not a visualization to be trusted blindly.

---
