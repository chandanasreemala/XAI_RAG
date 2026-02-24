"""
RAG-Ex Evaluation Suite
=======================
Evaluates the full RAG pipeline across four axes:

  1. Retrieval Effectiveness  — Recall@K, MRR, MAP, NDCG@K
  2. Explanation Quality      — Explanation Token-F1, Explanation MRR
  3. Confidence Signal        — Spearman(Δc_i, w'_i) [confidence_retrieval_fusion only]
  4. Generation Quality       — Exact Match, Token-F1, BERTScore, RAGAS metrics
                                evaluated per model across 3-5 HuggingFace models

Dataset expected:
  data/hotpot/hotpot_answers.jsonl  — {id, question, answer}
  data/hotpot/hotpot_docs.jsonl     — {id, text, meta}

  Relevance is encoded in the document ID:
    Relevant    →  {qid}_{n}          (no _irr_)
    Distractor  →  {qid}_irr_{n}

Usage:
  ragex/bin/python scripts/evaluate.py \\
      --models google/flan-t5-large google/flan-t5-xl \\
               tiiuae/falcon-rw-1b \\
      --retrieval_results_csv results/retrieval/per_query_results.csv \\
      --n_samples 200 \\
      --api_url http://localhost:8000 \\
      --k_docs 3 \\
      --retriever dense \\
      --output_dir results/

  The script prints results to the console and saves two files:
    results/retrieval_explanation_metrics.csv
    results/generation_metrics.csv
"""

import argparse
import csv
import json
import math
import os
import re
import string
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CLI Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG-Ex evaluation — retrieval, explanation, confidence, generation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", required=True,
        metavar="HF_MODEL_ID",
        help="HuggingFace model IDs to evaluate (3-5 recommended).\n"
             "Example: google/flan-t5-large tiiuae/falcon-rw-1b",
    )
    p.add_argument(
        "--n_samples", type=int, default=200,
        help="Number of questions to evaluate (default: 200).",
    )
    p.add_argument(
        "--api_url", default="http://localhost:8000",
        help="Running RAG-Ex API base URL (default: http://localhost:8000).",
    )
    p.add_argument(
        "--k_docs", type=int, default=3,
        help="Number of documents to retrieve (default: 3).",
    )
    p.add_argument(
        "--retrievers", nargs="+", default=["dense", "bm25", "hybrid"],
        choices=["dense", "bm25", "hybrid"],
        help="Retrievers to evaluate (default: all three).",
    )
    p.add_argument(
        "--retrieval_results_csv", default=None,
        metavar="PATH",
        help="Path to an existing per_query_results.csv from evaluate_retrieval.py.\n"
             "If provided, retrieval metrics are loaded from this file instead of\n"
             "being recalculated via the /explain API.",
    )
    p.add_argument(
        "--explanation_level", default="sentence",
        choices=["word", "phrase", "sentence", "paragraph"],
        help="Granularity for explanation units (default: sentence).",
    )
    p.add_argument(
        "--importance_modes", nargs="+",
        default=["ragex_core", "retrieval_weighted"],
        choices=["ragex_core", "retrieval_weighted", "confidence_retrieval_fusion"],
        help="Importance scoring modes to evaluate for explanation metrics.\n"
             "Include 'confidence_retrieval_fusion' to also get Spearman correlation.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Fusion alpha for confidence_retrieval_fusion mode (default: 0.5).",
    )
    p.add_argument(
        "--answers_file", default="data/hotpot/hotpot_answers.jsonl",
        help="Path to hotpot_answers.jsonl.",
    )
    p.add_argument(
        "--docs_file", default="data/hotpot/hotpot_docs.jsonl",
        help="Path to hotpot_docs.jsonl.",
    )
    p.add_argument(
        "--hf_token", default=None,
        help="HuggingFace API token. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--output_dir", default="results",
        help="Directory to save CSV results (default: results/).",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=64,
        help="Max new tokens for HF Inference API calls (default: 64).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def _base_qid(doc_or_answer_id: str) -> str:
    """Strip trailing _{n} to get the base question ID."""
    return re.sub(r"_\d+$", "", doc_or_answer_id)


def _is_relevant(doc_id: str) -> bool:
    """Relevant docs have no '_irr_' in their ID."""
    return "_irr_" not in doc_id


def load_data(
    answers_file: str,
    docs_file: str,
    n_samples: int,
) -> Tuple[List[Dict], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
        samples        — list of {id, question, answer, qid} (deduplicated, n_samples)
        doc_id_to_text — {doc_id: text}
        qid_to_rel_ids — {base_qid: [relevant_doc_id, ...]}
    """
    # Load answers — deduplicate by question
    seen_questions: set = set()
    samples: List[Dict] = []
    with open(answers_file) as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            rec = json.loads(line)
            q = rec["question"]
            if q not in seen_questions:
                seen_questions.add(q)
                rec["qid"] = _base_qid(rec["id"])
                samples.append(rec)

    # Load docs
    doc_id_to_text: Dict[str, str] = {}
    qid_to_rel_ids: Dict[str, List[str]] = defaultdict(list)
    with open(docs_file) as f:
        for line in f:
            d = json.loads(line)
            doc_id_to_text[d["id"]] = d["text"]
            if _is_relevant(d["id"]):
                qid_to_rel_ids[_base_qid(d["id"])].append(d["id"])

    return samples, doc_id_to_text, qid_to_rel_ids


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Pre-computed Retrieval Results Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_precomputed_retrieval(csv_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Load retrieval metrics from a per_query_results.csv produced by
    evaluate_retrieval.py.

    Returns a dict keyed by (question_id, retriever) → {metric: value}
    so that evaluate.py can skip recomputing retrieval when these are
    already available.

    Expected CSV columns (at minimum):
        question_id, retriever, recall_at_k, mrr, map, ndcg_at_k
    """
    result: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return result

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["question_id"].strip(), row["retriever"].strip())
            result[key] = {
                "recall_at_k": float(row.get("recall_at_k", 0)),
                "mrr":         float(row.get("mrr",         0)),
                "map":         float(row.get("map",         0)),
                "ndcg_at_k":   float(row.get("ndcg_at_k",  0)),
                "retrieved_doc_ids": row.get("retrieved_doc_ids", ""),
            }
    print(f"  Loaded {len(result)} pre-computed retrieval rows from {csv_path}.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Text Normalisation (shared by EM, Token-F1, Explanation-F1)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """Lowercase, remove punctuation and articles — standard SQuAD normalisation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokenise(text: str) -> List[str]:
    return _normalise_text(text).split()


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (same formula used in SQuAD and HotpotQA official eval)."""
    pred_toks = _tokenise(pred)
    gold_toks = _tokenise(gold)
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec  = len(common) / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalise_text(pred) == _normalise_text(gold) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Retrieval Effectiveness Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _dcg(relevances: List[int]) -> float:
    """Discounted Cumulative Gain for a ranked list of binary relevance labels."""
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    gold_rel_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """
    Standard IR metrics for one query.

    retrieved_ids  — ordered list of retrieved doc IDs (rank 1 first)
    gold_rel_ids   — set of ground-truth relevant doc IDs
    k              — cutoff for NDCG/Recall

    Returns:
        recall_at_k, mrr, map, ndcg_at_k
    """
    gold_set = set(gold_rel_ids)
    n_rel    = len(gold_set)
    if n_rel == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "map": 0.0, "ndcg_at_k": 0.0}

    ranked_k = retrieved_ids[:k]

    # Recall@K
    recall_at_k = len(set(ranked_k) & gold_set) / n_rel

    # MRR — rank of first relevant doc in the full retrieved list
    mrr = 0.0
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in gold_set:
            mrr = 1.0 / rank
            break

    # MAP — average precision across all positions
    hits, precision_sum = 0, 0.0
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in gold_set:
            hits += 1
            precision_sum += hits / rank
    ap = precision_sum / n_rel

    # NDCG@K
    rel_labels = [1 if did in gold_set else 0 for did in ranked_k]
    dcg        = _dcg(rel_labels)
    ideal_rels = sorted(rel_labels, reverse=True)
    idcg       = _dcg(ideal_rels)
    ndcg_at_k  = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_k": recall_at_k,
        "mrr":         mrr,
        "map":         ap,
        "ndcg_at_k":   ndcg_at_k,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Explanation Quality Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_explanation_metrics(
    importance_dict: Dict[str, float],
    gold_answer: str,
) -> Dict[str, float]:
    """
    Measures how well the explanation importance ranking surfaces units that
    contain the gold answer — a proxy since we don't have sentence-level
    supporting_facts labels.

    Explanation Token-F1 : token F1 between the top-1 unit and gold answer.
    Explanation MRR      : reciprocal rank of the first unit whose text
                           contains the gold answer (case-insensitive substring).

    importance_dict — {unit_text: normalised_importance_score}
    gold_answer     — gold answer string
    """
    if not importance_dict:
        return {"expl_f1": 0.0, "expl_mrr": 0.0}

    # Rank units from highest to lowest importance
    ranked_units = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Explanation Token-F1 (top-1 unit vs gold)
    top_unit_text = ranked_units[0][0]
    expl_f1 = token_f1(top_unit_text, gold_answer)

    # Explanation MRR — rank of first unit that contains the gold answer
    gold_norm = _normalise_text(gold_answer)
    expl_mrr  = 0.0
    for rank, (unit_text, _) in enumerate(ranked_units, start=1):
        if gold_norm and gold_norm in _normalise_text(unit_text):
            expl_mrr = 1.0 / rank
            break

    return {"expl_f1": expl_f1, "expl_mrr": expl_mrr}


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Confidence Signal — Spearman Rank Correlation
# ──────────────────────────────────────────────────────────────────────────────

def compute_spearman(
    dissim_values: List[float],
    confidence_drop_values: List[float],
) -> Dict[str, float]:
    """
    Spearman rank correlation between response dissimilarity (w'_i) and
    confidence drop (Δc_i) across all (query, unit) pairs.

    A low correlation means the two signals are complementary and that
    confidence_retrieval_fusion genuinely adds information beyond the baseline.
    """
    if len(dissim_values) < 3:
        print("[WARN] Too few data points for Spearman correlation.")
        return {"spearman_rho": float("nan"), "spearman_p": float("nan")}

    rho, pval = spearmanr(dissim_values, confidence_drop_values)
    return {"spearman_rho": float(rho), "spearman_p": float(pval)}


# ──────────────────────────────────────────────────────────────────────────────
# 9.  API Calls — /explain endpoint
# ──────────────────────────────────────────────────────────────────────────────

def call_explain_api(
    api_url: str,
    question: str,
    retriever: str,
    k_docs: int,
    explanation_level: str,
    importance_mode: str,
    alpha: float,
    debug: bool = False,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    POST /explain and return the parsed JSON response.
    Returns None if the request fails.
    """
    payload = {
        "question":         question,
        "context":          "",          # RAG mode — let the retriever fetch
        "retriever":        retriever,
        "top_k_docs":       k_docs,
        "explanation_level": explanation_level,
        "perturber":        "leave_one_out",
        "comparator":       "semantic",
        "importance_mode":  importance_mode,
        "alpha":            alpha,
        "debug":            debug,
        "debug_max_perturbations": 10,
    }
    try:
        resp = requests.post(f"{api_url}/explain", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [API ERROR] {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 10. HuggingFace Inference API — multi-model generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_via_hf_api(
    model_id: str,
    prompt: str,
    hf_token: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Call the HuggingFace Inference API (serverless, free tier) for text generation.
    Returns the generated text string, or empty string on failure.
    """
    from huggingface_hub import InferenceClient
    try:
        client = InferenceClient(model=model_id, token=hf_token)
        result = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        # result is a string for text_generation
        return result.strip() if isinstance(result, str) else str(result).strip()
    except Exception as exc:
        print(f"  [HF API ERROR] model={model_id}: {exc}")
        return ""


def retrieve_for_question(
    api_url: str,
    question: str,
    retriever: str,
    k_docs: int,
    timeout: int = 30,
) -> List[Dict]:
    """
    Call GET /retrieve to get top-k docs for a question.
    Returns a list of {doc: {id, text, meta}, score} dicts.
    """
    try:
        resp = requests.get(
            f"{api_url}/retrieve",
            params={"q": question, "k": k_docs, "retriever_name": retriever},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return []   # caller handles empty list; connectivity checked at startup


def check_api_reachable(api_url: str, timeout: int = 5) -> bool:
    """
    Probe the API with a lightweight health-check request.
    Returns True if the server responds, False otherwise.
    """
    try:
        resp = requests.get(f"{api_url}/health", timeout=timeout)
        return resp.status_code < 500
    except Exception:
        pass
    # Fallback: try the root path
    try:
        resp = requests.get(api_url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 11. BERTScore — batch evaluation after collecting all predictions
# ──────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
) -> Dict[str, float]:
    """
    Computes mean BERTScore P, R, F1 using the bert_score library.
    Uses roberta-large by default (standard in literature).
    """
    from bert_score import score as bs_score
    print("  Computing BERTScore (this may take a minute)...")
    P, R, F = bs_score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall":    float(R.mean()),
        "bertscore_f1":        float(F.mean()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 12. RAGAS — batch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ragas_metrics(
    questions:   List[str],
    predictions: List[str],
    contexts:    List[List[str]],
    references:  List[str],
) -> Dict[str, float]:
    """
    Computes RAGAS metrics:
      - context_precision   : fraction of retrieved context relevant to the question
      - context_recall      : fraction of gold answer info covered by context
      - faithfulness        : is the answer grounded in the context?
      - answer_correctness  : semantic + factual match with gold answer

    Requires RAGAS >= 0.2.  Returns empty dict with warning if unavailable.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
        )
        data = {
            "question":    questions,
            "answer":      predictions,
            "contexts":    contexts,
            "ground_truth": references,
        }
        ds = Dataset.from_dict(data)
        result = evaluate(
            ds,
            metrics=[context_precision, context_recall, faithfulness, answer_correctness],
        )
        return {
            "ragas_context_precision": float(result["context_precision"]),
            "ragas_context_recall":    float(result["context_recall"]),
            "ragas_faithfulness":      float(result["faithfulness"]),
            "ragas_answer_correctness": float(result["answer_correctness"]),
        }
    except Exception as exc:
        print(f"  [RAGAS WARNING] Could not compute RAGAS metrics: {exc}")
        print("  Tip: RAGAS uses an LLM judge. Set OPENAI_API_KEY or configure a local LLM.")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# 13. Main Evaluation Loop
# ──────────────────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_retrieval_and_explanation_eval(
    samples:          List[Dict],
    qid_to_rel_ids:   Dict[str, List[str]],
    args:             argparse.Namespace,
    retriever:        str,
    precomputed:      Dict[Tuple[str, str], Dict[str, float]],
    api_online:       bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, List[float]], List[float], List[float]]:
    """
    Calls /explain for each sample × each importance_mode.

    If precomputed contains an entry for (sample["id"], retriever), the
    retrieval metrics are taken from there instead of from the API response,
    saving time when evaluate_retrieval.py has already been run.

    Returns:
        mode_metrics     — {mode: {metric: mean_score}}
        mode_expl_scores — {mode: [per-sample expl_f1 list]} (for CSV)
        all_dissim       — flat list of w'_i values  (for Spearman)
        all_conf_drop    — flat list of Δc_i values   (for Spearman)
    """
    # Accumulators per importance_mode
    ret_scores:  Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    expl_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_dissim:  List[float] = []
    all_conf_drop: List[float] = []

    total        = len(samples)
    print_step   = max(1, total // 10)   # print at every 10% milestone
    for idx, sample in enumerate(samples):
        q   = sample["question"]
        ans = sample["answer"]
        qid = sample["qid"]
        gold_rel_ids = qid_to_rel_ids.get(qid, [])

        if idx == 0 or (idx + 1) % print_step == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {q[:80]}...")

        # ── Check for pre-computed retrieval metrics ───────────────────────────
        precomp_key = (sample["id"], retriever)
        precomp_ret = precomputed.get(precomp_key)  # None if not found

        # If API is offline and no precomputed data, nothing to do for this sample
        if not api_online and precomp_ret is None:
            continue

        for mode in args.importance_modes:
            needs_debug = (mode == "confidence_retrieval_fusion")

            if api_online:
                result = call_explain_api(
                    api_url          = args.api_url,
                    question         = q,
                    retriever        = retriever,
                    k_docs           = args.k_docs,
                    explanation_level= args.explanation_level,
                    importance_mode  = mode,
                    alpha            = args.alpha,
                    debug            = needs_debug,
                )
            else:
                result = None   # API offline — will use precomputed retrieval only

            # ── Retrieval metrics — from precomputed CSV or from API ────────────
            if mode == args.importance_modes[0]:
                if precomp_ret is not None:
                    # Use pre-computed values — no recalculation needed
                    for metric in ("recall_at_k", "mrr", "map", "ndcg_at_k"):
                        ret_scores["retrieval"][metric].append(precomp_ret[metric])
                elif result is not None:
                    retrieved = result.get("retrieved_docs") or []
                    retrieved_ids = [
                        r["doc"]["id"] for r in retrieved
                        if isinstance(r, dict) and "doc" in r and "id" in r["doc"]
                    ]
                    rm = compute_retrieval_metrics(retrieved_ids, gold_rel_ids, args.k_docs)
                    for metric, val in rm.items():
                        ret_scores["retrieval"][metric].append(val)

            if result is None:
                continue   # no explanation data available

            # ── Explanation metrics ──
            importance = result.get("token_importances", {})
            em = compute_explanation_metrics(importance, ans)
            for metric, val in em.items():
                expl_scores[mode][metric].append(val)

            # ── Confidence signal: collect w'_i and Δc_i for Spearman ──
            if needs_debug and result.get("details"):
                details = result["details"]
                for unit_key, info in details.items():
                    if unit_key == "_baseline" or not isinstance(info, dict):
                        continue
                    dissim     = info.get("raw_dissimilarity")
                    conf_drop  = info.get("confidence_drop")
                    if dissim is not None and conf_drop is not None:
                        all_dissim.append(float(dissim))
                        all_conf_drop.append(float(conf_drop))

    # Aggregate
    mode_metrics: Dict[str, Dict[str, float]] = {}

    # Retrieval (mode-independent)
    mode_metrics["retrieval"] = {
        metric: _mean(vals)
        for metric, vals in ret_scores["retrieval"].items()
    }

    # Explanation (per mode)
    for mode in args.importance_modes:
        mode_metrics[f"explanation_{mode}"] = {
            metric: _mean(vals)
            for metric, vals in expl_scores[mode].items()
        }

    return mode_metrics, dict(expl_scores), all_dissim, all_conf_drop


def run_generation_eval(
    samples:   List[Dict],
    args:      argparse.Namespace,
    hf_token:  str,
) -> Dict[str, Dict[str, float]]:
    """
    For each HuggingFace model, retrieves context for each question via the
    running API, generates an answer using the HF Inference API, and computes
    EM, Token-F1, BERTScore, and RAGAS metrics.

    Returns: {model_id: {metric: score}}
    """
    model_results: Dict[str, Dict[str, float]] = {}

    for model_id in args.models:
        print(f"\n  === Evaluating model: {model_id} ===")
        predictions: List[str] = []
        references:  List[str] = []
        questions:   List[str] = []
        contexts:    List[List[str]] = []
        n_total    = len(samples)
        print_step = max(1, n_total // 10)
        api_errors = 0

        for idx, sample in enumerate(samples):
            q   = sample["question"]
            ans = sample["answer"]

            if idx == 0 or (idx + 1) % print_step == 0 or (idx + 1) == n_total:
                print(f"    [{idx+1}/{n_total}] {q[:70]}...")

            # Retrieve context
            retrieved = retrieve_for_question(
                args.api_url, q, args.retriever, args.k_docs
            )
            if not retrieved:
                api_errors += 1
                if api_errors == 3:
                    print(f"    [WARN] API unreachable — skipping remaining {n_total - idx - 1} questions for {model_id}.")
                    break
            retrieved_texts = [
                r["doc"]["text"] for r in retrieved
                if isinstance(r, dict) and "doc" in r and "text" in r["doc"]
            ]
            if not retrieved_texts:
                continue

            context_str = "\n".join(retrieved_texts)
            prompt      = f"Context: {context_str}\nQuestion: {q}\nAnswer:"

            # Generate via HF Inference API
            pred = generate_via_hf_api(model_id, prompt, hf_token, args.max_new_tokens)
            if not pred:
                continue

            predictions.append(pred)
            references.append(ans)
            questions.append(q)
            contexts.append(retrieved_texts)

        if not predictions:
            print(f"  [WARN] No predictions collected for {model_id}. Skipping.")
            continue

        # EM and Token-F1
        em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [token_f1(p, r)    for p, r in zip(predictions, references)]

        scores: Dict[str, float] = {
            "n_samples":  float(len(predictions)),
            "exact_match": _mean(em_scores),
            "token_f1":    _mean(f1_scores),
        }

        # BERTScore
        bs = compute_bertscore(predictions, references)
        scores.update(bs)

        # RAGAS
        ragas_scores = compute_ragas_metrics(questions, predictions, contexts, references)
        scores.update(ragas_scores)

        model_results[model_id] = scores
        print(f"  EM={scores['exact_match']:.4f}  F1={scores['token_f1']:.4f}  "
              f"BERTScore-F1={scores.get('bertscore_f1', float('nan')):.4f}")

    return model_results


# ──────────────────────────────────────────────────────────────────────────────
# 14. Results Output
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(title: str, rows: List[Dict[str, Any]], key_col: str) -> None:
    """Print a simple aligned table to stdout."""
    if not rows:
        return
    cols = list(rows[0].keys())
    col_w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    sep   = "  ".join("-" * col_w[c] for c in cols)
    hdr   = "  ".join(c.ljust(col_w[c]) for c in cols)
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(col_w[c]) for c in cols))


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ──────────────────────────────────────────────────────────────────────────────
# 15. Comparison Plots
# ──────────────────────────────────────────────────────────────────────────────

RETRIEVER_COLOURS = {"dense": "#2563eb", "bm25": "#16a34a", "hybrid": "#dc2626"}
METRIC_LABELS = {
    "recall_at_k": "Recall@K", "mrr": "MRR", "map": "MAP", "ndcg_at_k": "NDCG@K"
}


def plot_retriever_comparison(
    retriever_metrics: Dict[str, Dict[str, float]],
    k: int,
    out_dir: str,
) -> None:
    """
    Grouped bar chart comparing all evaluated retrievers across 4 IR metrics.
    retriever_metrics: {retriever_name: {metric: mean_score}}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics    = list(METRIC_LABELS.keys())
        labels     = [METRIC_LABELS[m] for m in metrics]
        retrievers = sorted(retriever_metrics.keys())
        n          = len(retrievers)
        x          = np.arange(len(labels))
        width      = 0.25
        offsets    = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, ret in enumerate(retrievers):
            values = [retriever_metrics[ret].get(m, 0.0) for m in metrics]
            bars   = ax.bar(x + offsets[i], values, width,
                            label=ret.upper(),
                            color=RETRIEVER_COLOURS.get(ret, "#888888"),
                            edgecolor="white", linewidth=0.6)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Retrieval Effectiveness Comparison  (k={k})", fontsize=12)
        ax.legend(title="Retriever", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        path = os.path.join(plots_dir, "retriever_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {path}")
    except Exception as exc:
        print(f"  [WARN] Retriever comparison plot failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 16. Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve HF token
    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("[WARN] No HuggingFace token found. Set --hf_token or HF_TOKEN env var.")

    # Validate model count
    if not (1 <= len(args.models) <= 5):
        print("[ERROR] Provide between 1 and 5 models via --models.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RAG-Ex Evaluation")
    print(f"  Models     : {args.models}")
    print(f"  N Samples  : {args.n_samples}")
    print(f"  Retrievers : {args.retrievers}  k={args.k_docs}")
    print(f"  Modes      : {args.importance_modes}")
    if args.retrieval_results_csv:
        print(f"  Pre-computed retrieval: {args.retrieval_results_csv}")
    print(f"{'='*60}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[1/4] Loading dataset...")
    samples, doc_id_to_text, qid_to_rel_ids = load_data(
        args.answers_file, args.docs_file, args.n_samples
    )
    print(f"      {len(samples)} unique questions loaded.")

    # ── Load pre-computed retrieval results (optional) ─────────────────────────
    precomputed: Dict[Tuple[str, str], Dict[str, float]] = {}
    if args.retrieval_results_csv:
        print("\n  Loading pre-computed retrieval metrics...")
        precomputed = load_precomputed_retrieval(args.retrieval_results_csv)

    # ── Check API connectivity once ────────────────────────────────────────────
    print(f"\n  Checking API connectivity at {args.api_url} ...")
    api_online = check_api_reachable(args.api_url)
    if api_online:
        print("  API is reachable. Explanation and generation evaluation will run.")
    else:
        print(f"  [WARN] API at {args.api_url} is NOT reachable.")
        print("         → Explanation + generation evaluation will be SKIPPED.")
        print("         Start the server with: uvicorn app.api:app --port 8000")

    # ── Retrieval + Explanation evaluation — looped per retriever ──────────────
    print("\n[2/4] Retrieval + Explanation evaluation (via /explain API)...")

    # Accumulate retrieval metrics per retriever for comparison plot
    retriever_ret_metrics: Dict[str, Dict[str, float]] = {}

    all_retr_expl_rows = []
    all_spearman_rows  = []

    for retriever in args.retrievers:
        precomp_count = sum(1 for (_, r) in precomputed if r == retriever)
        src = f"pre-computed ({precomp_count} rows)" if precomp_count else "API (live)"
        print(f"\n  ▶  Retriever: {retriever.upper()}  [retrieval source: {src}]")

        if not api_online and precomp_count == 0:
            # Nothing to compute — no API and no pre-computed data for this retriever
            print(f"  [SKIP] No API and no pre-computed data for '{retriever}'. Skipping.")
            continue

        mode_metrics, _, all_dissim, all_conf_drop = run_retrieval_and_explanation_eval(
            samples, qid_to_rel_ids, args, retriever, precomputed, api_online
        )

        # Store retrieval metrics for per-retriever comparison plot
        retriever_ret_metrics[retriever] = mode_metrics.get("retrieval", {})

        # ── Spearman per retriever ─────────────────────────────────────────────
        if "confidence_retrieval_fusion" in args.importance_modes and all_dissim:
            sp = compute_spearman(all_dissim, all_conf_drop)
            all_spearman_rows.append({
                "retriever":   retriever,
                "signal_pair": "w'_i vs Δc_i",
                **{k: fmt(v) for k, v in sp.items()},
            })

        # Build rows for CSV
        ret_row = {
            "retriever": retriever,
            "metric_group": "retrieval",
            **{k: fmt(v) for k, v in mode_metrics.get("retrieval", {}).items()},
        }
        all_retr_expl_rows.append(ret_row)

        for mode in args.importance_modes:
            key = f"explanation_{mode}"
            if key in mode_metrics:
                all_retr_expl_rows.append({
                    "retriever":        retriever,
                    "metric_group":     f"explanation_{mode}",
                    **{k: fmt(v) for k, v in mode_metrics[key].items()},
                })

    # ── Generation quality evaluation per model (uses first retriever) ─────────
    print("\n[3/4] Generation quality evaluation per HuggingFace model...")
    args.retriever = args.retrievers[0]
    if not api_online:
        print("  [SKIP] API not reachable — skipping generation evaluation.")
        generation_results = {}
    else:
        generation_results = run_generation_eval(samples, args, hf_token)

    # ── Display results ────────────────────────────────────────────────────────

    # Table: Retrieval metrics per retriever
    ret_display = [
        {"retriever": r, **{k: fmt(v) for k, v in metrics.items()}}
        for r, metrics in retriever_ret_metrics.items()
    ]
    _print_table("RETRIEVAL EFFECTIVENESS — ALL RETRIEVERS", ret_display, "retriever")

    # Table: Explanation metrics
    expl_display = [r for r in all_retr_expl_rows if "explanation" in r.get("metric_group", "")]
    if expl_display:
        _print_table("EXPLANATION QUALITY", expl_display, "retriever")

    # Table: Spearman
    if all_spearman_rows:
        _print_table("CONFIDENCE SIGNAL — SPEARMAN CORRELATION", all_spearman_rows, "retriever")

    # Table: Generation metrics
    gen_rows = []
    for model_id, scores in generation_results.items():
        row = {"model": model_id}
        for k, v in scores.items():
            if k != "n_samples":
                row[k] = fmt(v)
        gen_rows.append(row)
    _print_table("GENERATION QUALITY PER MODEL", gen_rows, "model")

    # ── Comparison plot: all retrievers side-by-side ───────────────────────────
    if len(retriever_ret_metrics) > 1:
        print("\n  Generating retriever comparison plot...")
        plot_retriever_comparison(retriever_ret_metrics, args.k_docs, args.output_dir)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Saving results...")
    save_csv(os.path.join(args.output_dir, "retrieval_explanation_metrics.csv"), all_retr_expl_rows)
    if all_spearman_rows:
        save_csv(os.path.join(args.output_dir, "spearman_metrics.csv"), all_spearman_rows)
    save_csv(os.path.join(args.output_dir, "generation_metrics.csv"), gen_rows)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
