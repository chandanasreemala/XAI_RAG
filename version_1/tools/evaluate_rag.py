#!/usr/bin/env python3
"""
Evaluate Retriever and RAG pipeline

Usage examples:
  # Retriever-only evaluation (top 10)
  python tools/evaluate_rag.py --dataset-prefix mydataset --mode retriever --topk 10 --retriever dense

  # Full RAG pipeline evaluation (generate answers)
  python tools/evaluate_rag.py --dataset-prefix mydataset --mode rag --topk 5 --retriever dense --output results.json

  # Run both
  python tools/evaluate_rag.py --dataset-prefix mydataset --mode both --topk 5 --retriever dense --output results.json

Assumptions about files:
  {dataset_prefix}_docs.jsonl  : each line is a JSON object with at least {"id": "<docid>", "text": "<doc text>", "meta": {...}}
  {dataset_prefix}_answers.jsonl : each line is a JSON object with at least {"id": "<qid>", "answer": "<gold answer>", "meta": {...}}

Notes about schema flexibility:
  - For queries/questions: the script looks for the question text in the answers JSONL entry in these places (in order):
      1) top-level key "question"
      2) top-level key "query"
      3) answers_obj["meta"]["question"] or ["meta"]["query"]
     If none is found, the script will raise an informative error and show examples of a valid format.
  - For relevant doc ids (for retriever evaluation): the script looks for:
      1) answers_obj["meta"]["relevant_doc_ids"] (list)
      2) answers_obj["meta"]["gold_doc_ids"] or ["meta"]["doc_ids"]
     If none found, the script will attempt to derive relevant docs by substring-matching the answer text inside doc texts. This is a heuristic and may not be perfect.
  - Retrieval function: expects `from app.retriever import retrieve` and supports either retrieve(q, k) or retrieve(q, k, retriever=name)
  - Generation function: expects `from app.generator import get_generator, generate_answer`
    The script wraps both to be backward-compatible.

Outputs:
  - Printed summary metrics (retriever: MRR@k, MAP@k, NDCG@k; RAG: EM, avg F1, precision, recall)
  - Optionally saves per-query details to a JSON file (use --output)
"""

import json
import argparse
import math
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any

# Import your project's retrieve/generator. Adapt the imports if your paths differ.
try:
    from app.retriever import retrieve  # type: ignore
except Exception as e:
    raise RuntimeError(f"Could not import retrieve from app.retriever: {e}")

try:
    from app.generator import get_generator, generate_answer  # type: ignore
except Exception:
    # We'll use generate wrapper below that tries to call generate_answer in flexible forms
    generate_answer = None  # type: ignore

# --------------------------
# Helpers: I/O for jsonl
# --------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items

# --------------------------
# Helpers: text normalization
# --------------------------
def normalize_text(s: str) -> str:
    # Lowercase, strip, remove punctuation (SQuAD-like normalization)
    s = s or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    return s

def f1_score_from_gold_pred(gold: str, pred: str) -> Tuple[float, float, float]:
    """
    Returns (f1, precision, recall) based on token overlap.
    """
    gold_toks = normalize_text(gold).split()
    pred_toks = normalize_text(pred).split()
    if len(gold_toks) == 0 and len(pred_toks) == 0:
        return 1.0, 1.0, 1.0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0, 0.0, 0.0
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def exact_match(gold: str, pred: str) -> int:
    return int(normalize_text(gold) == normalize_text(pred))

# --------------------------
# Retriever metrics
# --------------------------
def reciprocal_rank_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    for i, did in enumerate(retrieved_ids[:k], start=1):
        if did in relevant_set:
            return 1.0 / i
    return 0.0

def average_precision_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    num_rel = 0
    score = 0.0
    for i, did in enumerate(retrieved_ids[:k], start=1):
        if did in relevant_set:
            num_rel += 1
            score += num_rel / i
    if num_rel == 0:
        return 0.0
    # normalize by min(number of relevant docs, k)
    return score / min(len(relevant_set), k)

def dcg_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if did in relevant_set else 0.0
        denom = math.log2(i + 1)
        dcg += rel / denom
    return dcg

def idcg_at_k(relevant_set: set, k: int) -> float:
    # ideal DCG for binary relevance: all relevant items appear in top positions
    # If there are r relevant docs, ideal DCG = sum_{i=1..min(r,k)} (1 / log2(i+1))
    r = min(len(relevant_set), k)
    if r == 0:
        return 0.0
    return sum(1.0 / math.log2(i + 1) for i in range(1, r + 1))

def ndcg_at_k(retrieved_ids: List[str], relevant_set: set, k: int) -> float:
    idcg = idcg_at_k(relevant_set, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_set, k) / idcg

# --------------------------
# Retrieval wrapper (backwards compatible)
# --------------------------
def call_retrieve(query: str, k: int, retriever_name: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Returns list of (doc_id, score) in ranked order.
    The underlying retrieve(...) in your project may return different structures.
    We try common variants:
      - retrieve(query, k) -> list of {"doc": {"id":.., "text":..}, "score":..}
      - retrieve(query, k, retriever=name) -> same
      - or retrieve returns list of dicts {"id":..., "text":..., "score":...}
      - or retrieve returns list of (id, score)
    """
    try:
        if retriever_name is None:
            raw = retrieve(query, k)  # type: ignore
        else:
            # attempt named retriever
            raw = retrieve(query, k, retriever=retriever_name)  # type: ignore
    except TypeError:
        raw = retrieve(query, k)  # type: ignore

    # normalize raw into list of (id, score)
    out: List[Tuple[str, float]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, tuple) or isinstance(item, list):
                # (id, score)
                if len(item) >= 2:
                    out.append((str(item[0]), float(item[1])))
                    continue
            if isinstance(item, dict):
                # try nested "doc" structure
                if "doc" in item and isinstance(item["doc"], dict):
                    did = item["doc"].get("id") or item["doc"].get("doc_id") or item["doc"].get("id_str")
                    score = item.get("score", 0.0)
                    if did is None:
                        # maybe doc object has 'meta' with id
                        did = item["doc"].get("meta", {}).get("id")
                    if did is None:
                        # fallback: use doc text hash
                        did = str(hash(item["doc"].get("text", "")))[:16]
                    out.append((str(did), float(score)))
                    continue
                # flat dict form
                if "id" in item and "score" in item:
                    out.append((str(item["id"]), float(item["score"])))
                    continue
                # sometimes retrieve returns {'id':..., 'text':...} without explicit score
                if "id" in item and "text" in item:
                    out.append((str(item["id"]), float(item.get("score", 0.0))))
                    continue
    # last resort: if nothing parsed, try to coerce
    if not out and isinstance(raw, list):
        for i, r in enumerate(raw):
            out.append((str(i), 0.0))
    return out

# --------------------------
# Generation wrapper (backwards compatible)
# --------------------------
def safe_generate(gen, prompt: str, max_length: int, temperature: float,
                  return_sequence_confidence: bool = False,
                  return_token_logprobs: bool = False) -> Tuple[str, Optional[float]]:
    """
    Attempts to call generate_answer with extended args but falls back
    to simpler signatures.
    Expected outputs:
      - (answer_text) or
      - (answer_text, seq_conf) or
      - (answer_text, seq_conf, token_texts, token_logprobs)
    We return only (answer_text, seq_conf) here.
    """
    # If generate_answer is a function in your code, try calling with flexible signatures
    try:
        out = generate_answer(gen, prompt, max_length=max_length, temperature=temperature,
                              return_sequence_confidence=return_sequence_confidence,
                              return_token_logprobs=return_token_logprobs)
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], None
        return str(out), None
    except TypeError:
        # older signature generate_answer(gen, prompt, max_length, temperature)
        try:
            out = generate_answer(gen, prompt, max_length=max_length, temperature=temperature)
            if isinstance(out, tuple):
                if len(out) >= 2:
                    return out[0], out[1]
                return out[0], None
            return str(out), None
        except Exception as e:
            raise RuntimeError(f"generate_answer failed: {e}")

# --------------------------
# Eval driver
# --------------------------
def evaluate_retriever(
    dataset_prefix: str,
    topk: int,
    retriever_name: Optional[str] = None,
    out_file: Optional[str] = None,
):
    docs_path = f"{dataset_prefix}_docs.jsonl"
    ans_path = f"{dataset_prefix}_answers.jsonl"
    assert os.path.exists(docs_path), f"Missing {docs_path}"
    assert os.path.exists(ans_path), f"Missing {ans_path}"

    docs = load_jsonl(docs_path)
    answers = load_jsonl(ans_path)

    # build doc id->text map
    docid_to_text: Dict[str, str] = {}
    for d in docs:
        did = str(d.get("id") or d.get("doc_id") or d.get("id_str") or hash(d.get("text", "")))
        docid_to_text[did] = d.get("text", "")

    # build queries list: find question text and relevant doc ids for each answer entry
    queries: List[Tuple[str, str, set]] = []  # (qid, question, set(relevant_doc_ids))
    for a in answers:
        qid = str(a.get("id"))
        # find question text
        q_text = a.get("question") or a.get("query")
        if not q_text:
            meta = a.get("meta", {}) or {}
            q_text = meta.get("question") or meta.get("query")
        if not q_text:
            raise RuntimeError(
                "Couldn't find 'question' for an answer entry. Each answer item must include the query string "
                "either at top-level 'question'/'query' or in 'meta.question'. Example answer entry:\n"
                '{ "id": "q1", "question": "Who built X?", "answer": "Y", "meta": {"relevant_doc_ids": ["d1","d2"]} }\n'
            )
        # find relevant doc ids (if present)
        meta = a.get("meta", {}) or {}
        rel = set()
        for key in ("relevant_doc_ids", "gold_doc_ids", "doc_ids", "relevant_docs"):
            if key in meta and isinstance(meta[key], list):
                rel = set(str(x) for x in meta[key])
                break

        # if not present, try to derive by substring matching answer in docs
        if not rel:
            gold_answer = a.get("answer", "")
            if gold_answer:
                for did, dtext in docid_to_text.items():
                    if gold_answer.strip() and gold_answer.strip().lower() in dtext.lower():
                        rel.add(did)
        queries.append((qid, q_text, rel))

    # iterate queries and call retrieve
    metrics = {
        "MRR@k": [],
        "MAP@k": [],
        "NDCG@k": [],
    }
    per_query = []

    for qid, q_text, relevant in queries:
        ranked = call_retrieve(q_text, topk, retriever_name)
        retrieved_ids = [did for did, score in ranked]

        mrr = reciprocal_rank_at_k(retrieved_ids, relevant, topk)
        ap = average_precision_at_k(retrieved_ids, relevant, topk)
        ndcg = ndcg_at_k(retrieved_ids, relevant, topk)

        metrics["MRR@k"].append(mrr)
        metrics["MAP@k"].append(ap)
        metrics["NDCG@k"].append(ndcg)

        per_query.append({
            "qid": qid,
            "query": q_text,
            "relevant": list(relevant),
            "retrieved": retrieved_ids,
            "mrr": mrr,
            "ap": ap,
            "ndcg": ndcg,
        })

    # aggregate
    summary = {
        "MRR@k": sum(metrics["MRR@k"]) / len(metrics["MRR@k"]) if metrics["MRR@k"] else 0.0,
        "MAP@k": sum(metrics["MAP@k"]) / len(metrics["MAP@k"]) if metrics["MAP@k"] else 0.0,
        "NDCG@k": sum(metrics["NDCG@k"]) / len(metrics["NDCG@k"]) if metrics["NDCG@k"] else 0.0,
        "num_queries": len(queries),
    }

    out = {"summary": summary, "per_query": per_query}
    if out_file:
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    else:
        print(json.dumps(out, indent=2))
    return out

def evaluate_rag_pipeline(
    dataset_prefix: str,
    topk: int,
    retriever_name: Optional[str],
    out_file: Optional[str],
    gen_model_name: Optional[str] = None,
):
    docs_path = f"{dataset_prefix}_docs.jsonl"
    ans_path = f"{dataset_prefix}_answers.jsonl"
    assert os.path.exists(docs_path), f"Missing {docs_path}"
    assert os.path.exists(ans_path), f"Missing {ans_path}"

    docs = load_jsonl(docs_path)
    answers = load_jsonl(ans_path)

    # build doc id->text map and a list for retrieval use
    docid_to_text = {}
    for d in docs:
        did = str(d.get("id") or d.get("doc_id") or d.get("id_str") or hash(d.get("text", "")))
        docid_to_text[did] = d.get("text", "")

    # build queries
    queries: List[Tuple[str, str, str]] = []  # (qid, question, gold_answer)
    for a in answers:
        qid = str(a.get("id"))
        q_text = a.get("question") or a.get("query")
        if not q_text:
            meta = a.get("meta", {}) or {}
            q_text = meta.get("question") or meta.get("query")
        if not q_text:
            raise RuntimeError(
                "Couldn't find 'question' for an answer entry. Each answer item must include the query string "
                "either at top-level 'question'/'query' or in 'meta.question'."
            )
        gold = a.get("answer", "")
        queries.append((qid, q_text, gold))

    # prepare generator handle if your generator API needs it
    gen = None
    try:
        gen = get_generator(gen_model_name) if gen_model_name else getattr(__import__("app").state, "generator", None)
    except Exception:
        # fallback: try app.state if present
        try:
            from fastapi import current_app  # type: ignore
        except Exception:
            pass

    # metrics accumulators
    totals = {"EM": [], "F1": [], "Precision": [], "Recall": []}
    per_query = []

    for qid, q_text, gold in queries:
        # retrieve topk
        ranked = call_retrieve(q_text, topk, retriever_name)
        # build context as concatenation of topk retrieved doc texts
        top_docs_texts = []
        retrieved_ids = []
        for did, score in ranked:
            retrieved_ids.append(did)
            # find text from docs list or use empty
            txt = docid_to_text.get(did, "")
            top_docs_texts.append(txt)
        context = "\n\n".join([t for t in top_docs_texts if t])

        prompt = f"Context: {context}\nQuestion: {q_text}\nAnswer:"
        # call generation (safe wrapper)
        # Note: we try to pass a generator object (if available), else assume generate_answer is a wrapper that handles None
        try:
            pred, seq_conf = safe_generate(gen, prompt, max_length=256, temperature=0.0,
                                           return_sequence_confidence=False, return_token_logprobs=False)
        except Exception:
            # fallback: try to call generate_answer directly with prompt (some repos expect different signatures)
            try:
                pred = generate_answer(prompt)  # type: ignore
                seq_conf = None
            except Exception as e:
                raise RuntimeError(f"Generation failed for query {qid}: {e}")

        # compute metrics
        em = exact_match(gold, pred)
        f1, p, r = f1_score_from_gold_pred(gold, pred)

        totals["EM"].append(em)
        totals["F1"].append(f1)
        totals["Precision"].append(p)
        totals["Recall"].append(r)

        per_query.append({
            "qid": qid,
            "question": q_text,
            "gold": gold,
            "pred": pred,
            "retrieved_ids": retrieved_ids,
            "EM": em,
            "F1": f1,
            "Precision": p,
            "Recall": r,
        })

    summary = {
        "EM": float(sum(totals["EM"]) / max(1, len(totals["EM"]))),
        "F1": float(sum(totals["F1"]) / max(1, len(totals["F1"]))),
        "Precision": float(sum(totals["Precision"]) / max(1, len(totals["Precision"]))),
        "Recall": float(sum(totals["Recall"]) / max(1, len(totals["Recall"]))),
        "num_queries": len(per_query),
    }

    out = {"summary": summary, "per_query": per_query}
    if out_file:
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    else:
        print(json.dumps(out, indent=2))
    return out

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Retriever and RAG pipeline")
    parser.add_argument("--dataset-prefix", required=True, help="Prefix to dataset files, e.g. 'mydata' -> mydata_docs.jsonl mydata_answers.jsonl")
    parser.add_argument("--mode", choices=["retriever", "rag", "both"], default="both")
    parser.add_argument("--topk", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--retriever", type=str, default=None, help="Retriever name (if your retrieve supports it)")
    parser.add_argument("--output", type=str, default=None, help="File to save per-query details (JSON)")
    parser.add_argument("--gen-model", type=str, default=None, help="Generator model name if your get_generator supports building one")
    args = parser.parse_args()

    if args.mode in ("retriever", "both"):
        print("Running retriever evaluation...")
        r_out = evaluate_retriever(args.dataset_prefix, args.topk, args.retriever, out_file=(args.output and args.output + ".retriever.json"))
        print("Retriever summary:", r_out["summary"])

    if args.mode in ("rag", "both"):
        print("Running RAG pipeline evaluation...")
        rag_out = evaluate_rag_pipeline(args.dataset_prefix, args.topk, args.retriever, out_file=(args.output and args.output + ".rag.json"), gen_model_name=args.gen_model)
        print("RAG summary:", rag_out["summary"])

if __name__ == "__main__":
    main()