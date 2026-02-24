"""
Retrieval Effectiveness Evaluation
====================================
Evaluates all three retrievers (dense, bm25, hybrid) on the HotpotQA dataset
at data/hotpot/ using standard IR metrics.

Relevance labels are derived from document IDs:
  Relevant   →  {qid}_{n}          (e.g. 5a7a0693..._1, ..._2)
  Distractor →  {qid}_irr_{n}      (e.g. 5a7a0693..._irr_1)

Outputs (written to --output_dir, default: results/retrieval/):
  per_query_results.csv   — one row per (question × retriever)
  summary_metrics.csv     — mean metrics per retriever
  plots/bar_comparison.png
  plots/boxplot_ndcg.png
  plots/boxplot_mrr.png
  plots/recall_vs_k.png   — Recall@K curve for k=1..10

Usage:
  ragex/bin/python scripts/evaluate_retrieval.py \\
      --n_samples 500 \\
      --k_docs 3 \\
      --output_dir results/retrieval/
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ── ensure project root is on sys.path so 'app' is importable ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── silence noisy import warnings before loading retriever modules ─────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CLI Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval effectiveness evaluation across dense, bm25, and hybrid.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--n_samples",   type=int, default=None,
                   help="Number of unique questions to evaluate (default: all).")
    p.add_argument("--k_docs",      type=int, default=3,
                   help="Number of documents to retrieve (default: 3).")
    p.add_argument("--answers_file", default="data/hotpot/hotpot_answers.jsonl",
                   help="Path to hotpot_answers.jsonl.")
    p.add_argument("--docs_file",    default="data/hotpot/hotpot_docs.jsonl",
                   help="Path to hotpot_docs.jsonl (used for relevance labels).")
    p.add_argument("--output_dir",   default="results/retrieval",
                   help="Directory to save CSV and plots (default: results/retrieval/).")
    p.add_argument("--retrievers",   nargs="+", default=["dense", "bm25", "hybrid"],
                   choices=["dense", "bm25", "hybrid"],
                   help="Retrievers to evaluate (default: all three).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Data Loading & Relevance Labels
# ──────────────────────────────────────────────────────────────────────────────

def _base_qid(doc_id: str) -> str:
    """Remove trailing _{n} suffix to get the base question ID."""
    return re.sub(r"_\d+$", "", doc_id)


def _is_relevant(doc_id: str) -> bool:
    """Documents without '_irr_' in their ID are gold-relevant."""
    return "_irr_" not in doc_id


def load_samples(answers_file: str, n_samples) -> List[Dict]:
    """
    Load up to n_samples unique questions from hotpot_answers.jsonl.
    If n_samples is None, load all rows.
    Deduplicates by question text (multiple answer rows can share one question).
    """
    seen, samples = set(), []
    with open(answers_file) as f:
        for line in f:
            if n_samples is not None and len(samples) >= n_samples:
                break
            rec = json.loads(line)
            if rec["question"] not in seen:
                seen.add(rec["question"])
                rec["qid"] = _base_qid(rec["id"])
                samples.append(rec)
    return samples


def build_relevance_map(docs_file: str) -> Dict[str, List[str]]:
    """
    Returns {base_qid: [relevant_doc_id, ...]} from the docs file.
    Distractor docs (containing '_irr_') are excluded.
    """
    qid_to_rel: Dict[str, List[str]] = defaultdict(list)
    with open(docs_file) as f:
        for line in f:
            d = json.loads(line)
            if _is_relevant(d["id"]):
                qid_to_rel[_base_qid(d["id"])].append(d["id"])
    return dict(qid_to_rel)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  IR Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def compute_metrics(
    retrieved: List[Dict],  # list of {doc:{id,...}, score:float}
    gold_rel_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """
    Computes Recall@K, MRR, MAP, NDCG@K for a single query.

    retrieved     — ranked list from the retriever (rank-1 first)
    gold_rel_ids  — list of relevant document IDs
    k             — cutoff applied to retrieved list
    """
    gold_set = set(gold_rel_ids)
    n_rel    = len(gold_set)

    if n_rel == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "map": 0.0, "ndcg_at_k": 0.0}

    retrieved_ids = [r["doc"]["id"] for r in retrieved]
    top_k_ids     = retrieved_ids[:k]

    # ── Recall@K ──────────────────────────────────────────────────────────────
    recall_at_k = len(set(top_k_ids) & gold_set) / n_rel

    # ── MRR ───────────────────────────────────────────────────────────────────
    mrr = 0.0
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in gold_set:
            mrr = 1.0 / rank
            break

    # ── MAP ───────────────────────────────────────────────────────────────────
    hits = 0
    precision_sum = 0.0
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in gold_set:
            hits += 1
            precision_sum += hits / rank
    ap = precision_sum / n_rel

    # ── NDCG@K ────────────────────────────────────────────────────────────────
    rel_labels = [1 if did in gold_set else 0 for did in top_k_ids]
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


def recall_at_k_curve(
    retrieved_ids: List[str],
    gold_rel_ids: List[str],
    max_k: int,
) -> Dict[int, float]:
    """Returns {k: recall@k} for k in 1..max_k."""
    gold_set = set(gold_rel_ids)
    n_rel    = max(len(gold_set), 1)
    return {
        k: len(set(retrieved_ids[:k]) & gold_set) / n_rel
        for k in range(1, max_k + 1)
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Retriever Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_retriever_fn(name: str):
    """
    Returns a callable retrieve(query, k) -> List[{doc, score}] for the given
    retriever name. Loads index once (module-level cache handles repeats).
    """
    if name == "dense":
        from app.retrievers.dense_faiss import retrieve
    elif name == "bm25":
        from app.retrievers.bm25 import retrieve
    elif name == "hybrid":
        from app.retrievers.hybrid import retrieve
    else:
        raise ValueError(f"Unknown retriever: {name}")
    return retrieve


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Core Evaluation Loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_retriever(
    name: str,
    retrieve_fn,
    samples: List[Dict],
    qid_to_rel: Dict[str, List[str]],
    k: int,
    max_k_curve: int = 10,
) -> Tuple[List[Dict], Dict[int, List[float]]]:
    """
    Evaluates one retriever over all samples.

    Returns:
        rows         — per-query result dicts (for per_query_results.csv)
        recall_curve — {k: [per-query recall@k values]}  (for Recall@K plot)
    """
    rows: List[Dict] = []
    recall_curve: Dict[int, List[float]] = defaultdict(list)

    total = len(samples)
    for idx, sample in enumerate(samples):
        q   = sample["question"]
        qid = sample["qid"]
        gold_rel_ids = qid_to_rel.get(qid, [])

        try:
            retrieved = retrieve_fn(q, k)
        except Exception as exc:
            print(f"    [ERROR] retriever={name} q={q[:50]} → {exc}")
            retrieved = []

        retrieved_ids    = [r["doc"]["id"]  for r in retrieved]
        retrieved_scores = [r["score"]      for r in retrieved]

        metrics = compute_metrics(retrieved, gold_rel_ids, k)

        # Recall@K curve data (k=1..max_k_curve)
        try:
            full_retrieved = retrieve_fn(q, max_k_curve)
            full_ids = [r["doc"]["id"] for r in full_retrieved]
        except Exception:
            full_ids = retrieved_ids
        curve = recall_at_k_curve(full_ids, gold_rel_ids, max_k_curve)
        for kval, rec in curve.items():
            recall_curve[kval].append(rec)

        rows.append({
            "question_id":         sample["id"],
            "question":            q,
            "retriever":           name,
            "k_docs":              k,
            "n_gold_relevant_docs": len(gold_rel_ids),
            "gold_relevant_doc_ids": "|".join(gold_rel_ids),
            "retrieved_doc_ids":   "|".join(retrieved_ids),
            "retrieval_scores":    "|".join(f"{s:.4f}" for s in retrieved_scores),
            "n_retrieved_relevant": len(set(retrieved_ids) & set(gold_rel_ids)),
            "recall_at_k":         round(metrics["recall_at_k"], 4),
            "mrr":                 round(metrics["mrr"],         4),
            "map":                 round(metrics["map"],         4),
            "ndcg_at_k":           round(metrics["ndcg_at_k"],   4),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            mean_ndcg = sum(r["ndcg_at_k"] for r in rows) / len(rows)
            print(f"    [{idx+1}/{total}]  mean NDCG@{k}={mean_ndcg:.4f}")

    return rows, dict(recall_curve)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CSV Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


def build_summary_rows(all_rows: List[Dict]) -> List[Dict]:
    """Aggregate per-query rows into one summary row per retriever."""
    by_retriever: Dict[str, List[Dict]] = defaultdict(list)
    for row in all_rows:
        by_retriever[row["retriever"]].append(row)

    summary = []
    for name, rows in sorted(by_retriever.items()):
        n = len(rows)
        summary.append({
            "retriever":        name,
            "n_questions":      n,
            "k_docs":           rows[0]["k_docs"],
            "mean_recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4),
            "mean_mrr":         round(sum(r["mrr"]         for r in rows) / n, 4),
            "mean_map":         round(sum(r["map"]         for r in rows) / n, 4),
            "mean_ndcg_at_k":   round(sum(r["ndcg_at_k"]  for r in rows) / n, 4),
        })
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Plots
# ──────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "mean_recall_at_k": "Recall@K",
    "mean_mrr":         "MRR",
    "mean_map":         "MAP",
    "mean_ndcg_at_k":   "NDCG@K",
}

RETRIEVER_COLOURS = {
    "dense":  "#2563eb",   # blue
    "bm25":   "#16a34a",   # green
    "hybrid": "#dc2626",   # red
}


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def plot_bar_comparison(summary_rows: List[Dict], k: int, out_dir: str) -> None:
    """Grouped bar chart: 4 metrics × 3 retrievers side-by-side."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics  = list(METRIC_LABELS.keys())
    labels   = [METRIC_LABELS[m] for m in metrics]
    retrievers = [r["retriever"] for r in summary_rows]

    x      = np.arange(len(labels))
    width  = 0.25
    n      = len(retrievers)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(summary_rows):
        values = [row[m] for m in metrics]
        bars   = ax.bar(x + offsets[i], values, width,
                        label=row["retriever"].upper(),
                        color=RETRIEVER_COLOURS.get(row["retriever"], "#888888"),
                        edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Retrieval Effectiveness — Dense vs BM25 vs Hybrid  (k={k})", fontsize=12)
    ax.legend(title="Retriever", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    _save_fig(fig, os.path.join(out_dir, "plots", "bar_comparison.png"))
    plt.close(fig)


def plot_boxplot(all_rows: List[Dict], metric: str, k: int, out_dir: str) -> None:
    """Box plot showing per-query distribution of a metric across retrievers."""
    import matplotlib.pyplot as plt

    by_ret = defaultdict(list)
    for row in all_rows:
        by_ret[row["retriever"]].append(row[metric])

    retrievers = sorted(by_ret.keys())
    data     = [by_ret[r] for r in retrievers]
    colours  = [RETRIEVER_COLOURS.get(r, "#888") for r in retrievers]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(retrievers) + 1))
    ax.set_xticklabels([r.upper() for r in retrievers], fontsize=11)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
    ax.set_title(
        f"Per-Query {METRIC_LABELS.get(metric, metric)} Distribution  (k={k})",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fname = f"boxplot_{metric.replace('_at_k','').replace('mean_','')}.png"
    _save_fig(fig, os.path.join(out_dir, "plots", fname))
    plt.close(fig)


def plot_recall_curve(
    curve_data: Dict[str, Dict[int, List[float]]],
    out_dir: str,
) -> None:
    """
    Line chart: mean Recall@K for k=1..10, one line per retriever.
    Shows how quickly each retriever accumulates relevant documents.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for ret_name, curve in sorted(curve_data.items()):
        ks     = sorted(curve.keys())
        means  = [sum(curve[k]) / len(curve[k]) for k in ks]
        ax.plot(ks, means,
                marker="o", markersize=5, linewidth=2,
                label=ret_name.upper(),
                color=RETRIEVER_COLOURS.get(ret_name, "#888"))

    ax.set_xlabel("K  (number of retrieved documents)", fontsize=11)
    ax.set_ylabel("Mean Recall@K", fontsize=11)
    ax.set_title("Recall@K Curve — Dense vs BM25 vs Hybrid", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Retriever", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    _save_fig(fig, os.path.join(out_dir, "plots", "recall_vs_k.png"))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Console Summary Table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(summary_rows: List[Dict]) -> None:
    if not summary_rows:
        print("  [WARN] No results to display.")
        return
    cols     = ["retriever", "n_questions", "k_docs",
                "mean_recall_at_k", "mean_mrr", "mean_map", "mean_ndcg_at_k"]
    col_w    = {c: max(len(c), max(len(str(r[c])) for r in summary_rows)) for c in cols}
    sep      = "  ".join("-" * col_w[c] for c in cols)
    hdr      = "  ".join(c.ljust(col_w[c]) for c in cols)
    print(f"\n{'═'*70}")
    print("  RETRIEVAL EFFECTIVENESS — SUMMARY")
    print(f"{'═'*70}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in summary_rows:
        print("  " + "  ".join(str(row[c]).ljust(col_w[c]) for c in cols))
    print(f"{'═'*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print("  Retrieval Effectiveness Evaluation")
    print(f"  Retrievers : {args.retrievers}")
    print(f"  N Samples  : {args.n_samples if args.n_samples is not None else 'all'}")
    print(f"  K docs     : {args.k_docs}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/3] Loading dataset...")
    samples      = load_samples(args.answers_file, args.n_samples)
    qid_to_rel   = build_relevance_map(args.docs_file)
    print(f"      {len(samples)} unique questions loaded.")
    avg_rel = sum(len(v) for v in qid_to_rel.values()) / max(len(qid_to_rel), 1)
    print(f"      Avg relevant docs per question: {avg_rel:.2f}\n")

    # ── Evaluate each retriever ───────────────────────────────────────────────
    print("[2/3] Running retrieval...")
    all_rows:     List[Dict]                      = []
    curve_data:   Dict[str, Dict[int, List[float]]] = {}

    for name in args.retrievers:
        print(f"\n  ▶  Retriever: {name.upper()}")
        try:
            retrieve_fn = make_retriever_fn(name)
        except Exception as exc:
            print(f"    [ERROR] Could not initialise {name}: {exc}")
            continue

        rows, curve = evaluate_retriever(
            name, retrieve_fn, samples, qid_to_rel, args.k_docs
        )
        all_rows.extend(rows)
        curve_data[name] = curve

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\n[3/3] Saving results and plots...")
    per_query_path = os.path.join(args.output_dir, "per_query_results.csv")
    summary_path   = os.path.join(args.output_dir, "summary_metrics.csv")
    save_csv(per_query_path, all_rows)

    summary_rows = build_summary_rows(all_rows)
    save_csv(summary_path, summary_rows)

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        plot_bar_comparison(summary_rows, args.k_docs, args.output_dir)
        plot_boxplot(all_rows, "ndcg_at_k", args.k_docs, args.output_dir)
        plot_boxplot(all_rows, "mrr",        args.k_docs, args.output_dir)
        if curve_data:
            plot_recall_curve(curve_data, args.output_dir)
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plots.")
    except Exception as exc:
        print(f"  [WARN] Plot generation failed: {exc}")

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
