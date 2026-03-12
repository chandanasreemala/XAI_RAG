"""
Multi-Dataset Retrieval Evaluation
====================================
Evaluates BM25, Dense (FAISS), and Hybrid retrievers on HotpotQA and/or
TriviaQA using standard IR metrics:
    Recall@K  |  MRR  |  MAP  |  NDCG@K

Dataset ID conventions
----------------------
HotpotQA
  answers:  {qid}_{n}          →  base qid = {qid}  (strip trailing _{n})
  docs:     {qid}_{n}          →  relevant
            {qid}_irr_{n}      →  distractor (excluded)
TriviaQA
  answers:  trivia_q{n}        →  base qid = trivia_q{n}  (no strip needed)
  docs:     trivia_q{n}_{k}    →  relevant  (all docs are relevant — no _irr_)

Outputs (per dataset, saved under --output_dir/{dataset}/):
  per_query_results.csv    one row per (question × retriever)
  summary_metrics.csv      mean metrics per retriever
  plots/                   bar chart, boxplots, recall@k curve

A cross-dataset comparison plot is also saved under --output_dir/plots/.

Usage
-----
  # Quick smoke-test (5 samples):
  python scripts/eval_retrieval_multi.py --n_samples 5 --datasets hotpot trivia

  # Full run (5 000 samples):
  python scripts/eval_retrieval_multi.py --n_samples 5000 --datasets hotpot trivia

  # Single dataset:
  python scripts/eval_retrieval_multi.py --n_samples 5000 --datasets hotpot
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── ensure version_2/ is on sys.path so `app` is importable ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CFG: Dict[str, Dict[str, str]] = {
    "hotpot": {
        "answers_file": "data/hotpot/hotpot_answers.jsonl",
        "docs_file":    "data/hotpot/hotpot_docs.jsonl",
        "bm25_path":    "data/hotpot/hotpot_bm25.pkl",
        "faiss_path":   "data/hotpot/hotpot_faiss.index",
        "display":      "HotpotQA",
    },
    "trivia": {
        "answers_file": "data/trivia/trivia_answers.jsonl",
        "docs_file":    "data/trivia/trivia_docs.jsonl",
        "bm25_path":    "data/trivia/trivia_bm25.pkl",
        "faiss_path":   "data/trivia/trivia_faiss.index",
        "display":      "TriviaQA",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes — muted, eye-pleasing; no bright / harsh primaries
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVER_COLOURS: Dict[str, str] = {
    "bm25":   "#6aaa96",   # muted teal
    "dense":  "#5c85c4",   # dusty blue
    "hybrid": "#c47c5c",   # warm terracotta
}

DATASET_COLOURS: Dict[str, str] = {
    "hotpot": "#5c85c4",   # dusty blue
    "trivia":  "#c47c5c",   # warm terracotta
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-dataset retrieval evaluation (HotpotQA + TriviaQA).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", default=["hotpot", "trivia"],
        choices=list(DATASET_CFG.keys()),
        help="Datasets to evaluate (default: hotpot trivia).",
    )
    p.add_argument(
        "--retrievers", nargs="+", default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid"],
        help="Retrievers to evaluate (default: all three).",
    )
    p.add_argument(
        "--n_samples", type=int, default=5000,
        help="Max unique questions per dataset (default: 5000).",
    )
    p.add_argument(
        "--k_docs", type=int, default=3,
        help="Retrieval cut-off K (default: 3).",
    )
    p.add_argument(
        "--output_dir", default="results/retrieval",
        help="Root output directory (default: results/retrieval).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ID helpers  — work for both datasets
# ─────────────────────────────────────────────────────────────────────────────

def _base_qid(doc_id: str) -> str:
    """
    Strip a trailing _<digits> suffix to obtain the question ID.

    HotpotQA:  '5a7a06935542990198eaf050_2'  →  '5a7a06935542990198eaf050'
    TriviaQA:  'trivia_q0_1'                  →  'trivia_q0'
               'trivia_q0'                    →  'trivia_q0'  (no strip)
    """
    return re.sub(r"_\d+$", "", doc_id)


def _is_relevant(doc_id: str) -> bool:
    """Relevant documents do NOT contain '_irr_' in their ID."""
    return "_irr_" not in doc_id


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(answers_file: str, n_samples: int) -> List[Dict]:
    """
    Load up to n_samples unique questions from an answers JSONL file.
    Deduplicates on question text (handles HotpotQA's multi-row per question).
    """
    seen: set = set()
    samples: List[Dict] = []
    with open(answers_file, encoding="utf-8") as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            rec = json.loads(line.strip())
            q = rec.get("question", "")
            if q and q not in seen:
                seen.add(q)
                rec["qid"] = _base_qid(rec["id"])
                samples.append(rec)
    return samples


def build_relevance_map(docs_file: str) -> Dict[str, List[str]]:
    """
    Returns {base_qid: [relevant_doc_id, ...]} from the docs JSONL file.
    Distractor documents (containing '_irr_') are excluded.
    """
    qid_to_rel: Dict[str, List[str]] = defaultdict(list)
    with open(docs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if _is_relevant(d["id"]):
                qid_to_rel[_base_qid(d["id"])].append(d["id"])
    return dict(qid_to_rel)


# ─────────────────────────────────────────────────────────────────────────────
# IR metrics
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def compute_metrics(
    retrieved: List[Dict],
    gold_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """
    Computes Recall@K, MRR, MAP, NDCG@K for a single query.

    retrieved  — ranked list from the retriever: [{doc:{id,...},score:float},...]
    gold_ids   — list of relevant document IDs
    k          — retrieval cut-off
    """
    gold_set = set(gold_ids)
    n_rel = len(gold_set)

    if n_rel == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "map": 0.0, "ndcg_at_k": 0.0}

    ret_ids = [r["doc"]["id"] for r in retrieved]
    top_k   = ret_ids[:k]

    # Recall@K
    recall = len(set(top_k) & gold_set) / n_rel

    # MRR
    mrr = 0.0
    for rank, did in enumerate(ret_ids, 1):
        if did in gold_set:
            mrr = 1.0 / rank
            break

    # MAP
    hits, prec_sum = 0, 0.0
    for rank, did in enumerate(ret_ids, 1):
        if did in gold_set:
            hits += 1
            prec_sum += hits / rank
    ap = prec_sum / n_rel

    # NDCG@K
    rel_labels = [1 if d in gold_set else 0 for d in top_k]
    dcg  = _dcg(rel_labels)
    idcg = _dcg(sorted(rel_labels, reverse=True))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall_at_k": recall, "mrr": mrr, "map": ap, "ndcg_at_k": ndcg}


def recall_at_k_curve(
    ret_ids: List[str],
    gold_ids: List[str],
    max_k: int,
) -> Dict[int, float]:
    gold_set = set(gold_ids)
    n_rel = max(len(gold_set), 1)
    return {k: len(set(ret_ids[:k]) & gold_set) / n_rel for k in range(1, max_k + 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset switching (patches app.config.settings + invalidates caches)
# ─────────────────────────────────────────────────────────────────────────────

def switch_dataset(dataset: str) -> None:
    """
    Mutate app.config.settings to point at the chosen dataset's index files,
    then invalidate all in-memory retriever caches so the next retrieve() call
    loads the correct index.
    """
    cfg = DATASET_CFG[dataset]
    from app.config import settings
    settings.DOCUMENTS_PATH  = cfg["docs_file"]
    settings.BM25_INDEX_PATH = cfg["bm25_path"]
    settings.FAISS_INDEX_PATH= cfg["faiss_path"]

    from app.retrievers import bm25 as bm25_mod
    from app.retrievers import dense_faiss as dense_mod
    bm25_mod.invalidate_cache()
    dense_mod.invalidate_cache()
    print(f"    [switch] Dataset → {dataset}  "
          f"(docs: {cfg['docs_file']})")


def make_retrieve_fn(name: str):
    from app.retrievers.bm25       import retrieve as bm25_ret
    from app.retrievers.dense_faiss import retrieve as dense_ret
    from app.retrievers.hybrid      import retrieve as hybrid_ret
    return {"bm25": bm25_ret, "dense": dense_ret, "hybrid": hybrid_ret}[name]


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_retriever(
    name: str,
    retrieve_fn,
    samples: List[Dict],
    qid_to_rel: Dict[str, List[str]],
    k: int,
    max_k_curve: int = 10,
) -> Tuple[List[Dict], Dict[int, List[float]]]:
    rows: List[Dict] = []
    curve: Dict[int, List[float]] = defaultdict(list)
    total = len(samples)

    fetch_k = max(k, max_k_curve)   # single retrieve call covers both k and curve

    for idx, sample in enumerate(samples):
        q   = sample["question"]
        qid = sample["qid"]
        gold = qid_to_rel.get(qid, [])

        try:
            full_ret = retrieve_fn(q, fetch_k)
        except Exception as exc:
            print(f"    [WARN] retriever={name} idx={idx} → {exc}")
            full_ret = []

        retrieved = full_ret[:k]          # top-k for main metrics
        full_ids  = [r["doc"]["id"] for r in full_ret]

        metrics = compute_metrics(retrieved, gold, k)
        crv     = recall_at_k_curve(full_ids, gold, max_k_curve)
        for kv, rv in crv.items():
            curve[kv].append(rv)

        rows.append({
            "question_id":          sample["id"],
            "question":             q,
            "retriever":            name,
            "k_docs":               k,
            "n_gold_relevant_docs": len(gold),
            "gold_relevant_doc_ids":"|".join(gold),
            "retrieved_doc_ids":    "|".join(r["doc"]["id"] for r in retrieved),
            "retrieval_scores":     "|".join(f"{r['score']:.4f}" for r in retrieved),
            "n_retrieved_relevant": len(set(r["doc"]["id"] for r in retrieved) & set(gold)),
            "recall_at_k":          round(metrics["recall_at_k"], 4),
            "mrr":                  round(metrics["mrr"],         4),
            "map":                  round(metrics["map"],         4),
            "ndcg_at_k":            round(metrics["ndcg_at_k"],   4),
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            mean_ndcg = sum(r["ndcg_at_k"] for r in rows) / len(rows)
            print(f"      [{idx+1:>5}/{total}]  NDCG@{k}={mean_ndcg:.4f}")

    return rows, dict(curve)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → Saved {path}  ({len(rows)} rows)")


def build_summary(rows: List[Dict]) -> List[Dict]:
    by_ret: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_ret[r["retriever"]].append(r)
    summary = []
    for name, rrows in sorted(by_ret.items()):
        n = len(rrows)
        summary.append({
            "retriever":        name,
            "n_questions":      n,
            "k_docs":           rrows[0]["k_docs"],
            "mean_recall_at_k": round(sum(r["recall_at_k"] for r in rrows) / n, 4),
            "mean_mrr":         round(sum(r["mrr"]         for r in rrows) / n, 4),
            "mean_map":         round(sum(r["map"]         for r in rrows) / n, 4),
            "mean_ndcg_at_k":   round(sum(r["ndcg_at_k"]  for r in rrows) / n, 4),
        })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "mean_recall_at_k": "Recall@K",
    "mean_mrr":         "MRR",
    "mean_map":         "MAP",
    "mean_ndcg_at_k":   "NDCG@K",
}


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → Saved {path}")


def _apply_style(ax, ylabel: str, title: str) -> None:
    """Shared axes styling — clean, light background."""
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#d0d0d0")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)


def plot_bar_comparison(summary: List[Dict], k: int, out_dir: str, dataset: str) -> None:
    """Grouped bar chart: 4 metrics × 3 retrievers."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics    = list(METRIC_LABELS.keys())
    labels     = [METRIC_LABELS[m] for m in metrics]
    retrievers = [r["retriever"] for r in summary]
    n          = len(retrievers)
    x          = np.arange(len(labels))
    width      = 0.22
    offsets    = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(summary):
        vals = [row[m] for m in metrics]
        clr  = RETRIEVER_COLOURS.get(row["retriever"], "#999999")
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=row["retriever"].upper(),
            color=clr, edgecolor="white", linewidth=0.5, alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#444444",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    display = DATASET_CFG[dataset]["display"]
    _apply_style(ax, "Score", f"{display} — BM25 vs Dense vs Hybrid  (k={k})")
    ax.legend(title="Retriever", fontsize=9, framealpha=0.7)
    fig.patch.set_facecolor("#fafafa")
    _save_fig(fig, os.path.join(out_dir, "plots", "bar_comparison.png"))
    plt.close(fig)


def plot_boxplot(all_rows: List[Dict], metric: str, k: int, out_dir: str, dataset: str) -> None:
    """Per-query score distribution across retrievers."""
    import matplotlib.pyplot as plt

    by_ret: Dict[str, List[float]] = defaultdict(list)
    for r in all_rows:
        by_ret[r["retriever"]].append(r[metric])

    retrievers = sorted(by_ret.keys())
    data    = [by_ret[r] for r in retrievers]
    RETRIEVER_COLOURS = {"bm25": "#6aaa96", "dense": "#5c85c4", "hybrid": "#c47c5c"}
    colours = [RETRIEVER_COLOURS.get(r, "#999999") for r in retrievers]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        data, patch_artist=True, notch=False, widths=0.45,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2, color="#888888"),
        capprops=dict(linewidth=1.2, color="#888888"),
        flierprops=dict(marker="o", markersize=3, alpha=0.3,
                        markerfacecolor="#aaaaaa", markeredgewidth=0),
    )
    for patch, clr in zip(bp["boxes"], colours):
        patch.set_facecolor(clr)
        patch.set_alpha(0.80)

    ax.set_xticks(range(1, len(retrievers) + 1))
    ax.set_xticklabels([r.upper() for r in retrievers], fontsize=11)
    mlabel = METRIC_LABELS.get(metric, metric)
    display = DATASET_CFG[dataset]["display"]
    _apply_style(ax, mlabel, f"{display} — Per-query {mlabel} Distribution  (k={k})")
    fig.patch.set_facecolor("#fafafa")

    fname = f"boxplot_{metric.replace('_at_k','').replace('mean_','').replace('_','')}.png"
    _save_fig(fig, os.path.join(out_dir, "plots", fname))
    plt.close(fig)


def plot_recall_curve(
    curve_data: Dict[str, Dict[int, List[float]]],
    out_dir: str,
    dataset: str,
) -> None:
    """Mean Recall@K for k=1..max_k, one line per retriever."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for ret_name, crv in sorted(curve_data.items()):
        ks    = sorted(crv.keys())
        means = [sum(crv[kv]) / len(crv[kv]) for kv in ks]
        ax.plot(
            ks, means,
            marker="o", markersize=5, linewidth=2.0,
            label=ret_name.upper(),
            color=RETRIEVER_COLOURS.get(ret_name, "#999999"),
            alpha=0.88,
        )

    ax.set_xlabel("K  (documents retrieved)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    display = DATASET_CFG[dataset]["display"]
    _apply_style(ax, "Mean Recall@K", f"{display} — Recall@K Curve")
    ax.legend(title="Retriever", fontsize=9, framealpha=0.7)
    fig.patch.set_facecolor("#fafafa")
    _save_fig(fig, os.path.join(out_dir, "plots", "recall_vs_k.png"))
    plt.close(fig)


def plot_cross_dataset_comparison(
    all_summaries: Dict[str, List[Dict]],
    k: int,
    out_dir: str,
) -> None:
    """
    Grouped bar chart comparing each retriever across datasets.
    One subplot per metric, bars grouped by retriever with dataset as hue.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    datasets   = list(all_summaries.keys())
    metrics    = list(METRIC_LABELS.keys())
    retrievers = sorted({r["retriever"] for rows in all_summaries.values() for r in rows})

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        x      = np.arange(len(retrievers))
        nd     = len(datasets)
        width  = 0.35
        offsets = np.linspace(-(nd - 1) * width / 2, (nd - 1) * width / 2, nd)

        for di, ds in enumerate(datasets):
            vals_by_ret = {r["retriever"]: r[metric] for r in all_summaries[ds]}
            vals = [vals_by_ret.get(ret, 0.0) for ret in retrievers]
            clr  = DATASET_COLOURS.get(ds, "#999999")
            bars = ax.bar(
                x + offsets[di], vals, width,
                label=DATASET_CFG[ds]["display"],
                color=clr, edgecolor="white", linewidth=0.5, alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#444444",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([r.upper() for r in retrievers], fontsize=10)
        ax.set_ylim(0, 1.12)
        _apply_style(ax, "Score", METRIC_LABELS[metric])
        if di == len(datasets) - 1:
            ax.legend(title="Dataset", fontsize=8, framealpha=0.7)

    fig.suptitle(
        f"Cross-Dataset Comparison — HotpotQA vs TriviaQA  (k={k})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.patch.set_facecolor("#fafafa")
    _save_fig(fig, os.path.join(out_dir, "plots", "cross_dataset_comparison.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(summary: List[Dict], dataset: str) -> None:
    if not summary:
        return
    COLS = ["retriever", "n_questions", "k_docs",
            "mean_recall_at_k", "mean_mrr", "mean_map", "mean_ndcg_at_k"]
    cw = {c: max(len(c), max(len(str(r[c])) for r in summary)) for c in COLS}
    sep = "  ".join("-" * cw[c] for c in COLS)
    print(f"\n{'═'*72}")
    print(f"  {DATASET_CFG[dataset]['display']} — Retrieval Effectiveness")
    print(f"{'═'*72}")
    print("  " + "  ".join(c.ljust(cw[c]) for c in COLS))
    print(f"  {sep}")
    for row in summary:
        print("  " + "  ".join(str(row[c]).ljust(cw[c]) for c in COLS))
    print(f"{'═'*72}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{'='*65}")
    print("  Multi-Dataset Retrieval Evaluation")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Retrievers : {args.retrievers}")
    print(f"  N samples  : {args.n_samples}")
    print(f"  K docs     : {args.k_docs}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*65}\n")

    all_summaries: Dict[str, List[Dict]] = {}

    for dataset in args.datasets:
        ds_display = DATASET_CFG[dataset]["display"]
        ds_out     = os.path.join(args.output_dir, dataset)
        cfg        = DATASET_CFG[dataset]

        print(f"\n{'─'*65}")
        print(f"  Dataset: {ds_display}")
        print(f"{'─'*65}")

        # ── Step 1: Switch dataset (patch settings, invalidate caches) ────────
        print("[1/3] Switching dataset index…")
        switch_dataset(dataset)

        # ── Step 2: Load data ─────────────────────────────────────────────────
        print("[2/3] Loading data…")
        samples    = load_samples(cfg["answers_file"], args.n_samples)
        qid_to_rel = build_relevance_map(cfg["docs_file"])
        print(f"      {len(samples)} unique questions loaded.")
        avg_rel = (sum(len(v) for v in qid_to_rel.values())
                   / max(len(qid_to_rel), 1))
        print(f"      Avg relevant docs per question: {avg_rel:.2f}")

        # ── Step 3: Evaluate each retriever ───────────────────────────────────
        print("[3/3] Running retrieval…")
        all_rows:   List[Dict]                        = []
        curve_data: Dict[str, Dict[int, List[float]]] = {}

        for ret_name in args.retrievers:
            print(f"\n  ▶  {ret_name.upper()}")
            try:
                retrieve_fn = make_retrieve_fn(ret_name)
            except Exception as exc:
                print(f"    [ERROR] Could not load retriever '{ret_name}': {exc}")
                continue

            rows, crv = evaluate_retriever(
                ret_name, retrieve_fn, samples, qid_to_rel, args.k_docs
            )
            all_rows.extend(rows)
            curve_data[ret_name] = crv

        # ── Save CSVs ─────────────────────────────────────────────────────────
        print()
        save_csv(os.path.join(ds_out, "per_query_results.csv"), all_rows)
        summary = build_summary(all_rows)
        save_csv(os.path.join(ds_out, "summary_metrics.csv"), summary)
        all_summaries[dataset] = summary

        # ── Plots ─────────────────────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "font.family":  "DejaVu Sans",
                "axes.facecolor": "#f8f8f8",
                "figure.facecolor": "#fafafa",
            })
            plot_bar_comparison(summary, args.k_docs, ds_out, dataset)
            plot_boxplot(all_rows, "ndcg_at_k", args.k_docs, ds_out, dataset)
            plot_boxplot(all_rows, "mrr",        args.k_docs, ds_out, dataset)
            if curve_data:
                plot_recall_curve(curve_data, ds_out, dataset)
        except ImportError:
            print("  [WARN] matplotlib not available — skipping plots.")
        except Exception as exc:
            print(f"  [WARN] Plotting failed: {exc}")

        print_summary_table(summary, dataset)

    # ── Cross-dataset comparison (only when both datasets evaluated) ──────────
    if len(all_summaries) >= 2:
        print("\n[Cross-dataset] Generating comparison plot…")
        try:
            import matplotlib
            matplotlib.use("Agg")
            plot_cross_dataset_comparison(all_summaries, args.k_docs, args.output_dir)
        except Exception as exc:
            print(f"  [WARN] Cross-dataset plot failed: {exc}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
