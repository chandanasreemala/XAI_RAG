"""
Build a clean TriviaQA (Wikipedia RC split) dataset for RAG-Ex.

Data is sourced entirely from HuggingFace (trivia_qa / rc.wikipedia).
This script does NOT read from any existing trivia_docs.jsonl or
trivia_answers.jsonl files — it always builds fresh from the HF dataset.

Outputs (written to the same directory as this script by default):
  trivia_docs.jsonl    — one doc per Wikipedia passage, unique by content
  trivia_answers.jsonl — one record per question with the canonical answer

IDs follow the convention expected by the gold-cache in api.py:
  question  → trivia_q{qidx}          (no _N suffix → base_id = full id)
  doc       → trivia_q{qidx}_{didx}   (has _N suffix → base  = trivia_q{qidx})

Usage:
    python data/trivia/build_trivia_dataset.py
    python data/trivia/build_trivia_dataset.py --max-questions 5000
    python data/trivia/build_trivia_dataset.py --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TriviaQA Wikipedia RC dataset from HuggingFace")
    p.add_argument(
        "--out-dir",
        # Default: same directory as this script (data/trivia/)
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write trivia_docs.jsonl and trivia_answers.jsonl (default: script directory)",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=5000,
        help="Maximum number of questions to include (default: 5000)",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="HuggingFace dataset split to use (default: train)",
    )
    p.add_argument(
        "--min-passage-chars",
        type=int,
        default=100,
        help="Minimum characters a passage must have to be included (default: 100)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_key(text: str) -> str:
    """Normalised SHA-256 of text — used to detect duplicate passages globally."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def _canonical_answer(answer_field: Dict[str, Any]) -> str:
    """Extract the canonical answer string from a TriviaQA answer dict."""
    value = answer_field.get("value") or ""
    if value:
        return str(value)
    aliases = answer_field.get("aliases") or []
    if aliases:
        return str(aliases[0])
    return str(answer_field.get("normalized_value") or "")


def _passages_for_item(item: Dict[str, Any], min_chars: int) -> List[Dict[str, str]]:
    """
    Extract Wikipedia passages from a TriviaQA rc.wikipedia item.

    entity_pages is a dict with parallel lists:
        title        : list[str]
        wiki_context : list[str]   ← the actual passage text

    Returns a list of {title, text} dicts with text >= min_chars.
    """
    ep = item.get("entity_pages") or {}
    titles: List[str]   = ep.get("title") or []
    contexts: List[str] = ep.get("wiki_context") or []

    passages: List[Dict[str, str]] = []
    for title, ctx in zip(titles, contexts):
        ctx = (ctx or "").strip()
        if len(ctx) >= min_chars:
            passages.append({"title": str(title or ""), "text": ctx})

    return passages


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(out_dir: str, max_questions: int, split: str, min_passage_chars: int) -> None:
    # Data is sourced exclusively from HuggingFace — existing JSONL files
    # in out_dir are completely ignored (overwritten on success).
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("  Run: pip install datasets")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    docs_path    = os.path.join(out_dir, "trivia_docs.jsonl")
    answers_path = os.path.join(out_dir, "trivia_answers.jsonl")

    print(f"Sourcing dataset from HuggingFace: trivia_qa / rc.wikipedia ({split} split) …")
    print("  (existing trivia_docs.jsonl / trivia_answers.jsonl are NOT used as input)")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split=split, trust_remote_code=True)
    print(f"  Loaded {len(ds):,} items from HuggingFace rc.wikipedia/{split}")

    seen_text_keys: Set[str]  = set()   # global content-hash dedup (same passage → skip)
    seen_questions: Set[str]  = set()   # dedup identical question texts

    docs_written    = 0
    answers_written = 0
    questions_done  = 0

    with open(docs_path, "w", encoding="utf-8") as docs_fh, \
         open(answers_path, "w", encoding="utf-8") as ans_fh:
        for item in ds:
            if questions_done >= max_questions:
                break

            question = (item.get("question") or "").strip()
            if not question:
                continue

            # Deduplicate identical question texts
            q_key = question.lower()
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)

            answer = _canonical_answer(item.get("answer") or {})
            if not answer:
                continue

            passages = _passages_for_item(item, min_passage_chars)
            if not passages:
                # Skip questions with no usable Wikipedia passage
                continue

            qidx = questions_done       # 0-indexed
            q_id = f"trivia_q{qidx}"   # e.g. trivia_q0

            # --- Write answer record ---
            ans_record = {
                "id":       q_id,
                "question": question,
                "answer":   answer,
            }
            ans_fh.write(json.dumps(ans_record, ensure_ascii=False) + "\n")
            answers_written += 1

            # --- Write doc records (deduplicated globally by content hash) ---
            didx = 0
            for psg in passages:
                tkey = _text_key(psg["text"])
                if tkey in seen_text_keys:
                    continue
                seen_text_keys.add(tkey)

                doc_id = f"{q_id}_{didx}"   # e.g. trivia_q0_0, trivia_q0_1 …
                doc_record = {
                    "id":   doc_id,
                    "text": psg["text"],
                    "meta": {
                        "source":      "triviaqa_hf",
                        "title":       psg["title"],
                        "split":       split,
                        "question_id": q_id,
                    },
                }
                docs_fh.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
                docs_written += 1
                didx += 1

            questions_done += 1

            if questions_done % 500 == 0:
                print(f"  {questions_done:,} questions processed, "
                      f"{docs_written:,} docs, {answers_written:,} answers …")

    print(f"\nDone. Data written to: {out_dir}")
    print(f"  Questions : {answers_written:,}")
    print(f"  Docs      : {docs_written:,}")
    print(f"  Answers   → {answers_path}")
    print(f"  Docs      → {docs_path}")
    print(f"\nNext step — build FAISS + BM25 indices:")
    print(f"  cd <version_2 root> && python -m scripts.build_index {docs_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    build(
        out_dir=os.path.abspath(args.out_dir),
        max_questions=args.max_questions,
        split=args.split,
        min_passage_chars=args.min_passage_chars,
    )
