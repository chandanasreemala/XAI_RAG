"""
TriviaQA Data Preparation
=========================
Downloads TriviaQA (rc.wikipedia split) from HuggingFace and converts it
into the standard format used by this project:

  trivia_docs.jsonl    — all documents (relevant only; no distractors)
  trivia_answers.jsonl — question/answer records linked to each relevant doc

ID convention (mirrors HotpotQA prep):
  Relevant doc  →  {question_id}_{n}   (e.g.  tc_0_1, tc_0_2)

  There are NO irrelevant/_irr_ docs because TriviaQA does not provide
  explicit distractor passages. The evaluate_retrieval.py script detects
  relevance by the absence of '_irr_' in the doc ID, so every doc written
  here is treated as relevant.

docs.jsonl schema  (one JSON object per line):
  {
    "id":   "<question_id>_<n>",
    "text": "<full Wikipedia article text>",
    "meta": {
      "source": "triviaqa",
      "title":  "<Wikipedia article title>",
      "split":  "<train|validation|test>"
    }
  }

answers.jsonl schema  (one JSON object per line, one per relevant doc):
  {
    "id":       "<question_id>_<n>",   # same as the doc id
    "question": "<question text>",
    "answer":   "<canonical answer string>"
  }

Usage:
  python triviaqa_dataprep.py                        # all train examples
  python triviaqa_dataprep.py --split validation
  python triviaqa_dataprep.py --split train --n_samples 5000

Output files are written in the same directory as this script.
"""

import argparse
import json
import os
import re
import sys


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_id(raw_id: str) -> str:
    """Replace characters that are unsafe in IDs with underscores."""
    return re.sub(r"[^A-Za-z0-9\-]", "_", raw_id)


def clean_text(text: str) -> str:
    """Collapse excessive whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare TriviaQA (rc.wikipedia) for RAG evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="HuggingFace dataset split to use (default: train).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Max number of questions to process (default: all).",
    )
    p.add_argument(
        "--docs_output",
        default=None,
        help="Output path for docs JSONL (default: trivia_docs.jsonl next to this script).",
    )
    p.add_argument(
        "--answers_output",
        default=None,
        help="Output path for answers JSONL (default: trivia_answers.jsonl next to this script).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve output paths ──────────────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    docs_path    = args.docs_output    or os.path.join(_here, "trivia_docs.jsonl")
    answers_path = args.answers_output or os.path.join(_here, "trivia_answers.jsonl")

    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package not found. Install it with:\n"
              "  pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading TriviaQA rc.wikipedia / split='{args.split}' from HuggingFace …")
    dataset = load_dataset("trivia_qa", "rc.wikipedia", split=args.split,
                           trust_remote_code=True)
    print(f"  {len(dataset):,} examples loaded.")

    # ── Process ───────────────────────────────────────────────────────────────
    n_questions   = 0
    n_docs        = 0
    n_skipped     = 0                # questions with no evidence passages

    with open(docs_path,    "w", encoding="utf-8") as f_docs, \
         open(answers_path, "w", encoding="utf-8") as f_ans:

        for example in dataset:

            if args.n_samples is not None and n_questions >= args.n_samples:
                break

            raw_qid     = example["question_id"]
            question    = clean_text(example["question"])
            answer_val  = example["answer"]["value"]  # canonical answer string

            # entity_pages holds the Wikipedia evidence for the rc.wikipedia split
            wiki_contexts = example.get("entity_pages", {}).get("wiki_context", [])
            wiki_titles   = example.get("entity_pages", {}).get("title", [])

            # Skip questions that have no evidence passages at all
            if not wiki_contexts:
                n_skipped += 1
                continue

            qid = sanitize_id(raw_qid)

            # ── One relevant doc per Wikipedia article ────────────────────────
            # ID: {qid}_{n}  (n is 1-based)
            # The absence of '_irr_' marks these as relevant to the evaluator.
            for n, (ctx, title) in enumerate(
                zip(wiki_contexts, wiki_titles + [""] * len(wiki_contexts)), start=1
            ):
                ctx = clean_text(ctx)
                if not ctx:
                    continue                   # skip empty passages

                doc_id = f"{qid}_{n}"

                # docs.jsonl
                doc_entry = {
                    "id":   doc_id,
                    "text": ctx,
                    "meta": {
                        "source": "triviaqa",
                        "title":  title.strip() if title else "",
                        "split":  args.split,
                    },
                }
                f_docs.write(json.dumps(doc_entry, ensure_ascii=False) + "\n")

                # answers.jsonl
                ans_entry = {
                    "id":       doc_id,
                    "question": question,
                    "answer":   answer_val,
                }
                f_ans.write(json.dumps(ans_entry, ensure_ascii=False) + "\n")

                n_docs += 1

            n_questions += 1

            if n_questions % 1000 == 0:
                print(f"  Processed {n_questions:,} questions  |  {n_docs:,} docs written …")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone!")
    print(f"  Questions processed : {n_questions:,}")
    print(f"  Questions skipped   : {n_skipped:,}  (no evidence passages)")
    print(f"  Total docs written  : {n_docs:,}")
    print(f"  Avg docs/question   : {n_docs / max(n_questions, 1):.2f}")
    print(f"\n  {docs_path}")
    print(f"  {answers_path}")


if __name__ == "__main__":
    main()
