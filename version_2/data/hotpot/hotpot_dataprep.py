import csv
import json
import ast
import re
import os

INPUT_FILE = "hotpotqa_fulldataset_cleaned.csv"
DOCS_OUTPUT = "docs.jsonl"
ANSWERS_OUTPUT = "answers.jsonl"

def clean_text(text):
    """Clean text by stripping leading/trailing whitespace and quotes."""
    text = text.strip().strip('"').strip("'").strip()
    return text

def parse_passage(passage_str):
    """Safely parse the passage column which is a list represented as a string."""
    try:
        items = ast.literal_eval(passage_str)
        return [clean_text(item) for item in items]
    except Exception:
        # Fallback: manual split if ast fails
        passage_str = passage_str.strip().strip("[]")
        items = re.split(r",\s*(?=['\"])", passage_str)
        return [clean_text(item) for item in items]

def parse_groundtruth(gt_str):
    """Safely parse the groundtruth_docs column."""
    try:
        items = ast.literal_eval(gt_str)
        return [clean_text(item) for item in items]
    except Exception:
        gt_str = gt_str.strip().strip("[]")
        items = re.split(r",\s*(?=['\"])", gt_str)
        return [clean_text(item) for item in items]

def normalize(text):
    """Normalize text for comparison: lowercase and strip extra whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(DOCS_OUTPUT, "w", encoding="utf-8") as f_docs, \
         open(ANSWERS_OUTPUT, "w", encoding="utf-8") as f_ans:

        reader = csv.DictReader(f_in)

        for row in reader:
            row_id = row["id"].strip()
            question = row["question"].strip()
            answer = row["answer"].strip()
            passage_items = parse_passage(row["passage"])
            gt_items = parse_groundtruth(row["groundtruth_docs"])

            # Normalize groundtruth for comparison
            gt_normalized = [normalize(g) for g in gt_items]

            rel_counter = 1
            irr_counter = 1

            for item in passage_items:
                item_normalized = normalize(item)

                if item_normalized in gt_normalized:
                    doc_id = f"{row_id}_{rel_counter}"
                    rel_counter += 1

                    # Write to docs.jsonl
                    doc_entry = {
                        "id": doc_id,
                        "text": item,
                        "meta": {"source": "hotpotqa"}
                    }
                    f_docs.write(json.dumps(doc_entry) + "\n")

                    # Write to answers.jsonl with question
                    ans_entry = {
                        "id": doc_id,
                        "question": question,
                        "answer": answer
                    }
                    f_ans.write(json.dumps(ans_entry) + "\n")

                else:
                    doc_id = f"{row_id}_irr_{irr_counter}"
                    irr_counter += 1

                    # Write to docs.jsonl only
                    doc_entry = {
                        "id": doc_id,
                        "text": item,
                        "meta": {"source": "hotpotqa"}
                    }
                    f_docs.write(json.dumps(doc_entry) + "\n")

    print(f"Done! Files saved:\n  {DOCS_OUTPUT}\n  {ANSWERS_OUTPUT}")

if __name__ == "__main__":
    main()