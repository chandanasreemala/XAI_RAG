import json
from datasets import load_dataset

DATASET_NAME = "PatronusAI/HaluBench"
SPLIT = "test"

DOCS_OUT = "docs.jsonl"
ANSWERS_OUT = "answers.jsonl"

def main():
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # Filter first: keep only PASS
    ds_pass = ds.filter(lambda x: x["label"] == "PASS")

    with open(DOCS_OUT, "w", encoding="utf-8") as f_docs, \
         open(ANSWERS_OUT, "w", encoding="utf-8") as f_answers:

        for row in ds_pass:
            _id = row["id"]
            passage = row["passage"]
            answer = row["answer"]

            doc_obj = {
                "id": _id,
                "text": passage,
                "meta": {"source": "halubench"},
            }
            ans_obj = {
                "id": _id,
                "answer": answer,
                "meta": {"source": "halubench"},
            }

            f_docs.write(json.dumps(doc_obj, ensure_ascii=False) + "\n")
            f_answers.write(json.dumps(ans_obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds_pass)} PASS rows to {DOCS_OUT} and {ANSWERS_OUT}")

if __name__ == "__main__":
    main()
