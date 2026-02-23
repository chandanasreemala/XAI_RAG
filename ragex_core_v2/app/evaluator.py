from typing import List

def f1_score_sets(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    pred = set(pred_tokens)
    gold = set(gold_tokens)
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 2 * (prec * rec) / (prec + rec)

def mrr(ranked_list: List[str], gold: str) -> float:
    for idx, item in enumerate(ranked_list):
        if gold in item:
            return 1.0 / (idx + 1)
    return 0.0

def response_match(pred: str, gold: str) -> bool:
    return gold.lower().strip() in pred.lower()
