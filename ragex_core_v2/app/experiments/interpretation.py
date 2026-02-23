# app/experiments/interpretation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import random
import numpy as np

from app.generator import generate_answer
from app.perturb import perturb_sentence, split_context
from app.comparator import COMPARATORS
import math
from typing import Callable

def _minmax_scale(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = float(min(scores)), float(max(scores))
    if mx - mn < 1e-12:
        return [1.0 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]

def _split_docs(context: str) -> List[str]:
    # In your pipeline, docs are joined with '\n'
    parts = [p.strip() for p in (context or "").split("\n") if p.strip()]
    return parts if parts else [(context or "").strip()]

def _unit_doc_index(unit: str, doc_texts: List[str]) -> int:
    # Best-effort mapping: find which doc chunk contains the unit text
    u = (unit or "").strip()
    for i, d in enumerate(doc_texts):
        if u and u in d:
            return i
    return 0

def _safe_confidence(c: float | None) -> float:
    # If confidence is unavailable, treat as 0 contribution (keeps backward compat)
    return float(c) if c is not None else 0.0


def generate_answer_with_confidence(
    generator: Any,
    prompt: str,
    *,
    max_length: int,
    temperature: float,
) -> Tuple[str, Optional[float]]:
    """
    Backward-compatible wrapper.
    If your generator supports confidence, return it; else return None.
    """
    # Option A: if your generate_answer already supports return_confidence
    try:
        out = generate_answer(generator, prompt, max_length=max_length, temperature=temperature, return_confidence=True)
        # expected: (text, confidence)
        if isinstance(out, tuple) and len(out) == 2:
            return str(out[0]), float(out[1]) if out[1] is not None else None
        return str(out), None
    except TypeError:
        # generate_answer doesn't accept return_confidence
        txt = generate_answer(generator, prompt, max_length=max_length, temperature=temperature)
        return str(txt), None

# ----------------------------
# Bucketing (copied to avoid importing app.api)
# ----------------------------
def bucket_importances(
    importances: Dict[str, float],
    method: str = "quantile",
    q_hi: float = 0.80,
    q_mid: float = 0.50,
    hi_thr: float = 0.66,
    mid_thr: float = 0.33,
) -> Dict[str, List[str]]:
    if not importances:
        return {"most": [], "important": [], "least": []}

    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    values = [v for _, v in items]

    def quantile(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        q = max(0.0, min(1.0, float(q)))
        idx = int(round((len(vals) - 1) * q))
        idx = max(0, min(len(vals) - 1, idx))
        return float(sorted(vals)[idx])

    if method == "quantile":
        hi_cut = quantile(values, q_hi)
        mid_cut = quantile(values, q_mid)
    elif method == "threshold":
        hi_cut = float(hi_thr)
        mid_cut = float(mid_thr)
    else:
        raise ValueError("bucket_importances method must be 'quantile' or 'threshold'")

    most: List[str] = []
    important: List[str] = []
    least: List[str] = []

    for k, v in items:
        if v >= hi_cut:
            most.append(k)
        elif v >= mid_cut:
            important.append(k)
        else:
            least.append(k)

    return {"most": most, "important": important, "least": least}


# ----------------------------
# Metrics helpers
# ----------------------------
def exact_match(a: str, b: str) -> float:
    return 1.0 if (a or "").strip() == (b or "").strip() else 0.0


def token_overlap(a: str, b: str) -> float:
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / max(1, len(ta | tb)))


def mean_ci(xs: List[float], z: float = 1.96) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    arr = np.array(xs, dtype=float)
    m = float(arr.mean())
    if len(arr) == 1:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return {"mean": m, "lo": m - z * se, "hi": m + z * se, "n": int(len(arr))}


# ----------------------------
# Explainer: compute normalized importances + buckets
# ----------------------------
def compute_importances_and_buckets(
    *,
    generator: Any,
    question: str,
    context: str,
    explanation_level: str = "sentence",
    perturber: str = "leave_one_out",
    comparator: str = "semantic",
    max_length: int = 256,
    temperature: float = 0.0,
    debug: bool = False,
    debug_max_perturbations: int = 5,

    # NEW (additive)
    importance_mode: str = "ragex_core",   # "ragex_core" | "modified_ragex"
    alpha: float = 0.5,                   # balance response diff vs confidence drop
    use_relevance_weight: bool = True,
    doc_scores: Optional[List[float]] = None,  # retrieval relevance scores per doc chunk
) -> Dict[str, Any]:

    # Baseline
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    y, conf0 = generate_answer_with_confidence(
        generator,
        prompt,
        max_length=max_length,
        temperature=temperature,
    )
    conf0 = _safe_confidence(conf0)
    units = split_context(context, explanation_level)
    if not units:
        units = [context]
    doc_texts = _split_docs(context)
    doc_scores_local = doc_scores if (doc_scores and len(doc_scores) == len(doc_texts)) else [1.0] * len(doc_texts)
    doc_scores_norm = _minmax_scale([float(s) for s in doc_scores_local])

    # comparator function used to score response similarity
    comparator_fn = COMPARATORS.get(comparator) or COMPARATORS["semantic"]

    raw_dissimilarities: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    for i, unit in enumerate(units):
        perturbs = perturb_sentence(unit, perturber)
        if not perturbs:
            perturbs = [""]

        sims: List[float] = []
        samples: List[Dict[str, Any]] = []

        for p in perturbs:
            new_units = units.copy()
            new_units[i] = p
            new_ctx = " ".join([u for u in new_units if u is not None])

            new_prompt = f"Context: {new_ctx}\nQuestion: {question}\nAnswer:"
            y_p, conf_p = generate_answer_with_confidence(
                generator,
                new_prompt,
                max_length=max_length,
                temperature=temperature,
            )
            sim = float(comparator_fn(y_p, y))
            sims.append(sim)

            # NEW: collect confidences (if available)
            if "confs" not in locals():
                confs = []
            confs.append(conf_p)

            if debug:
                samples.append({
                    "perturbed_unit": p,
                    "answer": y_p,
                    "similarity_to_original": sim,
                    "confidence": conf_p,
                })

        mean_sim = float(sum(sims) / max(1, len(sims)))
        resp_diff = float(1.0 - mean_sim)  # classic RAG-Ex signal

        # Confidence drop ΔC_i: average confidence under perturbations
        conf_ps = [_safe_confidence(c) for c in confs] if "confs" in locals() else []
        mean_conf_p = float(sum(conf_ps) / max(1, len(conf_ps))) if conf_ps else conf0
        delta_c = max(0.0, conf0 - mean_conf_p)  # drop in confidence when unit is perturbed

        # doc relevance s_d
        didx = _unit_doc_index(unit, doc_texts)
        s_d = float(doc_scores_norm[didx]) if (0 <= didx < len(doc_scores_norm)) else 1.0

        if importance_mode == "ragex_core":
            raw = resp_diff
        elif importance_mode == "modified_ragex":
            # unified raw score: combine response diff & confidence drop, then weight by relevance
            mix = float(alpha) * resp_diff + (1.0 - float(alpha)) * float(delta_c)
            raw = mix
        else:
            raise ValueError("importance_mode must be 'ragex_core' or 'modified_ragex'")

        if use_relevance_weight:
            raw = raw * s_d

        raw_dissimilarities[unit] = float(raw)
        if debug:
            details[unit] = {
                "unit_index": i,
                "mean_similarity": mean_sim,
                "raw_dissimilarity": raw,
                "num_perturbations": len(perturbs),
                "samples": samples,
            }

    maxv = max(raw_dissimilarities.values()) if raw_dissimilarities else 1.0
    if maxv <= 0:
        normalized = {k: 0.0 for k in raw_dissimilarities}
    else:
        normalized = {k: float(v / maxv) for k, v in raw_dissimilarities.items()}

    buckets = bucket_importances(normalized, method="quantile", q_hi=0.80, q_mid=0.50)

    return {
        "prompt": prompt,
        "baseline_answer": y,
        "units": units,
        "normalized_importances": normalized,
        "buckets": buckets,
        "debug_details": details if debug else None,
        "importance_mode": importance_mode,
        "alpha": alpha,
        "use_relevance_weight": use_relevance_weight,
        "doc_scores_norm": doc_scores_norm if debug else None,
        "baseline_confidence": conf0 if debug else None
    }


# ----------------------------
# Deletion arms
# ----------------------------
def _k_to_count(k_spec: str, n_units: int) -> int:
    k_spec = (k_spec or "").strip().lower()
    if k_spec in ("top-1", "1", "k=1"):
        return 1
    if k_spec in ("top-3", "3", "k=3"):
        return min(3, n_units)
    if "20" in k_spec and "%" in k_spec:
        return max(1, int(math.ceil(0.20 * n_units)))
    if k_spec.startswith("k="):
        try:
            return max(1, min(n_units, int(k_spec.split("=", 1)[1])))
        except Exception:
            return 1
    # default
    return 1


def _delete_units(units: List[str], to_remove: List[str]) -> List[str]:
    remove_set = set(to_remove)
    return [u for u in units if u not in remove_set]


def _ranked_units(importances: Dict[str, float]) -> List[str]:
    return [u for u, _ in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)]


def generate_with_context(
    *,
    generator: Any,
    question: str,
    context_units: List[str],
    max_length: int,
    temperature: float,
) -> str:
    ctx = " ".join([u for u in context_units if u is not None]).strip()
    prompt = f"Context: {ctx}\nQuestion: {question}\nAnswer:"
    return generate_answer(generator, prompt, max_length=max_length, temperature=temperature)


def run_interpretation(
    *,
    generator: Any,
    question: str,
    context: str,
    explanation_level: str = "sentence",
    perturber: str = "leave_one_out",
    comparator: str = "semantic",
    max_length: int = 256,
    temperature: float = 0.0,
    # importance computation controls (forwarded to compute_importances_and_buckets)
    importance_mode: str = "ragex_core",
    alpha: float = 0.5,
    use_relevance_weight: bool = True,
    doc_scores: Optional[List[float]] = None,
    # experiment controls
    k_values: Optional[List[str]] = None,
    random_repeats: int = 50,
    semantic_threshold_t: float = 0.85,
    high_sim_for_case_c: float = 0.90,
    approx_eps: float = 0.02,
    much_less_delta: float = 0.05,
    perturb_only_context: bool = True,  # keep question fixed except DEL_Q
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Implements your Case A/B/C protocol with multiple arms and multiple K values.
    Returns summary + (optional) per-arm details.
    """
    if k_values is None:
        k_values = ["top-1", "top-3", "top-20%"]

    # Compute explanation ranking + buckets
    exp = compute_importances_and_buckets(
        generator=generator,
        question=question,
        context=context,
        explanation_level=explanation_level,
        perturber=perturber,
        comparator=comparator,
        max_length=max_length,
        temperature=temperature,
        # forward importance computation controls
        importance_mode=importance_mode,
        alpha=alpha,
        use_relevance_weight=use_relevance_weight,
        doc_scores=doc_scores,
        debug=debug,
    )

    baseline_y = exp["baseline_answer"]
    units: List[str] = exp["units"]
    importances: Dict[str, float] = exp["normalized_importances"]
    buckets: Dict[str, List[str]] = exp["buckets"]

    comparator_fn = COMPARATORS.get(comparator) or COMPARATORS["semantic"]

    # Extra metrics requested
    def score_all(y2: str) -> Dict[str, float]:
        sem = float(comparator_fn(baseline_y, y2))
        return {
            "semantic": sem,
            "exact_match": float(exact_match(baseline_y, y2)),
            "token_overlap": float(token_overlap(baseline_y, y2)),
            # optional: keep comparator-specific name too
            "comparator_used": sem,
        }

    ranked = _ranked_units(importances)
    least_ranked = list(reversed(ranked))

    results_per_k: Dict[str, Any] = {}

    for kspec in k_values:
        k_count = _k_to_count(kspec, len(units))

        # --- arms ---
        # baseline
        y_none = generate_with_context(
            generator=generator, question=question, context_units=units, max_length=max_length, temperature=temperature
        )
        sim_none = float(comparator_fn(baseline_y, y_none))

        # DEL_MOST: remove top-k by importance OR top bucket if requested via kspec
        most_units = ranked[:k_count]
        ctx_most = _delete_units(units, most_units)
        y_most = generate_with_context(
            generator=generator, question=question, context_units=ctx_most, max_length=max_length, temperature=temperature
        )
        sim_most = float(comparator_fn(baseline_y, y_most))

        # DEL_LEAST: remove bottom-k
        least_units = least_ranked[:k_count]
        ctx_least = _delete_units(units, least_units)
        y_least = generate_with_context(
            generator=generator, question=question, context_units=ctx_least, max_length=max_length, temperature=temperature
        )
        sim_least = float(comparator_fn(baseline_y, y_least))

        # DEL_RANDOM: repeat
        rand_sims: List[float] = []
        rand_details: List[Dict[str, Any]] = []
        for r in range(max(1, int(random_repeats))):
            idxs = list(range(len(units)))
            random.shuffle(idxs)
            rm = [units[i] for i in idxs[:k_count]]
            ctx_rand = _delete_units(units, rm)
            y_rand = generate_with_context(
                generator=generator, question=question, context_units=ctx_rand, max_length=max_length, temperature=temperature
            )
            sim_rand = float(comparator_fn(baseline_y, y_rand))
            rand_sims.append(sim_rand)
            if debug and r < 5:
                rand_details.append({"removed": rm, "answer": y_rand, "semantic": sim_rand})

        rand_ci = mean_ci(rand_sims)

        # DEL_Q: remove question (sanity check)
        # Keep formatting constant: blank question string
        y_del_q = generate_with_context(
            generator=generator, question="", context_units=units, max_length=max_length, temperature=temperature
        )
        sim_del_q = float(comparator_fn(baseline_y, y_del_q))

        # --- Case classification (your definitions) ---
        sim_random = float(rand_ci["mean"])

        def approx(a: float, b: float) -> bool:
            return abs(a - b) <= float(approx_eps)

        def much_less(a: float, b: float) -> bool:
            return a <= (b - float(much_less_delta))

        case = "UNSURE"
        # Case A: sim_most ≪ sim_random and sim_least ≈ sim_random
        if much_less(sim_most, sim_random) and approx(sim_least, sim_random):
            case = "A"
        # Case B: sim_most ≈ sim_random and sim_least ≈ sim_random
        elif approx(sim_most, sim_random) and approx(sim_least, sim_random):
            case = "B"
        # Case C: sim_most ≪ sim_random but sim_most still high
        elif much_less(sim_most, sim_random) and sim_most >= float(high_sim_for_case_c):
            case = "C"

        # Response match flags (semantic threshold)
        match_flags = {
            "DEL_NONE": sim_none >= semantic_threshold_t,
            "DEL_MOST": sim_most >= semantic_threshold_t,
            "DEL_LEAST": sim_least >= semantic_threshold_t,
            "DEL_RANDOM_mean": sim_random >= semantic_threshold_t,
            "DEL_Q": sim_del_q >= semantic_threshold_t,
        }

        results_per_k[kspec] = {
            "k_count": k_count,
            "case": case,
            "similarities": {
                "sim_none": sim_none,
                "sim_most": sim_most,
                "sim_least": sim_least,
                "sim_random": rand_ci,
                "sim_del_q": sim_del_q,
            },
            "match_flags": match_flags,
            "answers": {
                "baseline": baseline_y,
                "DEL_NONE": y_none,
                "DEL_MOST": y_most,
                "DEL_LEAST": y_least,
                "DEL_Q": y_del_q,
            } if debug else None,
            "random_debug_samples": rand_details if debug else None,
            "removed_units": {
                "DEL_MOST": most_units,
                "DEL_LEAST": least_units,
            } if debug else None,
        }

    summary = {
        "buckets": buckets,
        "num_units": len(units),
        "k_values": k_values,
        "cases_by_k": {k: results_per_k[k]["case"] for k in results_per_k},
        "notes": {
            "interpretation": {
                "A": "faithful ranking (most deletion hurts, least ≈ random)",
                "B": "explainer failing or model not using context",
                "C": "partial dependence / redundancy; inspect examples",
            }
        },
    }

    return {
        "summary": summary,
        "explainer": {
            "normalized_importances": exp["normalized_importances"],
            "buckets": exp["buckets"],
        },
        "results": results_per_k,
        "debug": {
            "units": units,
            "prompt": exp["prompt"],
            "explainer_debug_details": exp["debug_details"],
        } if debug else None,
    }
