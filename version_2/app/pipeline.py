"""
Shared RAG explanation pipeline used by /explain and /compare.

Keeps retrieval context building, perturbation loops, and importance scoring
in one place so the API layer stays thin.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as torch_F

from app.generator import GeneratorWrapper, generate_answer
from app.perturb import perturb_sentence, split_context
from app.retriever import retrieve

logger = logging.getLogger(__name__)

# Context / perturbation limits (shared with API)
MAX_DOC_CHARS: int = 1400
MAX_UNITS: int = 25

VALID_IMPORTANCE_MODES = frozenset(
    {
        "ragex_core",
        "retrieval_weighted",
        "confidence_retrieval_fusion",
    }
)


def cap_doc_text(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[:cut] if cut > 0 else text[:max_chars]


def build_rag_context(
    question: str,
    *,
    context: str = "",
    top_k_docs: int = 3,
    retriever: Optional[str] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Return (context_used, retrieved_docs).
    Uses provided context when non-empty; otherwise retrieves from the index.
    """
    context_used = (context or "").strip()
    if context_used:
        return context_used, None

    try:
        try:
            retrieved_docs = retrieve(question, top_k_docs, retriever=retriever)  # type: ignore[arg-type]
        except TypeError:
            retrieved_docs = retrieve(question, top_k_docs)  # type: ignore[misc]
    except FileNotFoundError:
        raise

    context_used = "\n".join(
        cap_doc_text(item["doc"]["text"])
        for item in (retrieved_docs or [])
        if isinstance(item, dict)
        and "doc" in item
        and isinstance(item["doc"], dict)
        and "text" in item["doc"]
    ).strip()

    if not context_used:
        raise ValueError("Retriever returned no usable text.")

    return context_used, retrieved_docs


def max_retrieval_score(retrieved_docs: Optional[List[Dict[str, Any]]]) -> float:
    if not retrieved_docs:
        return 0.0
    return max(
        (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)),
        default=0.0,
    )


def softmax_retrieval_weights(retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    if not retrieved_docs:
        return {}
    scores = [
        item.get("score", 0.0)
        for item in retrieved_docs
        if isinstance(item, dict) and "doc" in item
    ]
    max_s = max(scores) if scores else 0.0
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps) or 1.0
    weights = [e / total for e in exps]
    result: Dict[str, float] = {}
    for item, w in zip(retrieved_docs, weights):
        doc_text = item.get("doc", {}).get("text", "")
        if doc_text:
            result[doc_text] = w
    return result


def unit_retrieval_weight(unit: str, doc_weights: Dict[str, float]) -> float:
    unit_norm = re.escape(unit.strip())
    for doc_text, weight in doc_weights.items():
        if re.search(unit_norm, re.sub(r"\s+", " ", doc_text)):
            return weight
    return 1e-8


def softmax_normalise(scores_dict: Dict[str, float]) -> Dict[str, float]:
    if not scores_dict:
        return {}
    keys = list(scores_dict.keys())
    vals = torch.tensor([scores_dict[k] for k in keys], dtype=torch.float32)
    softmax_vals = torch_F.softmax(vals, dim=0)
    return {k: float(softmax_vals[i].item()) for i, k in enumerate(keys)}


@dataclass
class PerturbationSignals:
    raw_dissimilarities: Dict[str, float] = field(default_factory=dict)
    confidence_drops: Dict[str, float] = field(default_factory=dict)
    debug_details: Dict[str, Any] = field(default_factory=dict)


def run_perturbation_loop(
    *,
    gen: GeneratorWrapper,
    make_prompt: Callable[[str, str], str],
    context_used: str,
    question: str,
    units: List[str],
    perturber: str,
    comparator_fn: Callable[[str, str], float],
    original: str,
    c0: float,
    max_length: int,
    temperature: float,
    collect_confidence: bool,
    debug: bool = False,
    debug_max_perturbations: int = 5,
) -> PerturbationSignals:
    signals = PerturbationSignals()
    is_full_ctx_perturber = perturber == "leave_one_out"

    for i, unit in enumerate(units):
        perturbs = perturb_sentence(unit, perturber, full_text=context_used)
        if not perturbs:
            perturbs = [
                ""
                if is_full_ctx_perturber
                else " ".join(u for j, u in enumerate(units) if j != i)
            ]

        sims: List[float] = []
        conf_vals: List[float] = []
        samples: List[Dict[str, Any]] = []

        for p in perturbs:
            if is_full_ctx_perturber:
                new_ctx = p
            else:
                new_units = units.copy()
                new_units[i] = p
                new_ctx = " ".join(u for u in new_units if u is not None)

            new_prompt = make_prompt(new_ctx, question)

            if collect_confidence:
                pert_answer, pert_conf = generate_answer(
                    gen,
                    new_prompt,
                    max_length=max_length,
                    temperature=temperature,
                    return_confidence=True,
                )
                if pert_conf is not None:
                    conf_vals.append(pert_conf)
            else:
                pert_answer = generate_answer(
                    gen, new_prompt, max_length=max_length, temperature=temperature
                )

            sims.append(float(comparator_fn(pert_answer, original)))

            if debug and len(samples) < debug_max_perturbations:
                entry: Dict[str, Any] = {
                    "perturbed_unit": p,
                    "new_prompt": new_prompt,
                    "answer": pert_answer,
                    "similarity_to_original": sims[-1],
                }
                if collect_confidence and conf_vals:
                    entry["perturbed_confidence"] = conf_vals[-1]
                samples.append(entry)

        mean_sim = sum(sims) / max(1, len(sims))
        raw_dissim = float(1.0 - mean_sim)
        signals.raw_dissimilarities[unit] = raw_dissim

        if collect_confidence:
            mean_pert_conf = (
                sum(conf_vals) / max(1, len(conf_vals)) if conf_vals else c0
            )
            signals.confidence_drops[unit] = abs(c0 - mean_pert_conf)

        if debug:
            signals.debug_details[unit] = {
                "unit_index": i,
                "mean_similarity": float(mean_sim),
                "raw_dissimilarity": raw_dissim,
                "num_perturbations": len(perturbs),
                "samples": samples,
                **(
                    {"confidence_drop": signals.confidence_drops.get(unit, 0.0)}
                    if collect_confidence
                    else {}
                ),
            }

    return signals


def compute_importance(
    *,
    mode: str,
    units: List[str],
    raw_dissimilarities: Dict[str, float],
    confidence_drops: Dict[str, float],
    doc_weights: Dict[str, float],
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Return (normalized_importances, raw_rw_or_empty, raw_fusion_or_empty).
    Intermediate raw dicts are populated only for retrieval_weighted / fusion modes.
    """
    raw_rw: Dict[str, float] = {}
    raw_fusion: Dict[str, float] = {}

    if mode == "ragex_core":
        return softmax_normalise(raw_dissimilarities), raw_rw, raw_fusion

    if mode == "retrieval_weighted":
        normed_dissim = softmax_normalise(raw_dissimilarities)
        raw_rw = {
            unit: normed_dissim[unit] * unit_retrieval_weight(unit, doc_weights)
            for unit in units
        }
        return softmax_normalise(raw_rw), raw_rw, raw_fusion

    # confidence_retrieval_fusion: (α·retrieval + (1−α)·confidence_drop) · softmax(dissim)
    normed_dissim = softmax_normalise(raw_dissimilarities)
    for unit in units:
        raw_fusion[unit] = (
            alpha * unit_retrieval_weight(unit, doc_weights)
            + (1.0 - alpha) * confidence_drops.get(unit, 0.0)
        ) * normed_dissim.get(unit, 0.0)
    return softmax_normalise(raw_fusion), raw_rw, raw_fusion


def compute_all_mode_importances(
    *,
    units: List[str],
    raw_dissimilarities: Dict[str, float],
    confidence_drops: Dict[str, float],
    doc_weights: Dict[str, float],
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Baseline (PD), retrieval-weighted, and fusion importances in one pass."""
    baseline = compute_importance(
        mode="ragex_core",
        units=units,
        raw_dissimilarities=raw_dissimilarities,
        confidence_drops=confidence_drops,
        doc_weights=doc_weights,
        alpha=alpha,
    )[0]

    ret_weighted = compute_importance(
        mode="retrieval_weighted",
        units=units,
        raw_dissimilarities=raw_dissimilarities,
        confidence_drops=confidence_drops,
        doc_weights=doc_weights,
        alpha=alpha,
    )[0]

    fusion = compute_importance(
        mode="confidence_retrieval_fusion",
        units=units,
        raw_dissimilarities=raw_dissimilarities,
        confidence_drops=confidence_drops,
        doc_weights=doc_weights,
        alpha=alpha,
    )[0]

    return baseline, ret_weighted, fusion


def split_explanation_units(context_used: str, explanation_level: str) -> List[str]:
    units = split_context(context_used, explanation_level)
    if not units:
        units = [context_used]
    if len(units) > MAX_UNITS:
        logger.info("Capping explanation units from %d to %d", len(units), MAX_UNITS)
        units = units[:MAX_UNITS]
    return units
