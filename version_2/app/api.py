# app/api.py
import os
# Must be set before ANY library imports to prevent macOS OMP segfault
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

"""
RAG-Ex Core API

This version adds:
- GET / homepage
- GET /health readiness check
- Pluggable retrieval via req.retriever ("dense" | "bm25") if app.retriever.retrieve supports it
- Explanation granularity via req.explanation_level ("word" | "phrase" | "sentence" | "paragraph")
- Debug mode (req.debug=true) returning:
    - per-unit perturbed answers (sampled)
    - similarity scores
    - mean similarity, raw dissimilarity (pre-normalization)
    - normalized importance
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from app.config import settings
from app.generator import get_generator, generate_answer
from app.retriever import retrieve
from app.perturb import perturb_sentence, split_context
from app.comparator import COMPARATORS

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="RAG-Ex Core API",
    default_response_class=ORJSONResponse,
    version="1.1.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# ----------------------------
# Models
# ----------------------------
class RetrieveResponseItem(BaseModel):
    doc: Dict[str, Any]
    score: float


class ExplainRequest(BaseModel):
    # core
    question: str = Field(..., min_length=1, description="User question")

    # If context is empty, API uses retrieval (RAG mode).
    context: str = Field("", description="Optional context. If empty, retriever is used to build context.")

    # retrieval
    retriever: str = Field(
        default=getattr(settings, "RETRIEVER_DEFAULT", "dense"),
        description="Retriever to use when context is empty (RAG mode): dense | bm25 | hybrid",
    )
    top_k_docs: int = Field(3, ge=1, le=50, description="Number of docs to retrieve when context is empty")

    # explanation / perturbation
    explanation_level: str = Field(
        "sentence",
        description="Granularity of explanation: word | phrase | sentence | paragraph",
    )
    perturber: str = Field(
        "leave_one_out",
        description="Perturbation strategy: leave_one_out | random_noise | entity_perturber | antonym_perturber | synonym_perturber | reorder_perturber",
    )

    # comparison
    comparator: str = Field(
        "semantic",
        description="Comparator: semantic | levenshtein | jaro_winkler | n_gram",
    )

    # generation
    max_length: int = Field(256, ge=16, le=2048, description="Max generation length")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Generation temperature (0 = deterministic)")

    # confidence threshold
    confidence_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval score required to generate an answer (prevents hallucinations)",
    )

    # importance mode
    importance_mode: str = Field(
        "ragex_core",
        description=(
            "Importance scoring method: "
            "'ragex_core' — perturbation-based generator importance (baseline); "
            "'retrieval_weighted' — generator importance gated by retrieval relevance; "
            "'confidence_retrieval_fusion' — fusion of response dissimilarity, "
            "generation confidence drop, and retrieval relevance."
        ),
    )
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fusion weight used only by 'confidence_retrieval_fusion'. "
            "α=1.0 weights entirely on response dissimilarity; "
            "α=0.0 weights entirely on generation confidence drop; "
            "α=0.5 (default) gives equal weight to both generator-side signals "
            "before retrieval relevance gating is applied."
        ),
    )

    # debug/inspection
    debug: bool = Field(
        False,
        description="If true, return per-unit details incl. perturbed answers and similarity scores (can be large)",
    )
    debug_max_perturbations: int = Field(
        5,
        ge=1,
        le=50,
        description="Max number of perturbation samples to return per unit in debug mode",
    )


class ExplainResponse(BaseModel):
    
    context_used: str
    prompt: str
    original_answer: str
    token_importances: Dict[str, float]
    retrieved_docs: Optional[List[RetrieveResponseItem]] = None
    details: Optional[Dict[str, Any]] = None


# ----------------------------
# Startup / basic routes
# ----------------------------
@app.on_event("startup")
def startup_event():
    app.state.generator = get_generator(settings.HF_MODEL)
    logger.info("Generator initialized")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>RAG-Ex Core</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h2>RAG-Ex Core API is running ✅</h2>
        <p>Choose your interface:</p>
        <ul>
          <li><a href="/static/index.html"><strong>🚀 Interactive Web Interface</strong></a> (Recommended)</li>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/redoc">ReDoc</a></li>
          <li><a href="/health">Health check</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/health")
def health():
    ok = hasattr(app.state, "generator") and app.state.generator is not None
    return {"status": "ok" if ok else "not_ready"}


# ----------------------------
# API endpoints
# ----------------------------
@app.post("/retrieve", response_model=List[RetrieveResponseItem])
def api_retrieve(
    q: str = Query(..., description="Query string"),
    k: int = Query(3, ge=1, le=50, description="Top-k documents"),
    retriever_name: str = Query("dense", description="Retriever: dense | bm25"),
):
    """
    Retrieve top-k docs. Supports switching retrievers if app.retriever.retrieve supports it.
    """
    try:
        return retrieve(q, k, retriever=retriever_name)  # type: ignore[arg-type]
    except TypeError:
        # Backward compat: old retrieve(query, k) signature
        return retrieve(q, k)  # type: ignore[misc]
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Retriever index/doc store not found. "
                "Make sure you created data/docs.jsonl and built indices."
            ),
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e


# ---------------------------------------------------------------------------
# Helpers for retrieval relevance weighting
# ---------------------------------------------------------------------------

def _softmax_retrieval_weights(retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute a softmax-normalised weight for each retrieved document.
    Returns {doc_text: weight} so weights sum to 1.0.
    Falls back to uniform weights if scores are unavailable.
    """
    import math
    if not retrieved_docs:
        return {}
    scores = [
        item.get("score", 0.0)
        for item in retrieved_docs
        if isinstance(item, dict) and "doc" in item
    ]
    # Softmax
    max_s = max(scores) if scores else 0.0
    exps  = [math.exp(s - max_s) for s in scores]   # subtract max for numerical stability
    total = sum(exps)
    weights = [e / total for e in exps]

    result: Dict[str, float] = {}
    for item, w in zip(retrieved_docs, weights):
        doc_text = item.get("doc", {}).get("text", "")
        if doc_text:
            result[doc_text] = w
    return result


def _unit_retrieval_weight(
    unit: str,
    doc_weights: Dict[str, float],
    fallback: float = 1.0,
) -> float:
    """
    Return the retrieval weight of the document that contains `unit`.
    Uses substring matching.  Falls back to `fallback` (default 1.0) so
    'retrieval_weighted' and 'confidence_retrieval_fusion' gracefully degrade
    to the baseline when no retrieval scores are found (e.g. direct-context mode).
    """
    for doc_text, weight in doc_weights.items():
        if unit in doc_text:
            return weight
    return fallback


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    gen = getattr(app.state, "generator", None)
    if gen is None:
        raise HTTPException(status_code=503, detail="Generator not ready. Try again in a moment.")

    comparator_fn = COMPARATORS.get(req.comparator)
    if comparator_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown comparator '{req.comparator}'")

    # ----------------------------
    # Context building (direct vs RAG)
    # ----------------------------
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    context_used = (req.context or "").strip()

    if not context_used:
        try:
            try:
                retrieved_docs = retrieve(req.question, req.top_k_docs, retriever=req.retriever)  # type: ignore[arg-type]
            except TypeError:
                retrieved_docs = retrieve(req.question, req.top_k_docs)  # type: ignore[misc]

            context_used = "\n".join(
                [
                    item["doc"]["text"]
                    for item in (retrieved_docs or [])
                    if isinstance(item, dict)
                    and "doc" in item
                    and isinstance(item["doc"], dict)
                    and "text" in item["doc"]
                ]
            ).strip()

            if not context_used:
                raise HTTPException(
                    status_code=500,
                    detail="Retriever returned no usable text. Ensure docs.jsonl contains 'text' fields.",
                )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No context provided AND retriever index/doc store missing. "
                    "Create data/docs.jsonl and run index building."
                ),
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}") from e

    # ----------------------------
    # Confidence threshold check
    # ----------------------------
    # Check confidence only if we retrieved documents (RAG mode)
    if retrieved_docs and len(retrieved_docs) > 0:
        # Get the max score from retrieved docs
        max_score = max(
            (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)),
            default=0.0
        )
        
        if max_score < req.confidence_threshold:
            # Return low confidence response without generating answer
            return ExplainResponse(
                context_used=context_used,
                prompt=f"Context: {context_used}\nQuestion: {req.question}\nAnswer:",
                original_answer="Sorry, I am not confident to answer your question.",
                token_importances={},
                retrieved_docs=retrieved_docs,
                details={"confidence_check": {
                    "max_retrieval_score": max_score,
                    "threshold": req.confidence_threshold,
                    "reason": "Retrieved documents have low relevance scores"
                }} if req.debug else None,
            )

    # Validate importance_mode early
    valid_modes = {"ragex_core", "retrieval_weighted", "confidence_retrieval_fusion"}
    if req.importance_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown importance_mode '{req.importance_mode}'. Choose from: {sorted(valid_modes)}",
        )

    # Pre-compute retrieval relevance weights once (used by retrieval_weighted
    # and confidence_retrieval_fusion modes). Empty dict in direct-context mode;
    # _unit_retrieval_weight falls back to 1.0 (no-op) when no match is found.
    doc_weights = _softmax_retrieval_weights(retrieved_docs or [])
    needs_confidence = (req.importance_mode == "confidence_retrieval_fusion")

    # ----------------------------
    # Baseline generation
    # ----------------------------
    prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"

    if needs_confidence:
        original, c0 = generate_answer(
            gen, prompt, max_length=req.max_length, temperature=req.temperature,
            return_confidence=True,
        )
        c0 = c0 if c0 is not None else 1.0   # default to 1.0 if unavailable
    else:
        original = generate_answer(gen, prompt, max_length=req.max_length, temperature=req.temperature)
        c0 = 1.0  # unused but defined for clarity

    # ----------------------------
    # Split context into explanation units
    # ----------------------------
    try:
        units = split_context(context_used, req.explanation_level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

    if not units:
        units = [context_used]

    # ----------------------------
    # Perturbation loop
    # ----------------------------
    raw_dissimilarities: Dict[str, float] = {}
    confidence_drops: Dict[str, float] = {}  # populated only for confidence_retrieval_fusion
    debug_details: Dict[str, Any] = {} if req.debug else {}

    # leave_one_out returns the full perturbed context (paper behaviour);
    # all other strategies return a replacement string for the unit only.
    is_full_ctx_perturber = (req.perturber == "leave_one_out")

    for i, unit in enumerate(units):
        perturbs = perturb_sentence(unit, req.perturber, full_text=context_used)
        if not perturbs:
            perturbs = ["" if is_full_ctx_perturber else " ".join(
                u for j, u in enumerate(units) if j != i
            )]

        sims: List[float] = []
        conf_vals: List[float] = []   # perturbed confidences (confidence_retrieval_fusion only)
        samples: List[Dict[str, Any]] = []

        for p in perturbs:
            if is_full_ctx_perturber:
                new_ctx = p
            else:
                new_units = units.copy()
                new_units[i] = p
                new_ctx = " ".join([u for u in new_units if u is not None])

            new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"

            if needs_confidence:
                # confidence_retrieval_fusion: collect generation confidence alongside answer
                pert_answer, pert_conf = generate_answer(
                    gen, new_prompt, max_length=req.max_length,
                    temperature=req.temperature, return_confidence=True,
                )
                if pert_conf is not None:
                    conf_vals.append(pert_conf)
            else:
                pert_answer = generate_answer(
                    gen, new_prompt, max_length=req.max_length, temperature=req.temperature
                )

            sim = float(comparator_fn(pert_answer, original))
            sims.append(sim)

            if req.debug and len(samples) < req.debug_max_perturbations:
                sample_entry: Dict[str, Any] = {
                    "perturbed_unit": p,
                    "new_prompt": new_prompt,
                    "answer": pert_answer,
                    "similarity_to_original": sim,
                }
                if needs_confidence and conf_vals:
                    sample_entry["perturbed_confidence"] = conf_vals[-1]
                samples.append(sample_entry)

        mean_sim   = sum(sims) / max(1, len(sims))
        raw_dissim = 1.0 - mean_sim
        raw_dissimilarities[unit] = float(raw_dissim)

        # Confidence drop (confidence_retrieval_fusion): Δc_i = max(0, c0 - mean(c_i^(j)))
        if needs_confidence:
            mean_pert_conf  = sum(conf_vals) / max(1, len(conf_vals)) if conf_vals else c0
            delta_c         = max(0.0, c0 - mean_pert_conf)
            confidence_drops[unit] = float(delta_c)

        if req.debug:
            debug_entry: Dict[str, Any] = {
                "unit_index": i,
                "mean_similarity": float(mean_sim),
                "raw_dissimilarity": float(raw_dissim),
                "num_perturbations": len(perturbs),
                "samples": samples,
            }
            if needs_confidence:
                debug_entry["confidence_drop"] = confidence_drops.get(unit, 0.0)
            debug_details[unit] = debug_entry

    # ----------------------------
    # Compute final importance scores based on importance_mode
    # ----------------------------

    def _normalise(scores_dict: Dict[str, float]) -> Dict[str, float]:
        """Paper formula: w_i = w'_i / max_j w'_j  (most important unit → 1.0)"""
        vals = list(scores_dict.values())
        maxv = max(vals) if vals else 1.0
        if maxv <= 0:
            return {k: 0.0 for k in scores_dict}
        return {k: float(v / maxv) for k, v in scores_dict.items()}

    if req.importance_mode == "ragex_core":
        # ── Perturbation-based generator importance (baseline) ────────────
        # w_i = w'_i / max_j w'_j
        normalized = _normalise(raw_dissimilarities)

    elif req.importance_mode == "retrieval_weighted":
        # ── Generator importance gated by retrieval relevance ─────────────
        # Multiplies each unit's generator importance by the softmax-normalised
        # retrieval relevance score of the document it originates from.
        # w_i = normalise( w_i^baseline * r̃_d(i) )
        # Degrades gracefully to baseline when no retrieval scores exist.
        normed_baseline = _normalise(raw_dissimilarities)
        raw_retrieval_weighted = {
            unit: normed_baseline[unit] * _unit_retrieval_weight(unit, doc_weights)
            for unit in units
        }
        normalized = _normalise(raw_retrieval_weighted)

    else:
        # ── Confidence–retrieval fusion ────────────────────────────────────
        # Fuses two generator-side signals (response dissimilarity + generation
        # confidence drop) via α, then gates the result by retrieval relevance.
        # w_i = normalise( [α·w'_i + (1−α)·Δc_i] · r̃_d(i) )
        raw_fusion: Dict[str, float] = {}
        for unit in units:
            dissim          = raw_dissimilarities.get(unit, 0.0)
            confidence_drop = confidence_drops.get(unit, 0.0)
            retrieval_rel   = _unit_retrieval_weight(unit, doc_weights)
            raw_fusion[unit] = (req.alpha * dissim + (1.0 - req.alpha) * confidence_drop) * retrieval_rel
        normalized = _normalise(raw_fusion)

    if req.debug:
        for unit, info in debug_details.items():
            info["normalized_importance"] = normalized.get(unit, 0.0)
            if req.importance_mode in ("retrieval_weighted", "confidence_retrieval_fusion"):
                info["retrieval_weight"] = _unit_retrieval_weight(unit, doc_weights)
        debug_details["_baseline"] = {
            "original_answer": original,
            "prompt": prompt,
            "importance_mode": req.importance_mode,
            "alpha": req.alpha if req.importance_mode == "confidence_retrieval_fusion" else None,
            "baseline_confidence": c0 if needs_confidence else None,
        }

    return ExplainResponse(
        
        context_used=context_used,
        prompt=prompt,
        original_answer=original,
        token_importances=normalized,
        retrieved_docs=retrieved_docs,
        details=debug_details if req.debug else None,
    )


