# # app/api.py
# """
# RAG-Ex Core API

# This version adds:
# - GET / homepage
# - GET /health readiness check
# - Pluggable retrieval via req.retriever ("dense" | "bm25") if app.retriever.retrieve supports it
# - Explanation granularity via req.explanation_level ("word" | "phrase" | "sentence" | "paragraph")
# - Debug mode (req.debug=true) returning:
#     - per-unit perturbed answers (sampled)
#     - similarity scores
#     - mean similarity, raw dissimilarity (pre-normalization)
#     - normalized importance
# """

# import logging
# from typing import Any, Dict, List, Optional

# from fastapi import FastAPI, HTTPException, Query
# from fastapi.responses import HTMLResponse, ORJSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import os

# from app.config import settings
# from app.generator import get_generator, generate_answer
# from app.retriever import retrieve
# from app.perturb import perturb_sentence, split_context
# from app.comparator import COMPARATORS

# logger = logging.getLogger("uvicorn.error")

# app = FastAPI(
#     title="RAG-Ex Core API",
#     default_response_class=ORJSONResponse,
#     version="1.1.1",
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, specify exact origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static files
# static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
# if os.path.exists(static_path):
#     app.mount("/static", StaticFiles(directory=static_path), name="static")

# # ----------------------------
# # Models
# # ----------------------------
# class RetrieveResponseItem(BaseModel):
#     doc: Dict[str, Any]
#     score: float


# class ExplainRequest(BaseModel):
#     # core
#     question: str = Field(..., min_length=1, description="User question")

#     # If context is empty, API uses retrieval (RAG mode).
#     context: str = Field("", description="Optional context. If empty, retriever is used to build context.")

#     # retrieval
#     retriever: str = Field(
#         default=getattr(settings, "RETRIEVER_DEFAULT", "dense"),
#         description="Retriever to use when context is empty (RAG mode): dense | bm25 | hybrid",
#     )
#     top_k_docs: int = Field(3, ge=1, le=50, description="Number of docs to retrieve when context is empty")

#     # explanation / perturbation
#     explanation_level: str = Field(
#         "sentence",
#         description="Granularity of explanation: word | phrase | sentence | paragraph",
#     )
#     perturber: str = Field(
#         "leave_one_out",
#         description="Perturbation strategy: leave_one_out | random_noise | entity_perturber | antonym_perturber | synonym_perturber | reorder_perturber",
#     )

#     # comparison
#     comparator: str = Field(
#         "semantic",
#         description="Comparator: semantic | levenshtein | jaro_winkler | n_gram",
#     )

#     # generation
#     max_length: int = Field(256, ge=16, le=2048, description="Max generation length")
#     temperature: float = Field(0.0, ge=0.0, le=2.0, description="Generation temperature (0 = deterministic)")

#     # confidence threshold
#     confidence_threshold: float = Field(
#         0.6,
#         ge=0.0,
#         le=1.0,
#         description="Minimum retrieval score required to generate an answer (prevents hallucinations)",
#     )

#     # debug/inspection
#     debug: bool = Field(
#         False,
#         description="If true, return per-unit details incl. perturbed answers and similarity scores (can be large)",
#     )
#     debug_max_perturbations: int = Field(
#         5,
#         ge=1,
#         le=50,
#         description="Max number of perturbation samples to return per unit in debug mode",
#     )


# class ExplainResponse(BaseModel):
    
#     context_used: str
#     prompt: str
#     original_answer: str
#     token_importances: Dict[str, float]
#     retrieved_docs: Optional[List[RetrieveResponseItem]] = None
#     details: Optional[Dict[str, Any]] = None


# # ----------------------------
# # Startup / basic routes
# # ----------------------------
# @app.on_event("startup")
# def startup_event():
#     app.state.generator = get_generator(settings.HF_MODEL)
#     logger.info("Generator initialized")


# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <html>
#       <head><title>RAG-Ex Core</title></head>
#       <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
#         <h2>RAG-Ex Core API is running ✅</h2>
#         <p>Choose your interface:</p>
#         <ul>
#           <li><a href="/static/index.html"><strong>🚀 Interactive Web Interface</strong></a> (Recommended)</li>
#           <li><a href="/docs">Swagger UI</a></li>
#           <li><a href="/redoc">ReDoc</a></li>
#           <li><a href="/health">Health check</a></li>
#         </ul>
#       </body>
#     </html>
#     """


# @app.get("/health")
# def health():
#     ok = hasattr(app.state, "generator") and app.state.generator is not None
#     return {"status": "ok" if ok else "not_ready"}


# # ----------------------------
# # API endpoints
# # ----------------------------
# @app.post("/retrieve", response_model=List[RetrieveResponseItem])
# def api_retrieve(
#     q: str = Query(..., description="Query string"),
#     k: int = Query(3, ge=1, le=50, description="Top-k documents"),
#     retriever_name: str = Query("dense", description="Retriever: dense | bm25"),
# ):
#     """
#     Retrieve top-k docs. Supports switching retrievers if app.retriever.retrieve supports it.
#     """
#     try:
#         return retrieve(q, k, retriever=retriever_name)  # type: ignore[arg-type]
#     except TypeError:
#         # Backward compat: old retrieve(query, k) signature
#         return retrieve(q, k)  # type: ignore[misc]
#     except FileNotFoundError as e:
#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 "Retriever index/doc store not found. "
#                 "Make sure you created data/docs.jsonl and built indices."
#             ),
#         ) from e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e


# @app.post("/explain", response_model=ExplainResponse)
# def explain(req: ExplainRequest):
#     gen = getattr(app.state, "generator", None)
#     if gen is None:
#         raise HTTPException(status_code=503, detail="Generator not ready. Try again in a moment.")

#     comparator_fn = COMPARATORS.get(req.comparator)
#     if comparator_fn is None:
#         raise HTTPException(status_code=400, detail=f"Unknown comparator '{req.comparator}'")

#     # ----------------------------
#     # Context building (direct vs RAG)
#     # ----------------------------
#     retrieved_docs: Optional[List[Dict[str, Any]]] = None
#     context_used = (req.context or "").strip()

#     if not context_used:
#         try:
#             try:
#                 retrieved_docs = retrieve(req.question, req.top_k_docs, retriever=req.retriever)  # type: ignore[arg-type]
#             except TypeError:
#                 retrieved_docs = retrieve(req.question, req.top_k_docs)  # type: ignore[misc]

#             context_used = "\n".join(
#                 [
#                     item["doc"]["text"]
#                     for item in (retrieved_docs or [])
#                     if isinstance(item, dict)
#                     and "doc" in item
#                     and isinstance(item["doc"], dict)
#                     and "text" in item["doc"]
#                 ]
#             ).strip()

#             if not context_used:
#                 raise HTTPException(
#                     status_code=500,
#                     detail="Retriever returned no usable text. Ensure docs.jsonl contains 'text' fields.",
#                 )
#         except FileNotFoundError as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=(
#                     "No context provided AND retriever index/doc store missing. "
#                     "Create data/docs.jsonl and run index building."
#                 ),
#             ) from e
#         except HTTPException:
#             raise
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}") from e

#     # ----------------------------
#     # Confidence threshold check
#     # ----------------------------
#     # Check confidence only if we retrieved documents (RAG mode)
#     if retrieved_docs and len(retrieved_docs) > 0:
#         # Get the max score from retrieved docs
#         max_score = max(
#             (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)),
#             default=0.0
#         )
        
#         if max_score < req.confidence_threshold:
#             # Return low confidence response without generating answer
#             return ExplainResponse(
#                 context_used=context_used,
#                 prompt=f"Context: {context_used}\nQuestion: {req.question}\nAnswer:",
#                 original_answer="Sorry, I am not confident to answer your question.",
#                 token_importances={},
#                 retrieved_docs=retrieved_docs,
#                 details={"confidence_check": {
#                     "max_retrieval_score": max_score,
#                     "threshold": req.confidence_threshold,
#                     "reason": "Retrieved documents have low relevance scores"
#                 }} if req.debug else None,
#             )

#     # ----------------------------
#     # Baseline generation
#     # ----------------------------
#     prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
#     original = generate_answer(gen, prompt, max_length=req.max_length, temperature=req.temperature)

#     # ----------------------------
#     # Split context into explanation units
#     # ----------------------------
#     try:
#         units = split_context(context_used, req.explanation_level)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

#     if not units:
#         units = [context_used]

#     # ----------------------------
#     # Perturbation loop
#     # ----------------------------
#     raw_dissimilarities: Dict[str, float] = {}
#     debug_details: Dict[str, Any] = {} if req.debug else {}

#     for i, unit in enumerate(units):
#         perturbs = perturb_sentence(unit, req.perturber)
#         if not perturbs:
#             perturbs = [""]  # simulate removal

#         sims: List[float] = []
#         samples: List[Dict[str, Any]] = []

#         for p in perturbs:
#             new_units = units.copy()
#             new_units[i] = p
#             new_ctx = " ".join([u for u in new_units if u is not None])

#             new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
#             pert_answer = generate_answer(gen, new_prompt, max_length=req.max_length, temperature=req.temperature)

#             sim = float(comparator_fn(pert_answer, original))
#             sims.append(sim)

#             if req.debug and len(samples) < req.debug_max_perturbations:
#                 samples.append(
#                     {
#                         "perturbed_unit": p,
#                         "new_prompt": new_prompt,
#                         "answer": pert_answer,
#                         "similarity_to_original": sim,
#                     }
#                 )

#         mean_sim = sum(sims) / max(1, len(sims))
#         raw_dissim = 1.0 - mean_sim
#         raw_dissimilarities[unit] = float(raw_dissim)

#         if req.debug:
#             debug_details[unit] = {
#                 "unit_index": i,
#                 "mean_similarity": float(mean_sim),
#                 "raw_dissimilarity": float(raw_dissim),
#                 "num_perturbations": len(perturbs),
#                 "samples": samples,
#             }

#     # ----------------------------
#     # Normalize importance scores
#     # ----------------------------
#     maxv = max(raw_dissimilarities.values()) if raw_dissimilarities else 1.0
#     if maxv <= 0:
#         normalized = {k: 0.0 for k in raw_dissimilarities}
#     else:
#         normalized = {k: float(v / maxv) for k, v in raw_dissimilarities.items()}

#     if req.debug:
#         for unit, info in debug_details.items():
#             info["normalized_importance"] = normalized.get(unit, 0.0)
#         # also provide baseline answer for convenience
#         debug_details["_baseline"] = {"original_answer": original, "prompt": prompt}

#     return ExplainResponse(
        
#         context_used=context_used,
#         prompt=prompt,
#         original_answer=original,
#         token_importances=normalized,
#         retrieved_docs=retrieved_docs,
#         details=debug_details if req.debug else None,
#     )

"""
RAG-Ex Core API with modified formulas on confidence, doc relevance showing comparison endpoint.

This version adds:
- GET / homepage
- GET /health readiness check
- Pluggable retrieval via req.retriever ("dense" | "bm25" | "hybrid") if app.retriever.retrieve supports it
- Explanation granularity via req.explanation_level ("word" | "phrase" | "sentence" | "paragraph")
- Debug mode (req.debug=true) returning:
    - per-unit perturbed answers (sampled)
    - similarity scores
    - mean similarity, raw dissimilarity (pre-normalization)
    - normalized importance

NEW (Comparison endpoint):
- POST /compare: returns baseline RAG-Ex scores AND modified fusion scores side-by-side.
- Fusion methods are explicitly named (no A/B/C exposed):
    * multiplicative_gated_weighted_additive
    * weighted_additive
    * triple_multiplicative
- Exposes doc relevance scores (normalized) and model confidence signals (sequence + token logprobs optionally)
- Fusion lambda is configurable via API.
- Doc relevance is correctly applied per-unit across all granularities by splitting per-document and perturbing within that document.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

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
    version="1.2.0",
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

    # confidence threshold (retrieval score)
    confidence_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval score required to generate an answer (prevents hallucinations)",
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

    # ----------------------------
    # NEW: fusion comparison controls (used by /compare)
    # ----------------------------
    fusion_methods: List[str] = Field(
        default_factory=lambda: [
            "multiplicative_gated_weighted_additive",
            "weighted_additive",
            "triple_multiplicative",
        ],
        description=(
            "Fusion methods to compute for modified scoring. Options: "
            "multiplicative_gated_weighted_additive | weighted_additive | triple_multiplicative"
        ),
    )
    fusion_lambda: float = Field(
        0.5, ge=0.0, le=1.0, description="Lambda ∈ [0,1] weighting doc relevance vs confidence drop in weighted additive fusion."
    )

    # NEW: confidence outputs (if supported by generator)
    return_sequence_confidence: bool = Field(
        False, description="If true, include sequence-level confidence scores (if generator supports it)."
    )
    return_token_logprobs: bool = Field(
        False, description="If true, include per-token logprobs for baseline generation (if generator supports it)."
    )


class ExplainResponse(BaseModel):
    context_used: str
    prompt: str
    original_answer: str
    token_importances: Dict[str, float]
    retrieved_docs: Optional[List[RetrieveResponseItem]] = None
    details: Optional[Dict[str, Any]] = None


class CompareResponse(BaseModel):
    """
    Response for /compare endpoint: baseline + modified fusion variants.
    Keeps baseline identical to /explain, but adds modified scores and signals.
    """
    context_used: str
    prompt: str
    original_answer: str

    # baseline: same as /explain
    baseline_token_importances: Dict[str, float]

    # modified: method_name -> {unit_text: score}
    modified_token_importances: Dict[str, Dict[str, float]]

    # signals (unit-aligned, keyed by unit_text for convenience)
    unit_doc_relevance_norm: Dict[str, float]
    unit_confidence_drop_norm: Dict[str, float]
    unit_response_sensitivity_baseline: Dict[str, float]

    # retrieval visibility
    retrieved_docs: Optional[List[RetrieveResponseItem]] = None
    doc_scores_raw: Optional[List[float]] = None
    doc_scores_norm: Optional[List[float]] = None

    # model confidence visibility (if available)
    baseline_sequence_confidence: Optional[float] = None
    baseline_token_texts: Optional[List[str]] = None
    baseline_token_logprobs: Optional[List[float]] = None

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
# Helpers
# ----------------------------
def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin == 0:
        # all equal: treat as 1.0 across docs for interpretability
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _safe_generate(
    gen: Any,
    prompt: str,
    max_length: int,
    temperature: float,
    return_sequence_confidence: bool,
    return_token_logprobs: bool,
) -> Tuple[str, Optional[float], Optional[List[str]], Optional[List[float]]]:
    """
    Backward compatible wrapper around generate_answer.
    If your generator supports extra outputs, return them.
    Otherwise, return (answer, None, None, None).
    """
    try:
        # Newer signature (if you implemented it):
        # generate_answer(gen, prompt, max_length=..., temperature=..., return_sequence_confidence=..., return_token_logprobs=...)
        out = generate_answer(
            gen,
            prompt,
            max_length=max_length,
            temperature=temperature,
            return_sequence_confidence=return_sequence_confidence,
            return_token_logprobs=return_token_logprobs,
        )
        # Expected: (answer, seq_conf, token_texts, token_logprobs)
        if isinstance(out, tuple) and len(out) == 4:
            return out  # type: ignore[return-value]
        # If someone returns (answer, seq_conf) only:
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1], None, None  # type: ignore[return-value]
        # Else just answer
        return str(out), None, None, None
    except TypeError:
        # Old signature: generate_answer(gen, prompt, max_length=..., temperature=...)
        ans = generate_answer(gen, prompt, max_length=max_length, temperature=temperature)
        return ans, None, None, None


def _extract_docs_and_scores(retrieved_docs: Optional[List[Dict[str, Any]]], context_used: str) -> Tuple[List[str], List[float]]:
    """
    Returns docs_texts and doc_scores (raw).
    If context_used is provided directly, treat it as a single doc with score 1.0.
    """
    if retrieved_docs:
        docs_texts = []
        doc_scores = []
        for item in retrieved_docs:
            if (
                isinstance(item, dict)
                and "doc" in item
                and isinstance(item["doc"], dict)
                and "text" in item["doc"]
            ):
                docs_texts.append(str(item["doc"]["text"]))
                doc_scores.append(float(item.get("score", 0.0)))
        if docs_texts:
            return docs_texts, doc_scores

    # direct context mode fallback
    return [context_used], [1.0]


def _split_per_doc(docs_texts: List[str], explanation_level: str) -> Tuple[List[str], List[int]]:
    """
    Split each document separately to preserve unit->doc mapping.
    Returns:
      - units_flat: list of unit strings
      - unit_doc_idx: list mapping unit index -> doc index
    """
    units_flat: List[str] = []
    unit_doc_idx: List[int] = []
    for d_idx, doc_text in enumerate(docs_texts):
        doc_units = split_context(doc_text, explanation_level)
        if not doc_units:
            doc_units = [doc_text]
        for u in doc_units:
            units_flat.append(u)
            unit_doc_idx.append(d_idx)
    return units_flat, unit_doc_idx


def _build_context_from_docs(docs_texts: List[str]) -> str:
    return "\n".join([d for d in docs_texts if d is not None]).strip()


def _compute_fusions(
    unit_texts: List[str],
    baseline_norm: List[float],
    doc_rel_norm: List[float],
    conf_drop_norm: List[float],
    fusion_methods: List[str],
    fusion_lambda: float,
) -> Dict[str, List[float]]:
    """
    Returns method_name -> normalized scores aligned with unit_texts
    Methods:
      - multiplicative_gated_weighted_additive:
          baseline_norm * (λ*doc_rel_norm + (1-λ)*conf_drop_norm)
      - weighted_additive:
          (λ*doc_rel_norm + (1-λ)*conf_drop_norm)
      - triple_multiplicative:
          baseline_norm * doc_rel_norm * conf_drop_norm
    """
    out: Dict[str, List[float]] = {}

    def _normalize_max(vals: List[float]) -> List[float]:
        if not vals:
            return []
        m = max(vals)
        if m <= 0:
            return [0.0 for _ in vals]
        return [float(v / m) for v in vals]

    for method in fusion_methods:
        raw: List[float] = []
        for i in range(len(unit_texts)):
            if method == "multiplicative_gated_weighted_additive":
                fused = fusion_lambda * doc_rel_norm[i] + (1.0 - fusion_lambda) * conf_drop_norm[i]
                raw.append(baseline_norm[i] * fused)
            elif method == "weighted_additive":
                fused = fusion_lambda * doc_rel_norm[i] + (1.0 - fusion_lambda) * conf_drop_norm[i]
                raw.append(fused)
            elif method == "triple_multiplicative":
                raw.append(baseline_norm[i] * doc_rel_norm[i] * conf_drop_norm[i])
            else:
                # ignore unknown method names
                continue

        out[method] = _normalize_max(raw)

    return out


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


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """
    Original baseline explain endpoint. Keeps response contract stable:
    returns baseline token_importances only (paper-style).
    """
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
    # Confidence threshold check (retrieval scores)
    # ----------------------------
    if retrieved_docs and len(retrieved_docs) > 0:
        max_score = max(
            (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)),
            default=0.0
        )
        if max_score < req.confidence_threshold:
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

    # ----------------------------
    # Baseline generation
    # ----------------------------
    prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
    original, _, _, _ = _safe_generate(
        gen, prompt, req.max_length, req.temperature,
        return_sequence_confidence=False,
        return_token_logprobs=False
    )

    # ----------------------------
    # Split context into explanation units (original behavior)
    # ----------------------------
    try:
        units = split_context(context_used, req.explanation_level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

    if not units:
        units = [context_used]

    # ----------------------------
    # Perturbation loop (original behavior)
    # ----------------------------
    raw_dissimilarities: Dict[str, float] = {}
    debug_details: Dict[str, Any] = {} if req.debug else {}

    for i, unit in enumerate(units):
        perturbs = perturb_sentence(unit, req.perturber)
        if not perturbs:
            perturbs = [""]  # simulate removal

        sims: List[float] = []
        samples: List[Dict[str, Any]] = []

        for p in perturbs:
            new_units = units.copy()
            new_units[i] = p
            new_ctx = " ".join([u for u in new_units if u is not None])

            new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
            pert_answer, _, _, _ = _safe_generate(
                gen, new_prompt, req.max_length, req.temperature,
                return_sequence_confidence=False,
                return_token_logprobs=False
            )

            sim = float(comparator_fn(pert_answer, original))
            sims.append(sim)

            if req.debug and len(samples) < req.debug_max_perturbations:
                samples.append(
                    {
                        "perturbed_unit": p,
                        "new_prompt": new_prompt,
                        "answer": pert_answer,
                        "similarity_to_original": sim,
                    }
                )

        mean_sim = sum(sims) / max(1, len(sims))
        raw_dissim = 1.0 - mean_sim
        raw_dissimilarities[unit] = float(raw_dissim)

        if req.debug:
            debug_details[unit] = {
                "unit_index": i,
                "mean_similarity": float(mean_sim),
                "raw_dissimilarity": float(raw_dissim),
                "num_perturbations": len(perturbs),
                "samples": samples,
            }

    # ----------------------------
    # Normalize importance scores (original RAG-Ex baseline)
    # ----------------------------
    maxv = max(raw_dissimilarities.values()) if raw_dissimilarities else 1.0
    if maxv <= 0:
        normalized = {k: 0.0 for k in raw_dissimilarities}
    else:
        normalized = {k: float(v / maxv) for k, v in raw_dissimilarities.items()}

    if req.debug:
        for unit, info in debug_details.items():
            info["normalized_importance"] = normalized.get(unit, 0.0)
        debug_details["_baseline"] = {"original_answer": original, "prompt": prompt}

    return ExplainResponse(
        context_used=context_used,
        prompt=prompt,
        original_answer=original,
        token_importances=normalized,
        retrieved_docs=retrieved_docs,
        details=debug_details if req.debug else None,
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: ExplainRequest):
    """
    New endpoint: compares baseline with modified fusion approaches.

    Returns:
      - baseline_token_importances 
      - modified_token_importances per fusion method
      - signals: doc relevance norm, confidence drop norm, baseline response sensitivity
      - optional: baseline sequence confidence + token logprobs (if generator supports it)
    """
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
    # Confidence threshold check (retrieval scores)
    # ----------------------------
    if retrieved_docs and len(retrieved_docs) > 0:
        max_score = max(
            (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)),
            default=0.0
        )
        if max_score < req.confidence_threshold:
            # return early, but structured for compare
            return CompareResponse(
                context_used=context_used,
                prompt=f"Context: {context_used}\nQuestion: {req.question}\nAnswer:",
                original_answer="Sorry, I am not confident to answer your question.",
                baseline_token_importances={},
                modified_token_importances={},
                unit_doc_relevance_norm={},
                unit_confidence_drop_norm={},
                unit_response_sensitivity_baseline={},
                retrieved_docs=retrieved_docs,
                doc_scores_raw=[float(item.get("score", 0.0)) for item in retrieved_docs if isinstance(item, dict)] if retrieved_docs else None,
                doc_scores_norm=None,
                baseline_sequence_confidence=None,
                baseline_token_texts=None,
                baseline_token_logprobs=None,
                details={"confidence_check": {
                    "max_retrieval_score": max_score,
                    "threshold": req.confidence_threshold,
                    "reason": "Retrieved documents have low relevance scores"
                }} if req.debug else None,
            )

    # ----------------------------
    # Prepare docs + doc scores for fusion
    # ----------------------------
    docs_texts, doc_scores_raw = _extract_docs_and_scores(retrieved_docs, context_used)
    doc_scores_norm = _minmax_norm(doc_scores_raw)

    # ----------------------------
    # Baseline generation (with optional confidence outputs)
    # ----------------------------
    prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
    original, base_seq_conf, base_token_texts, base_token_logprobs = _safe_generate(
        gen,
        prompt,
        req.max_length,
        req.temperature,
        return_sequence_confidence=req.return_sequence_confidence or True,  # ensure we can compute ΔC if supported
        return_token_logprobs=req.return_token_logprobs,
    )

    # If generator doesn't provide it, keep confidence at 0.0 for math
    baseline_conf_for_math = float(base_seq_conf) if base_seq_conf is not None else 0.0

    # ----------------------------
    # Split into units PER DOC to preserve unit->doc association
    # ----------------------------
    try:
        units_flat, unit_doc_idx = _split_per_doc(docs_texts, req.explanation_level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

    if not units_flat:
        units_flat = [context_used]
        unit_doc_idx = [0]

    # ----------------------------
    # Perturbation loop (per unit, perturb inside its document)
    # Collect:
    #   - baseline raw dissimilarity per unit (like RAG-Ex)
    #   - mean perturbed confidence per unit (for ΔC)
    # ----------------------------
    raw_dissim_by_unit: Dict[str, float] = {}
    mean_pert_conf_by_unit: Dict[str, float] = {}
    debug_details: Dict[str, Any] = {} if req.debug else {}

    for i, unit in enumerate(units_flat):
        d_idx = unit_doc_idx[i]
        perturbs = perturb_sentence(unit, req.perturber)
        if not perturbs:
            perturbs = [""]  # simulate removal

        sims: List[float] = []
        pert_confs: List[float] = []
        samples: List[Dict[str, Any]] = []

        for p in perturbs:
            # rebuild docs with this unit perturbed within its own doc
            docs_copy = list(docs_texts)
            doc_text = docs_copy[d_idx]

            if unit in doc_text:
                doc_text_pert = doc_text.replace(unit, p, 1)
            else:
                # fallback if split produced mismatched whitespace; append perturb
                doc_text_pert = doc_text + "\n" + p

            docs_copy[d_idx] = doc_text_pert
            new_ctx = _build_context_from_docs(docs_copy)

            new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
            pert_answer, pert_seq_conf, _, _ = _safe_generate(
                gen,
                new_prompt,
                req.max_length,
                req.temperature,
                return_sequence_confidence=True,   # needed for ΔC if supported
                return_token_logprobs=False,
            )

            sim = float(comparator_fn(pert_answer, original))
            sims.append(sim)

            pert_confs.append(float(pert_seq_conf) if pert_seq_conf is not None else 0.0)

            if req.debug and len(samples) < req.debug_max_perturbations:
                samples.append(
                    {
                        "doc_index": d_idx,
                        "perturbed_unit": p,
                        "new_prompt": new_prompt,
                        "answer": pert_answer,
                        "similarity_to_original": sim,
                        "sequence_confidence": float(pert_seq_conf) if pert_seq_conf is not None else None,
                    }
                )

        mean_sim = sum(sims) / max(1, len(sims))
        raw_dissim = 1.0 - mean_sim
        raw_dissim_by_unit[unit] = float(raw_dissim)

        mean_pert_conf = sum(pert_confs) / max(1, len(pert_confs))
        mean_pert_conf_by_unit[unit] = float(mean_pert_conf)

        if req.debug:
            debug_details[unit] = {
                "unit_index": i,
                "doc_index": d_idx,
                "mean_similarity": float(mean_sim),
                "raw_dissimilarity": float(raw_dissim),
                "num_perturbations": len(perturbs),
                "samples": samples,
            }

    # ----------------------------
    # Baseline normalization (paper-style)
    # ----------------------------
    maxv = max(raw_dissim_by_unit.values()) if raw_dissim_by_unit else 1.0
    if maxv <= 0:
        baseline_norm = {k: 0.0 for k in raw_dissim_by_unit}
    else:
        baseline_norm = {k: float(v / maxv) for k, v in raw_dissim_by_unit.items()}

    # aligned lists in unit order
    baseline_norm_list = [baseline_norm.get(u, 0.0) for u in units_flat]

    # ----------------------------
    # Doc relevance per unit (normalized doc score, inherited by units)
    # ----------------------------
    unit_doc_rel_norm_list = [float(doc_scores_norm[unit_doc_idx[i]]) for i in range(len(units_flat))]
    unit_doc_rel_norm = {units_flat[i]: unit_doc_rel_norm_list[i] for i in range(len(units_flat))}

    # ----------------------------
    # Confidence drop per unit (normalized)
    # ΔC_i = max(0, C0 - mean(C_i))
    # ----------------------------
    conf_drop_raw_list: List[float] = []
    for u in units_flat:
        c_i = mean_pert_conf_by_unit.get(u, 0.0)
        conf_drop_raw_list.append(max(0.0, baseline_conf_for_math - float(c_i)))

    # min-max normalize across units
    conf_drop_norm_list = _minmax_norm(conf_drop_raw_list)
    unit_conf_drop_norm = {units_flat[i]: float(conf_drop_norm_list[i]) for i in range(len(units_flat))}

    # baseline sensitivity (for visibility)
    unit_resp_sens = {units_flat[i]: float(baseline_norm_list[i]) for i in range(len(units_flat))}

    # ----------------------------
    # Compute fusion variants
    # ----------------------------
    # Validate fusion method names (ignore unknown, but report in debug)
    allowed = {
        "multiplicative_gated_weighted_additive",
        "weighted_additive",
        "triple_multiplicative",
    }
    requested = [m for m in req.fusion_methods if m in allowed]

    fusion_scores_lists = _compute_fusions(
        unit_texts=units_flat,
        baseline_norm=baseline_norm_list,
        doc_rel_norm=unit_doc_rel_norm_list,
        conf_drop_norm=conf_drop_norm_list,
        fusion_methods=requested,
        fusion_lambda=req.fusion_lambda,
    )

    # method -> {unit_text: score}
    modified_token_importances: Dict[str, Dict[str, float]] = {}
    for method, vals in fusion_scores_lists.items():
        modified_token_importances[method] = {units_flat[i]: float(vals[i]) for i in range(len(units_flat))}

    if req.debug:
        # attach normalized baseline importance per unit in debug details
        for unit, info in debug_details.items():
            info["normalized_importance_baseline"] = baseline_norm.get(unit, 0.0)

        debug_details["_baseline"] = {
            "original_answer": original,
            "prompt": prompt,
            "baseline_sequence_confidence": float(base_seq_conf) if base_seq_conf is not None else None,
        }
        debug_details["_fusion"] = {
            "fusion_methods_requested": req.fusion_methods,
            "fusion_methods_used": requested,
            "fusion_lambda": req.fusion_lambda,
        }

    return CompareResponse(
        context_used=context_used,
        prompt=prompt,
        original_answer=original,
        baseline_token_importances=baseline_norm,
        modified_token_importances=modified_token_importances,
        unit_doc_relevance_norm=unit_doc_rel_norm,
        unit_confidence_drop_norm=unit_conf_drop_norm,
        unit_response_sensitivity_baseline=unit_resp_sens,
        retrieved_docs=retrieved_docs,
        doc_scores_raw=doc_scores_raw,
        doc_scores_norm=doc_scores_norm,
        baseline_sequence_confidence=float(base_seq_conf) if base_seq_conf is not None else None,
        baseline_token_texts=base_token_texts if req.return_token_logprobs else None,
        baseline_token_logprobs=base_token_logprobs if req.return_token_logprobs else None,
        details=debug_details if req.debug else None,
    )

