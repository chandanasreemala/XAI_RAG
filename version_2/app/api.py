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

import json
import logging
import re
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as torch_F

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from app.config import settings
from app.generator import get_generator, generate_answer, GeneratorWrapper
from app.retriever import retrieve

# ---------------------------------------------------------------------------
# Available models — keep in sync with what your hardware can serve
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    {"id": settings.HF_MODEL,                         "label": "Gemma 2b Instruct (default)"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct",        "label": "Llama 3.1 8B Instruct"},
    {"id": "Qwen/Qwen3-14B",                           "label": "Qwen3-14B"},
]

# Lazy cache: model_name -> GeneratorWrapper (loaded on first request)
_GENERATOR_CACHE: Dict[str, GeneratorWrapper] = {}
_GENERATOR_CACHE_LOCK = __import__("threading").Lock()


def _get_generator_for_model(model_name: str) -> GeneratorWrapper:
    """Return a cached GeneratorWrapper, loading it the first time it is requested."""
    if model_name not in _GENERATOR_CACHE:
        with _GENERATOR_CACHE_LOCK:
            if model_name not in _GENERATOR_CACHE:  # double-checked locking
                logger.info("Loading generator for model: %s", model_name)
                _GENERATOR_CACHE[model_name] = get_generator(model_name)
                logger.info("Generator loaded: %s", model_name)
    return _GENERATOR_CACHE[model_name]
from app.perturb import perturb_sentence, split_context
from app.comparator import COMPARATORS

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Gold-data cache (built once at startup from hotpot JSONL files)
# ---------------------------------------------------------------------------
_GOLD_ANSWERS: Optional[Dict[str, str]] = None        # question_id -> answer
_GOLD_DOCS: Optional[Dict[str, List[str]]] = None     # base_id     -> [relevant_doc_ids]
_GOLD_DOC_TEXTS: Optional[Dict[str, str]] = None      # doc_id      -> text  (relevant docs only)
_GOLD_QUESTION_LOOKUP: Optional[Dict[str, str]] = None # norm_question_text -> question_id


def _norm_question(q: str) -> str:
    """Normalise a question string for lookup: lowercase, collapse whitespace."""
    return " ".join(q.lower().split())


def _build_hotpot_cache() -> None:
    """Load hotpot_answers.jsonl and hotpot_docs.jsonl into in-memory dicts."""
    global _GOLD_ANSWERS, _GOLD_DOCS, _GOLD_DOC_TEXTS, _GOLD_QUESTION_LOOKUP

    docs_dir = os.path.dirname(os.path.abspath(settings.DOCUMENTS_PATH))
    answers_path = os.path.join(docs_dir, "hotpot_answers.jsonl")
    docs_path = settings.DOCUMENTS_PATH

    # --- answers + question-text reverse index ---
    _GOLD_ANSWERS = {}
    _GOLD_QUESTION_LOOKUP = {}  # norm_question -> question_id (first seen)
    if os.path.exists(answers_path):
        with open(answers_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    qid = item.get("id", "")
                    ans = item.get("answer", "")
                    qtxt = item.get("question", "")
                    if qid and ans:
                        if qid not in _GOLD_ANSWERS:
                            _GOLD_ANSWERS[qid] = ans
                        nq = _norm_question(qtxt)
                        if nq and nq not in _GOLD_QUESTION_LOOKUP:
                            _GOLD_QUESTION_LOOKUP[nq] = qid
                except Exception:
                    pass
        logger.info("Gold answers loaded: %d entries, %d unique questions",
                    len(_GOLD_ANSWERS), len(_GOLD_QUESTION_LOOKUP))
    else:
        logger.warning("hotpot_answers.jsonl not found at %s", answers_path)

    # --- docs (relevant = no '_irr_' in id) ---
    _GOLD_DOCS = {}       # base_id -> [doc_ids that are relevant]
    _GOLD_DOC_TEXTS = {}  # doc_id  -> text  (relevant docs only)
    if os.path.exists(docs_path):
        with open(docs_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    doc_id = item.get("id", "")
                    if not doc_id or "_irr_" in doc_id:
                        continue
                    # base_id is everything before the last '_<digit(s)>' suffix
                    m = re.match(r'^(.+)_\d+$', doc_id)
                    if not m:
                        continue
                    base = m.group(1)
                    _GOLD_DOCS.setdefault(base, []).append(doc_id)
                    _GOLD_DOC_TEXTS[doc_id] = item.get("text", "")
                except Exception:
                    pass
        logger.info("Gold doc groups loaded: %d base-IDs, %d relevant docs",
                    len(_GOLD_DOCS), len(_GOLD_DOC_TEXTS))
    else:
        logger.warning("hotpot_docs.jsonl not found at %s", docs_path)


def _gold_info_for_question(
    question_text: str,
    retrieved_docs: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Auto-resolve question_text to a gold question_id, then return gold answer,
    gold docs with full text, and retriever recall."""
    if _GOLD_ANSWERS is None or _GOLD_DOCS is None or _GOLD_QUESTION_LOOKUP is None:
        return {}

    # Look up by normalised question text
    question_id = _GOLD_QUESTION_LOOKUP.get(_norm_question(question_text))
    if not question_id:
        return {}

    gold_answer = _GOLD_ANSWERS.get(question_id)

    # Derive base_id: strip trailing _<N>
    m = re.match(r'^(.+)_\d+$', question_id)
    base_id = m.group(1) if m else question_id
    gold_doc_ids: List[str] = _GOLD_DOCS.get(base_id, [])
    n_gold = len(gold_doc_ids)

    # Count how many gold doc IDs appear in the retrieved set
    retrieved_ids: set = set()
    for item in (retrieved_docs or []):
        if isinstance(item, dict) and "doc" in item:
            did = item["doc"].get("id", "")
            if did:
                retrieved_ids.add(did)

    n_retrieved_relevant = sum(1 for gid in gold_doc_ids if gid in retrieved_ids)
    recall = n_retrieved_relevant / n_gold if n_gold > 0 else None

    # Build gold_docs list with full text for each relevant doc
    gold_docs: List[Dict[str, Any]] = []
    if _GOLD_DOC_TEXTS is not None:
        gold_docs = [
            {"id": gid, "text": _GOLD_DOC_TEXTS.get(gid, "")}
            for gid in gold_doc_ids
        ]

    return {
        "gold_answer": gold_answer,
        "gold_doc_ids": gold_doc_ids,
        "gold_docs": gold_docs,
        "n_gold_docs": n_gold,
        "n_retrieved_relevant": n_retrieved_relevant,
        "retriever_recall": recall,
    }

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

    # model selection
    model: Optional[str] = Field(
        None,
        description="HuggingFace model ID to use for generation. Defaults to the server's HF_MODEL setting.",
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

    # Ground-truth fields (auto-resolved from question text against the dataset)
    gold_answer: Optional[str] = None
    gold_doc_ids: Optional[List[str]] = None
    gold_docs: Optional[List[Dict[str, Any]]] = None
    n_gold_docs: Optional[int] = None
    n_retrieved_relevant: Optional[int] = None
    retriever_recall: Optional[float] = None


# ---------------------------------------------------------------------------
# Cross-mode comparison models  (used by /compare endpoint)
# ---------------------------------------------------------------------------

class UnitSignals(BaseModel):
    """All signals for a single explanation unit, computed in one perturbation pass."""
    unit: str
    raw_dissimilarity: float          # how much the answer changes when this unit is removed
    confidence_drop: float            # how much generation confidence falls when this unit is removed
    retrieval_weight: float           # softmax-normalised retrieval score of the source document
    baseline_importance: float        # normalised importance (ragex_core)
    ret_weighted_importance: float    # normalised importance (retrieval_weighted)
    fusion_importance: float          # normalised importance (confidence_retrieval_fusion)
    baseline_rank: int                # rank position in ragex_core (1 = highest)
    ret_weighted_rank: int            # rank position in retrieval_weighted
    fusion_rank: int                  # rank position in confidence_retrieval_fusion


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = ""
    retriever: str = Field(default=getattr(settings, "RETRIEVER_DEFAULT", "dense"))
    top_k_docs: int = Field(3, ge=1, le=50)
    explanation_level: str = "sentence"
    perturber: str = "leave_one_out"
    comparator: str = "semantic"
    max_length: int = Field(256, ge=16, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    model: Optional[str] = None


class CompareResponse(BaseModel):
    context_used: str
    prompt: str
    original_answer: str
    retrieved_docs: Optional[List[RetrieveResponseItem]] = None
    gold_answer: Optional[str] = None
    gold_doc_ids: Optional[List[str]] = None
    gold_docs: Optional[List[Dict[str, Any]]] = None
    n_gold_docs: Optional[int] = None
    n_retrieved_relevant: Optional[int] = None
    retriever_recall: Optional[float] = None
    alpha: float
    units: List[UnitSignals]


# ----------------------------
# Startup / basic routes
# ----------------------------
@app.on_event("startup")
def startup_event():
    # Pre-load the default (smallest) model so the first request is fast.
    _get_generator_for_model(settings.HF_MODEL)
    logger.info("Default generator initialized: %s", settings.HF_MODEL)
    _build_hotpot_cache()


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
    ok = bool(_GENERATOR_CACHE)  # at least one model loaded
    return {"status": "ok" if ok else "not_ready"}


@app.get("/models")
def list_models():
    """Return the list of available models and which one is the default."""
    return {
        "default": settings.HF_MODEL,
        "loaded": list(_GENERATOR_CACHE.keys()),
        "available": AVAILABLE_MODELS,
    }


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

#! -----> Check doc weights are ranked or not, if not find the max weighted doc and return the max weight for that unit ---Not Done


def _unit_retrieval_weight(unit: str, doc_weights: Dict[str, float]) -> float:
    """
    Return the retrieval weight for the doc(s) containing `unit`.
    - If the unit matches multiple docs, return the highest weight among them.
    - If no doc contains the unit, fallback 1e-8
      (best available signal) rather than a near-zero constant.
    """
    if not doc_weights:
        return 1e-8

    unit_norm = re.escape(unit.strip())
    matched_weights = [
        weight
        for doc_text, weight in doc_weights.items()
        if re.search(unit_norm, re.sub(r'\s+', ' ', doc_text))
    ]

    if matched_weights:
        return max(matched_weights)

    # No exact match — fall back to the max weight across all docs
    return max(doc_weights.values())



# def _unit_retrieval_weight(unit: str, doc_weights: Dict[str, float]) -> float:
#     unit_norm = re.escape(unit.strip())
#     for doc_text, weight in doc_weights.items():
#         if re.search(unit_norm, re.sub(r'\s+', ' ', doc_text)):
#             return weight
#     return 0.0 + 1e-8 # small constant to prevent zeroing out importance

# def _unit_retrieval_weight(
#     unit: str,
#     doc_weights: Dict[str, float],
#     fallback: float = 1.0,
# ) -> float:
#     """
#     Return the retrieval weight of the document that contains `unit`.
#     Uses substring matching.  Falls back to `fallback` (default 1.0) so
#     'retrieval_weighted' and 'confidence_retrieval_fusion' gracefully degrade
#     to the baseline when no retrieval scores are found (e.g. direct-context mode).
#     """
#     for doc_text, weight in doc_weights.items():
#         if unit in doc_text:
#             return weight
#     return fallback


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    model_id = (req.model or settings.HF_MODEL).strip()
    # Validate against allowed list
    allowed_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in allowed_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'. Choose from: {sorted(allowed_ids)}")
    try:
        gen = _get_generator_for_model(model_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load model '{model_id}': {exc}") from exc
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

    def _softmax_normalise(scores_dict: Dict[str, float]) -> Dict[str, float]:
        """Softmax normalisation using torch.nn.functional.softmax."""
        keys = list(scores_dict.keys())
        vals = torch.tensor([scores_dict[k] for k in keys], dtype=torch.float32)
        softmax_vals = torch_F.softmax(vals, dim=0)
        return {k: float(softmax_vals[i].item()) for i, k in enumerate(keys)}

    # Intermediate scores kept for debug output
    _normed_our: Dict[str, float] = {}   # softmax(raw_dissimilarities) — used by retrieval_weighted
    _raw_fusion: Dict[str, float] = {}   # pre-softmax fusion scores — used by confidence_retrieval_fusion

    if req.importance_mode == "ragex_core":
        # ── Perturbation-based generator importance (baseline) ────────────
        # w_i = w'_i / max_j w'_j
        normalized = _normalise(raw_dissimilarities)

    #! --------> Change the normalisation to softmax in both ret weighted and fusion.----Done

    elif req.importance_mode == "retrieval_weighted":
        # ── Generator importance gated by retrieval relevance ─────────────
        # Multiplies each unit's generator importance by the softmax-normalised
        # retrieval relevance score of the document it originates from.
        # w_i = normalise( w_i^baseline * r̃_d(i) )
        # Degrades gracefully to baseline when no retrieval scores exist.
        _normed_our = _softmax_normalise(raw_dissimilarities)

        retrieval_weighted = {
            unit: _normed_our[unit] * _unit_retrieval_weight(unit, doc_weights)
            for unit in units
        }
        # normalized = _normalise(raw_retrieval_weighted)
        normalized = retrieval_weighted

    else:
        # ── Confidence–retrieval fusion ────────────────────────────────────
        # Fuses two generator-side signals (response dissimilarity + generation
        # confidence drop) via α, then gates the result by retrieval relevance.
        # What is alpha here? It's a fusion weight that balances the contribution(controls the relative weighting) of two generator-side signals:
        # which is the ideal alpha number to set as default? It depends on the specific use case and the relative importance of the two signals in your context. A common starting point is α=0.5, which gives equal weight to both response dissimilarity and confidence drop. However, you may want to experiment with different values (e.g., 0.3, 0.7) to see how it affects the importance scores in your specific application.
        # If I set alpha = 1 that means I am giving more weight to? If you set α=1.0, you are giving full weight to response dissimilarity (w'_i) and ignoring generation confidence drop (Δc_i) in the fusion. In this case, the importance scores will be based entirely on how much the answer changes when unit i is perturbed, without considering how much the model's confidence in its answer decreases. Conversely, if you set α=0.0, you would be giving full weight to generation confidence drop and ignoring response dissimilarity.
        # - Response dissimilarity (w'_i): How much the answer changes when unit i is perturbed. Higher means more important.
        # - Generation confidence drop (Δc_i): How much the model's confidence in its answer decreases when unit i is perturbed. Higher means more important.
        # The formula is:
        # w_i = normalise( [α·w'_i + (1−α)·Δc_i] · r̃_d(i) )
        for unit in units:
            dissim          = raw_dissimilarities.get(unit, 0.0)
            confidence_drop = confidence_drops.get(unit, 0.0)
            retrieval_rel   = _unit_retrieval_weight(unit, doc_weights)
            _raw_fusion[unit] = (req.alpha * dissim + (1.0 - req.alpha) * confidence_drop) * retrieval_rel
        normalized = _softmax_normalise(_raw_fusion)

    if req.debug:
        for unit, info in debug_details.items():
            info["normalized_importance"] = normalized.get(unit, 0.0)
            if req.importance_mode in ("retrieval_weighted", "confidence_retrieval_fusion"):
                info["retrieval_weight"] = _unit_retrieval_weight(unit, doc_weights)
            if req.importance_mode == "retrieval_weighted":
                # computed_ret_weighted_score = softmax(dissim)[unit] * retrieval_weight
                info["computed_ret_weighted_score"] = (
                    _normed_our.get(unit, 0.0) * _unit_retrieval_weight(unit, doc_weights)
                )
            if req.importance_mode == "confidence_retrieval_fusion":
                # computed_fusion_score = (α·dissim + (1−α)·conf_drop) · retrieval_weight  (pre-softmax)
                info["computed_fusion_score"] = _raw_fusion.get(unit, 0.0)
        debug_details["_baseline"] = {
            "original_answer": original,
            "prompt": prompt,
            "importance_mode": req.importance_mode,
            "alpha": req.alpha if req.importance_mode == "confidence_retrieval_fusion" else None,
            "baseline_confidence": c0 if needs_confidence else None,
        }

    # Gold-truth metrics — always auto-resolved from the question text
    gold: Dict[str, Any] = _gold_info_for_question(req.question, retrieved_docs)

    return ExplainResponse(
        context_used=context_used,
        prompt=prompt,
        original_answer=original,
        token_importances=normalized,
        retrieved_docs=retrieved_docs,
        details=debug_details if req.debug else None,
        gold_answer=gold.get("gold_answer"),
        gold_doc_ids=gold.get("gold_doc_ids"),
        gold_docs=gold.get("gold_docs"),
        n_gold_docs=gold.get("n_gold_docs"),
        n_retrieved_relevant=gold.get("n_retrieved_relevant"),
        retriever_recall=gold.get("retriever_recall"),
    )


# ---------------------------------------------------------------------------
# /compare  — run all three importance modes in one perturbation pass
# Returns per-unit raw signals + all three normalised importance scores + ranks,
# enabling side-by-side cross-mode comparison in the Debug View tab.
# ---------------------------------------------------------------------------

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """
    Single-pass cross-mode comparison.
    Retrieves context once, generates the baseline answer once, then runs the
    perturbation loop once (always collecting confidence drops so fusion can be
    computed), and derives all three importance modes from the same raw signals.
    """
    model_id = (req.model or settings.HF_MODEL).strip()
    allowed_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in allowed_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'.")
    try:
        gen = _get_generator_for_model(model_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load model '{model_id}': {exc}") from exc

    comparator_fn = COMPARATORS.get(req.comparator)
    if comparator_fn is None:
        raise HTTPException(status_code=400, detail=f"Unknown comparator '{req.comparator}'")

    # --- Context building ---------------------------------------------------
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    context_used = (req.context or "").strip()
    if not context_used:
        try:
            try:
                retrieved_docs = retrieve(req.question, req.top_k_docs, retriever=req.retriever)  # type: ignore
            except TypeError:
                retrieved_docs = retrieve(req.question, req.top_k_docs)  # type: ignore
            context_used = "\n".join([
                item["doc"]["text"]
                for item in (retrieved_docs or [])
                if isinstance(item, dict) and "doc" in item and "text" in item["doc"]
            ]).strip()
            if not context_used:
                raise HTTPException(status_code=500, detail="Retriever returned no usable text.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}") from e

    if retrieved_docs:
        max_score = max(
            (item.get("score", 0.0) for item in retrieved_docs if isinstance(item, dict)), default=0.0
        )
        if max_score < req.confidence_threshold:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Max retrieval score ({max_score:.3f}) is below the confidence threshold "
                    f"({req.confidence_threshold}). Lower the threshold or rephrase the question."
                ),
            )

    doc_weights = _softmax_retrieval_weights(retrieved_docs or [])

    # --- Baseline generation (always collect confidence for fusion) ----------
    prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
    original, c0 = generate_answer(
        gen, prompt, max_length=req.max_length, temperature=req.temperature, return_confidence=True
    )
    c0 = c0 if c0 is not None else 1.0

    # --- Split context -------------------------------------------------------
    try:
        units = split_context(context_used, req.explanation_level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e
    if not units:
        units = [context_used]

    # --- Single perturbation pass: collect raw_dissim + confidence_drop ------
    raw_dissimilarities: Dict[str, float] = {}
    confidence_drops: Dict[str, float] = {}
    is_full_ctx_perturber = (req.perturber == "leave_one_out")

    for i, unit in enumerate(units):
        perturbs = perturb_sentence(unit, req.perturber, full_text=context_used)
        if not perturbs:
            perturbs = ["" if is_full_ctx_perturber else " ".join(
                u for j, u in enumerate(units) if j != i
            )]

        sims: List[float] = []
        conf_vals: List[float] = []

        for p in perturbs:
            if is_full_ctx_perturber:
                new_ctx = p
            else:
                new_units = units.copy()
                new_units[i] = p
                new_ctx = " ".join([u for u in new_units if u is not None])
            new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
            pert_answer, pert_conf = generate_answer(
                gen, new_prompt, max_length=req.max_length,
                temperature=req.temperature, return_confidence=True,
            )
            sims.append(float(comparator_fn(pert_answer, original)))
            if pert_conf is not None:
                conf_vals.append(pert_conf)

        mean_sim = sum(sims) / max(1, len(sims))
        raw_dissimilarities[unit] = float(1.0 - mean_sim)
        mean_pert_conf = sum(conf_vals) / max(1, len(conf_vals)) if conf_vals else c0
        #! --------------> Change the formula and apply the modulus(absolute) to c0 - mean_pert_conf and other parts of the code as well. ---- Done
        confidence_drops[unit] = abs(c0 - mean_pert_conf)

    # --- Compute all three importance modes ----------------------------------
    def _normalise(d: Dict[str, float]) -> Dict[str, float]:
        vals = list(d.values())
        maxv = max(vals) if vals else 1.0
        if maxv <= 0:
            return {k: 0.0 for k in d}
        return {k: float(v / maxv) for k, v in d.items()}

    def _softmax_normalise_cmp(d: Dict[str, float]) -> Dict[str, float]:
        """Softmax normalisation using torch.nn.functional.softmax."""
        keys = list(d.keys())
        vals = torch.tensor([d[k] for k in keys], dtype=torch.float32)
        softmax_vals = torch_F.softmax(vals, dim=0)
        return {k: float(softmax_vals[i].item()) for i, k in enumerate(keys)}

    baseline_imp = _normalise(raw_dissimilarities)

    normed_our = _softmax_normalise_cmp(raw_dissimilarities)
    raw_rw = {u: normed_our[u] * _unit_retrieval_weight(u, doc_weights) for u in units}
    ret_weighted_imp = _softmax_normalise_cmp(raw_rw)

    raw_fusion = {
        u: (req.alpha * raw_dissimilarities[u] + (1.0 - req.alpha) * confidence_drops[u])
           * _unit_retrieval_weight(u, doc_weights)
        for u in units
    }
    fusion_imp = _softmax_normalise_cmp(raw_fusion)

    # --- Assign ranks (1 = most important) -----------------------------------
    def _ranks(d: Dict[str, float]) -> Dict[str, int]:
        return {u: i + 1 for i, u in enumerate(sorted(d, key=lambda x: d[x], reverse=True))}

    base_ranks = _ranks(baseline_imp)
    rw_ranks   = _ranks(ret_weighted_imp)
    fus_ranks  = _ranks(fusion_imp)

    unit_signals: List[UnitSignals] = [
        UnitSignals(
            unit=u,
            raw_dissimilarity=raw_dissimilarities[u],
            confidence_drop=confidence_drops[u],
            retrieval_weight=_unit_retrieval_weight(u, doc_weights),
            baseline_importance=baseline_imp[u],
            ret_weighted_importance=ret_weighted_imp[u],
            fusion_importance=fusion_imp[u],
            baseline_rank=base_ranks[u],
            ret_weighted_rank=rw_ranks[u],
            fusion_rank=fus_ranks[u],
        )
        for u in units
    ]

    gold = _gold_info_for_question(req.question, retrieved_docs)

    return CompareResponse(
        context_used=context_used,
        prompt=prompt,
        original_answer=original,
        retrieved_docs=retrieved_docs,
        gold_answer=gold.get("gold_answer"),
        gold_doc_ids=gold.get("gold_doc_ids"),
        gold_docs=gold.get("gold_docs"),
        n_gold_docs=gold.get("n_gold_docs"),
        n_retrieved_relevant=gold.get("n_retrieved_relevant"),
        retriever_recall=gold.get("retriever_recall"),
        alpha=req.alpha,
        units=unit_signals,
    )
