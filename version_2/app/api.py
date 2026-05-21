# app/api.py
import os

# Must be set before ANY library imports to prevent macOS OMP segfault
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = (
    "1"  # suppress sequential-GPU advisory
)

import warnings

warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

# Suppress HuggingFace logging-based advisories (these use logging, not warnings)
import logging as _std_logging

_std_logging.getLogger("transformers.tokenization_utils_base").setLevel(
    _std_logging.ERROR
)
_std_logging.getLogger("transformers.pipelines.base").setLevel(_std_logging.ERROR)

"""
FusionRAG-Ex API

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
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.datasets import DATASET_CONFIGS
from app.generator import (
    get_generator,
    generate_answer,
    GeneratorWrapper,
    make_qa_prompt,
)
from app import pipeline as rag_pipeline
from app.retriever import retrieve

# ---------------------------------------------------------------------------
# Available models — keep in sync with what your hardware can serve
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    {"id": "google/flan-t5-small", "label": "Flan-T5 Small (default, fast)"},
    {"id": "google/gemma-2b-it", "label": "Gemma 2b Instruct"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B Instruct"},
    # Qwen3-14B requires transformers >=4.51 and Python >=3.10.
    # Uncomment when environment is upgraded:
    # {"id": "Qwen/Qwen3-14B",                             "label": "Qwen3-14B (needs Python 3.10)"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5 14B Instruct"},
]


# Infer the active dataset from the current settings path at import time
def _infer_active_dataset() -> str:
    for key, cfg in DATASET_CONFIGS.items():
        if cfg["DOCUMENTS_PATH"] in settings.DOCUMENTS_PATH:
            return key
    return list(DATASET_CONFIGS.keys())[0]


_ACTIVE_DATASET: str = _infer_active_dataset()

# Lazy cache: model_name -> GeneratorWrapper (loaded on first request)
_GENERATOR_CACHE: Dict[str, GeneratorWrapper] = {}
_GENERATOR_CACHE_LOCK = __import__("threading").Lock()
_DATASET_LOCK = __import__("threading").Lock()


def _get_generator_for_model(model_name: str) -> GeneratorWrapper:
    """Return a cached GeneratorWrapper, loading it the first time it is requested.
    Evicts any stale cache entry that is missing methods added in a hot-reload
    (e.g. make_qa_prompt) so uvicorn --reload never serves old objects.
    """
    with _GENERATOR_CACHE_LOCK:
        if model_name not in _GENERATOR_CACHE:
            logger.info("Loading generator for model: %s", model_name)
            _GENERATOR_CACHE[model_name] = get_generator(model_name)
            logger.info("Generator loaded: %s", model_name)
    return _GENERATOR_CACHE[model_name]


from app.comparator import COMPARATORS

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Gold-data cache (built once at startup, and refreshed on dataset switch)
# ---------------------------------------------------------------------------
_GOLD_ANSWERS: Optional[Dict[str, str]] = None  # question_id -> answer
_GOLD_DOCS: Optional[Dict[str, List[str]]] = None  # base_id     -> [relevant_doc_ids]
_GOLD_DOC_TEXTS: Optional[Dict[str, str]] = (
    None  # doc_id      -> text  (relevant docs only)
)
_GOLD_QUESTION_LOOKUP: Optional[Dict[str, str]] = (
    None  # norm_question_text -> question_id
)


def _norm_question(q: str) -> str:
    """Normalise a question string for lookup: lowercase, collapse whitespace."""
    return " ".join(q.lower().split())


def _build_answer_cache() -> None:
    """Load *_answers.jsonl and *_docs.jsonl for the active dataset into in-memory dicts."""
    global _GOLD_ANSWERS, _GOLD_DOCS, _GOLD_DOC_TEXTS, _GOLD_QUESTION_LOOKUP

    docs_path = settings.DOCUMENTS_PATH
    docs_dir = os.path.dirname(os.path.abspath(docs_path))
    basename = os.path.basename(docs_path)
    # Derive answers filename: replace trailing _docs.jsonl -> _answers.jsonl
    answers_basename = basename.replace("_docs.jsonl", "_answers.jsonl")
    answers_path = os.path.join(docs_dir, answers_basename)

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
        logger.info(
            "Gold answers loaded: %d entries, %d unique questions",
            len(_GOLD_ANSWERS),
            len(_GOLD_QUESTION_LOOKUP),
        )
    else:
        logger.warning("Answers file not found at %s", answers_path)

    # --- docs (relevant = no '_irr_' in id) ---
    _GOLD_DOCS = {}  # base_id -> [doc_ids that are relevant]
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
                    m = re.match(r"^(.+)_\d+$", doc_id)
                    if not m:
                        continue
                    base = m.group(1)
                    _GOLD_DOCS.setdefault(base, []).append(doc_id)
                    _GOLD_DOC_TEXTS[doc_id] = item.get("text", "")
                except Exception:
                    pass
        logger.info(
            "Gold doc groups loaded: %d base-IDs, %d relevant docs",
            len(_GOLD_DOCS),
            len(_GOLD_DOC_TEXTS),
        )
    else:
        logger.warning("Docs file not found at %s", docs_path)


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
    m = re.match(r"^(.+)_\d+$", question_id)
    base_id = m.group(1) if m else question_id
    gold_doc_ids: List[str] = _GOLD_DOCS.get(base_id, [])
    n_gold = len(gold_doc_ids)

    # Count how many gold doc IDs appear in the retrieved set
    retrieved_ids: set = set()
    for item in retrieved_docs or []:
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
            {"id": gid, "text": _GOLD_DOC_TEXTS.get(gid, "")} for gid in gold_doc_ids
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
    title="FusionRAG-Ex API",
    default_response_class=ORJSONResponse,
    version="1.1.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=False,
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
    context: str = Field(
        "",
        description="Optional context. If empty, retriever is used to build context.",
    )

    # retrieval
    retriever: str = Field(
        default=getattr(settings, "RETRIEVER_DEFAULT", "dense"),
        description="Retriever to use when context is empty (RAG mode): dense | bm25 | hybrid",
    )
    top_k_docs: int = Field(
        3, ge=1, le=50, description="Number of docs to retrieve when context is empty"
    )

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
    temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Generation temperature (0 = deterministic)"
    )

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
            "Fusion weight for 'confidence_retrieval_fusion' only. "
            "α=1.0 emphasizes retrieval relevance; α=0.0 emphasizes "
            "generation confidence drop; both are multiplied by softmax-normalised "
            "perturbation dissimilarity before the final softmax."
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
    raw_dissimilarity: float  # how much the answer changes when this unit is removed
    confidence_drop: (
        float  # how much generation confidence falls when this unit is removed
    )
    retrieval_weight: float  # softmax-normalised retrieval score of the source document
    baseline_importance: float  # normalised importance (ragex_core)
    ret_weighted_importance: float  # normalised importance (retrieval_weighted)
    fusion_importance: float  # normalised importance (confidence_retrieval_fusion)
    baseline_rank: int  # rank position in ragex_core (1 = highest)
    ret_weighted_rank: int  # rank position in retrieval_weighted
    fusion_rank: int  # rank position in confidence_retrieval_fusion


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


class CounterfactualResult(BaseModel):
    """Result of a single counterfactual re-ranking probe."""

    # 'already_correct' | 'rank_matters' | 'retriever_failure'
    diagnosis: str
    # 1-indexed rank of the retrieved doc that contained the gold answer (None = not found)
    gold_found_in_doc_rank: Optional[int] = None
    gold_doc_score: Optional[float] = None
    # Answer produced after boosting the gold doc to rank #1
    counterfactual_answer: Optional[str] = None
    counterfactual_matches_gold: Optional[bool] = None
    counterfactual_similarity: Optional[float] = None
    original_similarity: Optional[float] = None


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
    counterfactual: Optional[CounterfactualResult] = None


# ----------------------------
# Startup / basic routes
# ----------------------------
@app.on_event("startup")
def startup_event():
    # Pre-load the default (smallest) model so the first request is fast.
    _get_generator_for_model(settings.HF_MODEL)
    logger.info("Default generator initialized: %s", settings.HF_MODEL)
    _build_answer_cache()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><title>FusionRAG-Ex</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto;">
        <h2>FusionRAG-Ex API is running ✅</h2>
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


@app.get("/datasets")
def list_datasets():
    """Return the list of available datasets and which one is currently active."""
    return {
        "active": _ACTIVE_DATASET,
        "active_label": DATASET_CONFIGS[_ACTIVE_DATASET]["label"],
        "available": [
            {"id": k, "label": v["label"]} for k, v in DATASET_CONFIGS.items()
        ],
    }


class SwitchDatasetRequest(BaseModel):
    dataset: str = Field(..., description="Dataset key, e.g. 'hotpot' or 'trivia'")


@app.post("/switch-dataset")
def switch_dataset(req: SwitchDatasetRequest):
    """Switch the active dataset.  Mutates settings paths and rebuilds the gold cache."""
    global _ACTIVE_DATASET
    from app.retrievers import bm25 as _bm25_mod

    key = req.dataset
    if key not in DATASET_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{key}'. Available: {list(DATASET_CONFIGS.keys())}",
        )

    if key == _ACTIVE_DATASET:
        return {
            "active": _ACTIVE_DATASET,
            "label": DATASET_CONFIGS[_ACTIVE_DATASET]["label"],
            "changed": False,
        }

    cfg = DATASET_CONFIGS[key]
    with _DATASET_LOCK:
        settings.DOCUMENTS_PATH = cfg["DOCUMENTS_PATH"]
        settings.FAISS_INDEX_PATH = cfg["FAISS_INDEX_PATH"]
        settings.BM25_INDEX_PATH = cfg["BM25_INDEX_PATH"]

        _bm25_mod.invalidate_cache()
        from app.retrievers import dense_faiss as _faiss_mod

        _faiss_mod.invalidate_cache()

        _build_answer_cache()
        _ACTIVE_DATASET = key
    logger.info("Dataset switched to '%s' (%s)", key, cfg["label"])

    return {
        "active": _ACTIVE_DATASET,
        "label": cfg["label"],
        "changed": True,
    }


# ----------------------------
# API endpoints
# ----------------------------
@app.post("/retrieve", response_model=List[RetrieveResponseItem])
def api_retrieve(
    q: str = Query(..., description="Query string"),
    k: int = Query(3, ge=1, le=50, description="Top-k documents"),
    retriever_name: str = Query(
        "dense", description="Retriever: dense | bm25 | hybrid | multi_query"
    ),
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
# /retrieve/compare — run multiple retrievers side-by-side (Explorer)
# ---------------------------------------------------------------------------


class RetrieverCompareRequest(BaseModel):
    question: str = Field(..., description="Query to retrieve for")
    retrievers: List[str] = Field(
        ["dense", "bm25", "hybrid"],
        # "multi_query" temporarily disabled — uncomment to re-enable
        description="Which retrievers to compare",
    )
    top_k: int = Field(5, ge=1, le=20, description="Top-k per retriever")
    model: Optional[str] = None


class RetrieverCompareResponse(BaseModel):
    question: str
    results: Dict[str, Any]  # {retriever_name: {docs, queries_used?}}


@app.post("/retrieve/compare", response_model=RetrieverCompareResponse)
def retrieve_compare(req: RetrieverCompareRequest):
    """
    Run the same query through multiple retrievers and return results from each
    so the UI can diff which documents were found, what scores they got, and
    (for multi_query) which generated sub-queries were used.
    """
    output: Dict[str, Any] = {}
    for rname in req.retrievers:
        try:
            kw: Dict[str, Any] = {"retriever": rname}
            # if rname == "multi_query" and gen is not None:
            #     kw["generator"] = gen
            docs = retrieve(req.question, req.top_k, **kw)  # type: ignore
            # Extract queries_used from meta (attached by query_rewriter)
            queries_used: Optional[List[str]] = None
            for d in docs or []:
                if isinstance(d, dict):
                    _meta = d.get("meta") or (d.get("doc") or {}).get("meta") or {}
                    _qu = _meta.get("queries_used") if isinstance(_meta, dict) else None
                    if _qu:
                        queries_used = _qu
                        break
            output[rname] = {"docs": docs, "queries_used": queries_used}
        except Exception as e:
            output[rname] = {"error": str(e), "docs": [], "queries_used": None}

    return RetrieverCompareResponse(question=req.question, results=output)


def _low_confidence_explain_response(
    *,
    context_used: str,
    question: str,
    retrieved_docs: Optional[List[Dict[str, Any]]],
    max_score: float,
    threshold: float,
    debug: bool,
) -> ExplainResponse:
    return ExplainResponse(
        context_used=context_used,
        prompt=f"Context: {context_used}\nQuestion: {question}\nAnswer:",
        original_answer="Sorry, I am not confident to answer your question.",
        token_importances={},
        retrieved_docs=retrieved_docs,
        details={
            "confidence_check": {
                "max_retrieval_score": max_score,
                "threshold": threshold,
                "reason": "Retrieved documents have low relevance scores",
            }
        }
        if debug
        else None,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    model_id = (req.model or settings.HF_MODEL).strip()
    # Validate against allowed list
    allowed_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in allowed_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_id}'. Choose from: {sorted(allowed_ids)}",
        )
    try:
        gen = _get_generator_for_model(model_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not load model '{model_id}': {exc}"
        ) from exc
    if gen is None:
        raise HTTPException(
            status_code=503, detail="Generator not ready. Try again in a moment."
        )

    comparator_fn = COMPARATORS.get(req.comparator)
    if comparator_fn is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown comparator '{req.comparator}'"
        )

    with _DATASET_LOCK:
        try:
            context_used, retrieved_docs = rag_pipeline.build_rag_context(
                req.question,
                context=req.context,
                top_k_docs=req.top_k_docs,
                retriever=req.retriever,
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No context provided AND retriever index/doc store missing. "
                    "Create data/docs.jsonl and run index building."
                ),
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"RAG retrieval failed: {e}"
            ) from e

    if retrieved_docs:
        max_score = rag_pipeline.max_retrieval_score(retrieved_docs)
        if max_score < req.confidence_threshold:
            return _low_confidence_explain_response(
                context_used=context_used,
                question=req.question,
                retrieved_docs=retrieved_docs,
                max_score=max_score,
                threshold=req.confidence_threshold,
                debug=req.debug,
            )

    if req.importance_mode not in rag_pipeline.VALID_IMPORTANCE_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown importance_mode '{req.importance_mode}'. "
                f"Choose from: {sorted(rag_pipeline.VALID_IMPORTANCE_MODES)}"
            ),
        )

    doc_weights = rag_pipeline.softmax_retrieval_weights(retrieved_docs or [])
    needs_confidence = req.importance_mode == "confidence_retrieval_fusion"
    prompt = make_qa_prompt(gen, context_used, req.question)

    if needs_confidence:
        original, c0 = generate_answer(
            gen,
            prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            return_confidence=True,
        )
        c0 = c0 if c0 is not None else 1.0
    else:
        original = generate_answer(
            gen, prompt, max_length=req.max_length, temperature=req.temperature
        )
        c0 = 1.0

    try:
        units = rag_pipeline.split_explanation_units(
            context_used, req.explanation_level
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid explanation_level: {e}"
        ) from e

    signals = rag_pipeline.run_perturbation_loop(
        gen=gen,
        make_prompt=lambda ctx, q: make_qa_prompt(gen, ctx, q),
        context_used=context_used,
        question=req.question,
        units=units,
        perturber=req.perturber,
        comparator_fn=comparator_fn,
        original=original,
        c0=c0,
        max_length=req.max_length,
        temperature=req.temperature,
        collect_confidence=needs_confidence,
        debug=req.debug,
        debug_max_perturbations=req.debug_max_perturbations,
    )

    normalized, raw_rw, raw_fusion = rag_pipeline.compute_importance(
        mode=req.importance_mode,
        units=units,
        raw_dissimilarities=signals.raw_dissimilarities,
        confidence_drops=signals.confidence_drops,
        doc_weights=doc_weights,
        alpha=req.alpha,
    )

    debug_details = signals.debug_details
    if req.debug:
        for unit, info in debug_details.items():
            if unit.startswith("_"):
                continue
            info["normalized_importance"] = normalized.get(unit, 0.0)
            if req.importance_mode in (
                "retrieval_weighted",
                "confidence_retrieval_fusion",
            ):
                info["retrieval_weight"] = rag_pipeline.unit_retrieval_weight(
                    unit, doc_weights
                )
            if req.importance_mode == "retrieval_weighted":
                info["computed_ret_weighted_score"] = raw_rw.get(unit, 0.0)
            if req.importance_mode == "confidence_retrieval_fusion":
                info["computed_fusion_score"] = raw_fusion.get(unit, 0.0)
        debug_details["_baseline"] = {
            "original_answer": original,
            "prompt": prompt,
            "importance_mode": req.importance_mode,
            "alpha": req.alpha
            if req.importance_mode == "confidence_retrieval_fusion"
            else None,
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
        raise HTTPException(
            status_code=503, detail=f"Could not load model '{model_id}': {exc}"
        ) from exc

    comparator_fn = COMPARATORS.get(req.comparator)
    if comparator_fn is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown comparator '{req.comparator}'"
        )

    with _DATASET_LOCK:
        try:
            context_used, retrieved_docs = rag_pipeline.build_rag_context(
                req.question,
                context=req.context,
                top_k_docs=req.top_k_docs,
                retriever=req.retriever,
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail="Retriever index/doc store missing. Build indices first.",
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"RAG retrieval failed: {e}"
            ) from e

    if retrieved_docs:
        max_score = rag_pipeline.max_retrieval_score(retrieved_docs)
        if max_score < req.confidence_threshold:
            return CompareResponse(
                context_used=context_used,
                prompt=f"Context: {context_used}\nQuestion: {req.question}\nAnswer:",
                original_answer="Sorry, I am not confident to answer your question.",
                retrieved_docs=retrieved_docs,
                alpha=req.alpha,
                units=[],
            )

    doc_weights = rag_pipeline.softmax_retrieval_weights(retrieved_docs or [])
    prompt = make_qa_prompt(gen, context_used, req.question)
    original, c0 = generate_answer(
        gen,
        prompt,
        max_length=req.max_length,
        temperature=req.temperature,
        return_confidence=True,
    )
    c0 = c0 if c0 is not None else 1.0

    try:
        units = rag_pipeline.split_explanation_units(
            context_used, req.explanation_level
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid explanation_level: {e}"
        ) from e

    signals = rag_pipeline.run_perturbation_loop(
        gen=gen,
        make_prompt=lambda ctx, q: make_qa_prompt(gen, ctx, q),
        context_used=context_used,
        question=req.question,
        units=units,
        perturber=req.perturber,
        comparator_fn=comparator_fn,
        original=original,
        c0=c0,
        max_length=req.max_length,
        temperature=req.temperature,
        collect_confidence=True,
    )

    baseline_imp, ret_weighted_imp, fusion_imp = (
        rag_pipeline.compute_all_mode_importances(
            units=units,
            raw_dissimilarities=signals.raw_dissimilarities,
            confidence_drops=signals.confidence_drops,
            doc_weights=doc_weights,
            alpha=req.alpha,
        )
    )

    def _ranks(d: Dict[str, float]) -> Dict[str, int]:
        return {
            u: i + 1 for i, u in enumerate(sorted(d, key=lambda x: d[x], reverse=True))
        }

    base_ranks = _ranks(baseline_imp)
    rw_ranks = _ranks(ret_weighted_imp)
    fus_ranks = _ranks(fusion_imp)

    unit_signals: List[UnitSignals] = [
        UnitSignals(
            unit=u,
            raw_dissimilarity=signals.raw_dissimilarities[u],
            confidence_drop=signals.confidence_drops[u],
            retrieval_weight=rag_pipeline.unit_retrieval_weight(u, doc_weights),
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

    # ── Counterfactual re-ranking experiment ─────────────────────────────────
    # Logic:
    #   1. If the original answer already matches gold → diagnosis = 'already_correct'
    #   2. If wrong but gold text found in a retrieved doc → re-run generator with
    #      that doc promoted to rank #1 → diagnosis = 'rank_matters'
    #   3. If wrong and gold not found in any retrieved doc → diagnosis = 'retriever_failure'
    counterfactual: Optional[CounterfactualResult] = None
    gold_answer_text = gold.get("gold_answer")
    if gold_answer_text and retrieved_docs:
        gold_tokens = set(gold_answer_text.lower().split())

        def _tok_overlap(text: str) -> float:
            if not gold_tokens:
                return 0.0
            return len(gold_tokens & set(text.lower().split())) / len(gold_tokens)

        orig_sim = _tok_overlap(original)

        # Find the first retrieved doc that contains most of the gold answer tokens
        gold_doc_rank: Optional[int] = None
        gold_doc_score_cf: Optional[float] = None
        for _rank_i, _item in enumerate(retrieved_docs):
            if isinstance(_item, dict) and "doc" in _item:
                if _tok_overlap(_item["doc"].get("text", "")) >= 0.5:
                    gold_doc_rank = _rank_i + 1
                    gold_doc_score_cf = float(_item.get("score", 0.0))
                    break

        if orig_sim >= 0.5:
            counterfactual = CounterfactualResult(
                diagnosis="already_correct",
                gold_found_in_doc_rank=gold_doc_rank,
                gold_doc_score=gold_doc_score_cf,
                original_similarity=orig_sim,
            )
        elif gold_doc_rank is None:
            counterfactual = CounterfactualResult(
                diagnosis="retriever_failure",
                original_similarity=orig_sim,
            )
        else:
            # Re-rank: promote gold doc to position 0, keep rest in original order
            _gidx = gold_doc_rank - 1
            _reranked = [retrieved_docs[_gidx]] + [
                itm for _i, itm in enumerate(retrieved_docs) if _i != _gidx
            ]
            _cf_ctx = "\n".join(
                [
                    rag_pipeline.cap_doc_text(itm["doc"]["text"])
                    for itm in _reranked
                    if isinstance(itm, dict) and "doc" in itm and "text" in itm["doc"]
                ]
            ).strip()
            _cf_prompt = make_qa_prompt(gen, _cf_ctx, req.question)
            _cf_answer = generate_answer(
                gen, _cf_prompt, max_length=req.max_length, temperature=req.temperature
            )
            _cf_sim = _tok_overlap(_cf_answer)
            counterfactual = CounterfactualResult(
                diagnosis="rank_matters",
                gold_found_in_doc_rank=gold_doc_rank,
                gold_doc_score=gold_doc_score_cf,
                counterfactual_answer=_cf_answer,
                counterfactual_matches_gold=_cf_sim >= 0.5,
                counterfactual_similarity=_cf_sim,
                original_similarity=orig_sim,
            )

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
        counterfactual=counterfactual,
    )
