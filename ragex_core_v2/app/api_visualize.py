# # app/api.py with visualization endpoint
# """
# RAG-Ex Core API with visualization endpoint

# Additions:
# - POST /visualize : returns an HTML page with inline heatmap highlighting of context units
#   and an embedded bar chart suitable for figures.
# - A small helper render_heatmap_html(...) that generates the HTML and bar chart PNG.
# """

# import base64
# import io
# import logging
# import textwrap
# from typing import Any, Dict, List, Optional

# from fastapi import FastAPI, HTTPException, Query
# from fastapi.responses import HTMLResponse, ORJSONResponse
# from pydantic import BaseModel, Field

# from app.config import settings
# from app.generator import get_generator, generate_answer
# from app.retriever import retrieve
# from app.perturb import perturb_sentence, split_context
# from app.comparator import COMPARATORS
# from app.experiments.interpretation import run_interpretation
# import torch
# torch.cuda.set_device(0)  # if you set CUDA_VISIBLE_DEVICES=1, then "0" means that selected GPU


# # Matplotlib for generating figure-quality bar charts
# import matplotlib
# # Use a non-interactive backend
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# logger = logging.getLogger("uvicorn.error")

# app = FastAPI(
#     title="RAG-Ex Core API (with visualize)",
#     default_response_class=ORJSONResponse,
#     version="1.1.2",
# )


# # ----------------------------
# # Models (same as prior)
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
#         description="Retriever to use when context is empty (RAG mode): dense | bm25",
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
#     prompt: str
#     context_used: str
#     original_answer: str
#     token_importances: Dict[str, float]

#     # ✅ NEW: bucketed units
#     most_important_tokens: List[str]
#     important_tokens: List[str]
#     least_important_tokens: List[str]

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
#         <p>Use Swagger UI to test endpoints.</p>
#         <ul>
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
# # Core endpoints (unchanged)
# # ----------------------------
# @app.post("/retrieve", response_model=List[RetrieveResponseItem])
# def api_retrieve(
#     q: str = Query(..., description="Query string"),
#     k: int = Query(3, ge=1, le=50, description="Top-k documents"),
#     retriever_name: str = Query("dense", description="Retriever: dense | bm25"),
# ):
#     try:
#         return retrieve(q, k, retriever=retriever_name)  # type: ignore[arg-type]
#     except TypeError:
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

# def bucket_importances(
#     importances: Dict[str, float],
#     method: str = "quantile",
#     q_hi: float = 0.80,
#     q_mid: float = 0.50,
#     hi_thr: float = 0.66,
#     mid_thr: float = 0.33,
# ) -> Dict[str, List[str]]:
#     """
#     Return 3 buckets of units based on normalized importance in [0,1].

#     method="quantile":
#       - most_important: >= q_hi quantile
#       - important:      >= q_mid quantile and < q_hi quantile
#       - least:          < q_mid quantile

#     method="threshold":
#       - most_important: >= hi_thr
#       - important:      >= mid_thr and < hi_thr
#       - least:          < mid_thr
#     """
#     if not importances:
#         return {"most": [], "important": [], "least": []}

#     items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
#     values = [v for _, v in items]

#     def quantile(vals: List[float], q: float) -> float:
#         # simple deterministic quantile without numpy
#         if not vals:
#             return 0.0
#         q = max(0.0, min(1.0, float(q)))
#         idx = int(round((len(vals) - 1) * q))
#         idx = max(0, min(len(vals) - 1, idx))
#         return float(sorted(vals)[idx])

#     if method == "quantile":
#         hi_cut = quantile(values, q_hi)
#         mid_cut = quantile(values, q_mid)
#     elif method == "threshold":
#         hi_cut = float(hi_thr)
#         mid_cut = float(mid_thr)
#     else:
#         raise ValueError("bucket_importances method must be 'quantile' or 'threshold'")

#     most: List[str] = []
#     important: List[str] = []
#     least: List[str] = []

#     for k, v in items:
#         if v >= hi_cut:
#             most.append(k)
#         elif v >= mid_cut:
#             important.append(k)
#         else:
#             least.append(k)

#     return {"most": most, "important": important, "least": least}


# @app.post("/explain", response_model=ExplainResponse)
# def explain(req: ExplainRequest):
#     gen = getattr(app.state, "generator", None)
#     if gen is None:
#         raise HTTPException(status_code=503, detail="Generator not ready. Try again in a moment.")

#     comparator_fn = COMPARATORS.get(req.comparator)
#     if comparator_fn is None:
#         raise HTTPException(status_code=400, detail=f"Unknown comparator '{req.comparator}'")

#     # Build context (direct or RAG)
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
#                     detail="Retriever returned no usable text. Ensure docs have 'text' fields.",
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

#     # Baseline generation
#     prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
#     original = generate_answer(app.state.generator, prompt, max_length=req.max_length, temperature=req.temperature)

#     # Split into units
#     try:
#         units = split_context(context_used, req.explanation_level)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

#     if not units:
#         units = [context_used]

#     # Perturbation loop -> compute raw dissimilarities and (optionally) debug details
#     raw_dissimilarities: Dict[str, float] = {}
#     debug_details: Dict[str, Any] = {} if req.debug else {}

#     for i, unit in enumerate(units):
#         perturbs = perturb_sentence(unit, req.perturber)
#         if not perturbs:
#             perturbs = [""]

#         sims: List[float] = []
#         samples: List[Dict[str, Any]] = []

#         for p in perturbs:
#             new_units = units.copy()
#             new_units[i] = p
#             new_ctx = " ".join([u for u in new_units if u is not None])

#             new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
#             pert_answer = generate_answer(app.state.generator, new_prompt, max_length=req.max_length, temperature=req.temperature)

#             sim = float(COMPARATORS.get(req.comparator, COMPARATORS["semantic"])(pert_answer, original))
#             sims.append(sim)

#             if req.debug and len(samples) < req.debug_max_perturbations:
#                 samples.append(
#                     {
#                         "perturbed_unit": p,
#                         "prompt": new_prompt,  # include the actual prompt used
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

#     # Normalize importances
#     maxv = max(raw_dissimilarities.values()) if raw_dissimilarities else 1.0
#     if maxv <= 0:
#         normalized = {k: 0.0 for k in raw_dissimilarities}
#     else:
#         normalized = {k: float(v / maxv) for k, v in raw_dissimilarities.items()}
    
#     buckets = bucket_importances(normalized, method="quantile", q_hi=0.80, q_mid=0.50)

#     if req.debug:
#         for unit, info in debug_details.items():
#             info["normalized_importance"] = normalized.get(unit, 0.0)
#         debug_details["_baseline"] = {"original_answer": original, "prompt": prompt}

#     return ExplainResponse(
#         prompt=prompt,
#         context_used=context_used,
#         original_answer=original,
#         token_importances=normalized,

#         # ✅ NEW
#         most_important_tokens=buckets["most"],
#         important_tokens=buckets["important"],
#         least_important_tokens=buckets["least"],

#         retrieved_docs=retrieved_docs,
#         details=debug_details if req.debug else None,
#     )

# class InterpretationRequest(BaseModel):
#     question: str = Field(..., min_length=1)
#     context: str = Field("", description="If empty, retrieval is used to build context.")

#     retriever: str = Field(default=getattr(settings, "RETRIEVER_DEFAULT", "dense"))
#     top_k_docs: int = Field(3, ge=1, le=50)

#     explanation_level: str = Field("sentence")
#     perturber: str = Field("leave_one_out")
#     comparator: str = Field("semantic")

#     max_length: int = Field(256, ge=16, le=2048)
#     temperature: float = Field(0.0, ge=0.0, le=2.0)
#     importance_mode: str = Field("ragex_core", description="ragex_core | modified_ragex")
#     alpha: float = Field(0.5, ge=0.0, le=1.0)
#     use_relevance_weight: bool = Field(True)

#     # experiment controls
#     k_values: List[str] = Field(default_factory=lambda: ["top-1", "top-3", "top-20%"])
#     random_repeats: int = Field(50, ge=1, le=500)
#     semantic_threshold_t: float = Field(0.85, ge=0.0, le=1.0)

#     # keep question fixed except DEL_Q
#     perturb_only_context: bool = Field(True)

#     debug: bool = Field(False)


# @app.post("/interpretation")
# def interpretation(req: InterpretationRequest):
#     gen = getattr(app.state, "generator", None)
#     if gen is None:
#         raise HTTPException(status_code=503, detail="Generator not ready. Try again in a moment.")

#     # Build context (direct or RAG) – same logic as /explain
#     context_used = (req.context or "").strip()
#     retrieved_docs: Optional[List[Dict[str, Any]]] = None

#     if not context_used:
#         try:
#             retrieved_docs = retrieve(req.question, req.top_k_docs, retriever=req.retriever)  # type: ignore[arg-type]
#         except TypeError:
#             retrieved_docs = retrieve(req.question, req.top_k_docs)  # type: ignore[misc]

#         context_used = "\n".join(
#             [
#                 item["doc"]["text"]
#                 for item in (retrieved_docs or [])
#                 if isinstance(item, dict)
#                 and "doc" in item
#                 and isinstance(item["doc"], dict)
#                 and "text" in item["doc"]
#             ]
#         ).strip()

#         if not context_used:
#             raise HTTPException(status_code=500, detail="Retriever returned no usable text for context.")
#     doc_scores = [float(d.get("score", 1.0)) for d in (retrieved_docs or [])]

#     out = run_interpretation(
#         generator=gen,
#         question=req.question,
#         context=context_used,
#         importance_mode= req.importance_mode,
#         alpha=req.alpha,
#         use_relevance_weight=req.use_relevance_weight,
#         doc_scores=doc_scores if retrieved_docs is not None else None,
#         explanation_level=req.explanation_level,
#         perturber=req.perturber,
#         comparator=req.comparator,
#         max_length=req.max_length,
#         temperature=req.temperature,
#         k_values=req.k_values,
#         random_repeats=req.random_repeats,
#         semantic_threshold_t=req.semantic_threshold_t,
#         perturb_only_context=req.perturb_only_context,
#         debug=req.debug,
#     )

#     # Attach retrieval info (optional)
#     if retrieved_docs is not None:
#         out["retrieved_docs"] = retrieved_docs

#     out["context_used"] = context_used
#     return out


# # ----------------------------
# # Visualization helper & endpoint
# # ----------------------------
# def _importance_to_color(val: float) -> str:
#     """
#     Map [0,1] importance to a background color. We use a simple red intensity mapping:
#     0.0 -> transparent/very light
#     1.0 -> strong red
#     Returns an inline CSS background-color string (rgba).
#     """
#     # clamp
#     v = max(0.0, min(1.0, float(val)))
#     # map to alpha 0.05 .. 0.9 for visibility
#     alpha = 0.05 + 0.85 * v
#     # red color
#     return f"rgba(220,30,30,{alpha:.3f})"

# def _bucket_to_color(bucket: str) -> str:
#     # discrete colors
#     if bucket == "most":
#         return "rgba(220,30,30,0.90)"
#     if bucket == "important":
#         return "rgba(220,30,30,0.45)"
#     return "rgba(220,30,30,0.08)"

# def bucket_labels_for_units(
#     units: List[str],
#     importances: Dict[str, float],
#     method: str = "quantile",
#     q_hi: float = 0.80,
#     q_mid: float = 0.50,
#     hi_thr: float = 0.66,
#     mid_thr: float = 0.33,
# ) -> Dict[str, str]:
#     """
#     Returns {unit: bucket_label} where bucket_label in {"most", "important", "least"}.
#     """
#     buckets = bucket_importances(
#         importances,
#         method=method,
#         q_hi=q_hi,
#         q_mid=q_mid,
#         hi_thr=hi_thr,
#         mid_thr=mid_thr,
#     )
#     most_set = set(buckets["most"])
#     important_set = set(buckets["important"])

#     labels: Dict[str, str] = {}
#     for u in units:
#         if u in most_set:
#             labels[u] = "most"
#         elif u in important_set:
#             labels[u] = "important"
#         else:
#             labels[u] = "least"
#     return labels


# def render_heatmap_html(context: str, units: List[str], importances: Dict[str, float], bar_chart_b64: Optional[str] = None, buckets: Optional[Dict[str, List[str]]] = None) -> str:
#     """
#     Produce an HTML fragment showing:
#     - inline highlighted context (units highlighted by importance)
#     - an embedded bar chart image (base64 PNG) if provided
#     - a small legend and export instructions
#     """
#     # Build highlighted inline HTML by iterating units and using importance lookup
#     spans: List[str] = []
#     most_set = set((buckets or {}).get("most", []))
#     imp_set = set((buckets or {}).get("important", []))

#     for u in units:
#         if buckets is not None:
#             if u in most_set:
#                 color = _bucket_to_color("most")
#             elif u in imp_set:
#                 color = _bucket_to_color("important")
#             else:
#                 color = _bucket_to_color("least")
#         else:
#             imp = importances.get(u, 0.0)
#             color = _importance_to_color(imp)
#         # escape HTML lightly
#         safe_unit = (
#             u.replace("&", "&amp;")
#             .replace("<", "&lt;")
#             .replace(">", "&gt;")
#         )
#         span = f'<span style="background-color:{color};padding:2px;border-radius:4px;margin-right:3px;display:inline-block">{safe_unit}</span>'
#         spans.append(span)

#     highlighted = " ".join(spans)

#     # Build bar chart area (if provided)
#     chart_html = ""
#     if bar_chart_b64:
#         chart_html = f'<h3>Importance bar chart (for paper)</h3><img src="data:image/png;base64,{bar_chart_b64}" alt="importance chart" style="max-width:100%;height:auto;border:1px solid #ddd;padding:6px;background:#fff" />'

#     html = f"""
#     <html>
#     <head>
#       <title>RAG-Ex Visualization</title>
#       <meta charset="utf-8" />
#       <style>
#         body {{ font-family: Arial, sans-serif; padding: 24px; max-width: 1100px; margin: auto; }}
#         h1 {{ font-size: 22px; }}
#         .context-box {{ padding: 14px; border-radius: 8px; background: #f7f7f9; border: 1px solid #eee; }}
#         .legend {{ margin-top: 12px; font-size: 13px; color: #333; }}
#         .unit {{ display:inline-block; margin:2px 4px; }}
#         .note {{ color: #666; font-size: 13px; margin-top:8px; }}
#       </style>
#     </head>
#     <body>
#       <h1>RAG-Ex: Inline importance heatmap</h1>
#       <p class="note">Question-driven explanation. Importance is normalized across units to [0,1].</p>

#       <h3>Context (highlighted by importance)</h3>
#       <div class="context-box">{highlighted}</div>

#       <div class="legend">
#         <p><strong>Legend:</strong> stronger red = higher importance. Units are shown according to the chosen granularity.</p>
#       </div>

#       {chart_html}

#       <div style="margin-top:18px; font-size:13px; color:#444;">
#         <p><strong>Notes:</strong></p>
#         <ul>
#           <li>This visualization is behavioral: importance measures how much perturbing the unit changes the generated answer.</li>
#           <li>Use the embedded bar chart for figure-quality export (PNG). You can right-click & save the image.</li>
#         </ul>
#       </div>
#     </body>
#     </html>
#     """
#     return html


# def _make_bar_chart_base64(
#     units: List[str],
#     importances: Dict[str, float],
#     bucket_method: str = "quantile",   # "quantile" or "threshold"
#     q_hi: float = 0.80,
#     q_mid: float = 0.50,) -> str:
#     """
#     Make a horizontal bar chart using matplotlib, return base64-encoded PNG.
#     Bars are colored by bucket and annotated with numeric importance values.
#     """

#     names = units[:]
#     values = [float(importances.get(u, 0.0)) for u in names]

#     # Determine bucket labels for coloring
#     labels = bucket_labels_for_units(
#         units=names,
#         importances=importances,
#         method=bucket_method,
#         q_hi=q_hi,
#         q_mid=q_mid,
#     )

#     # ✅ Colors requested by you
#     color_map = {
#         "most": "#2ecc71",      # green
#         "important": "#9ad0f5",  # light blue
#         "least": "#f5a3a3",      # light red
#     }
#     colors = [color_map.get(labels.get(u, "least"), "#f5a3a3") for u in names]

#     # Figure size depending on number of units
#     h = max(2.5, 0.45 * len(names))
#     fig, ax = plt.subplots(figsize=(7.2, h))

#     # Reverse for top-down display
#     names_r = names[::-1]
#     values_r = values[::-1]
#     colors_r = colors[::-1]

#     y_pos = list(range(len(names_r)))

#     bars = ax.barh(y_pos, values_r, color=colors_r)

#     # Y tick labels (truncate)
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(
#         [(n[:80] + ("..." if len(n) > 80 else "")) for n in names_r]
#     )

#     ax.set_xlabel("Normalized importance (0..1)")
#     ax.set_xlim(0, 1.0)
#     ax.grid(axis="x", linestyle="--", alpha=0.3)

#     # ✅ Add numeric labels on bars
#     # Place slightly to the right of bar end; clamp near edge for readability.
#     for bar, v in zip(bars, values_r):
#         x = min(v + 0.02, 0.98)  # avoid going outside plot
#         y = bar.get_y() + bar.get_height() / 2
#         ax.text(x, y, f"{v:.2f}", va="center", ha="left", fontsize=10)

#     # Optional legend
#     from matplotlib.patches import Patch
#     legend_handles = [
#         Patch(facecolor=color_map["most"], label="Most important"),
#         Patch(facecolor=color_map["important"], label="Important"),
#         Patch(facecolor=color_map["least"], label="Least important"),
#     ]
#     ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)

#     plt.tight_layout()

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode("ascii")



# @app.post("/visualize", response_class=HTMLResponse)
# def visualize(req: ExplainRequest):
#     """
#     Run the explanation (same as /explain) and return an HTML page with:
#       - inline heatmap of units
#       - embedded bar chart for figure export
#     This endpoint is convenient for interactive inspection and for saving PNGs for papers.
#     """
#     # Reuse explain logic by calling the same steps inline (keeps minimal changes)
#     # We avoid calling the endpoint function directly to keep types and exceptions explicit.

#     gen = getattr(app.state, "generator", None)
#     if gen is None:
#         raise HTTPException(status_code=503, detail="Generator not ready. Try again in a moment.")

#     comparator_fn = COMPARATORS.get(req.comparator)
#     if comparator_fn is None:
#         raise HTTPException(status_code=400, detail=f"Unknown comparator '{req.comparator}'")

#     # Build context
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
#                 raise HTTPException(status_code=500, detail="Retriever returned no usable text.")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e

#     # Baseline generation
#     prompt = f"Context: {context_used}\nQuestion: {req.question}\nAnswer:"
#     original = generate_answer(app.state.generator, prompt, max_length=req.max_length, temperature=req.temperature)

#     # Split into units
#     try:
#         units = split_context(context_used, req.explanation_level)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid explanation_level: {e}") from e

#     if not units:
#         units = [context_used]

#     # Compute raw dissimilarities (same logic as /explain)
#     raw_dissimilarities: Dict[str, float] = {}
#     for i, unit in enumerate(units):
#         perturbs = perturb_sentence(unit, req.perturber)
#         if not perturbs:
#             perturbs = [""]

#         sims: List[float] = []
#         for p in perturbs:
#             new_units = units.copy()
#             new_units[i] = p
#             new_ctx = " ".join([u for u in new_units if u is not None])

#             new_prompt = f"Context: {new_ctx}\nQuestion: {req.question}\nAnswer:"
#             pert_answer = generate_answer(app.state.generator, new_prompt, max_length=req.max_length, temperature=req.temperature)

#             sim = float(COMPARATORS.get(req.comparator, COMPARATORS["semantic"])(pert_answer, original))
#             sims.append(sim)

#         mean_sim = sum(sims) / max(1, len(sims))
#         raw_dissim = 1.0 - mean_sim
#         raw_dissimilarities[unit] = float(raw_dissim)

#     # Normalize
#     maxv = max(raw_dissimilarities.values()) if raw_dissimilarities else 1.0
#     if maxv <= 0:
#         normalized = {k: 0.0 for k in raw_dissimilarities}
#     else:
#         normalized = {k: float(v / maxv) for k, v in raw_dissimilarities.items()}

#     # make bar chart
#     try:
#         bar_b64 = _make_bar_chart_base64(
#         units,
#         normalized,
#         bucket_method="quantile",  # or "threshold"
#         q_hi=0.80,
#         q_mid=0.50)

#     except Exception:
#         bar_b64 = None

#     buckets = bucket_importances(normalized, method="quantile", q_hi=0.80, q_mid=0.50)
#     html = render_heatmap_html(context_used, units, normalized, bar_chart_b64=bar_b64, buckets=buckets)

#     return HTMLResponse(content=html, status_code=200)
