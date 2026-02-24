"""
Generate technical report PDF: End-to-End Importance Scoring for RAG Systems.
Usage: ragex/bin/python scripts/generate_report.py
Output: report_rag_importance.pdf
"""

import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.mathtext
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import PageBreak

W, H = A4
MARGIN = 2.2 * cm

# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a LaTeX math string → PNG bytes via matplotlib
# ─────────────────────────────────────────────────────────────────────────────
def math_to_image(latex: str, fontsize: int = 14, dpi: int = 180) -> bytes:
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.axis("off")
    t = ax.text(0.5, 0.5, f"${latex}$", fontsize=fontsize,
                ha="center", va="center", transform=ax.transAxes)
    # Auto-size figure around text
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    pad = 0.3
    fig.set_size_inches((bbox.width / dpi) + pad, (bbox.height / dpi) + pad)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def math_img(latex: str, fontsize: int = 14, width_cm: float = 14) -> Image:
    data = math_to_image(latex, fontsize=fontsize)
    img = Image(io.BytesIO(data))
    img.drawWidth  = width_cm * cm
    img.drawHeight = img.drawWidth * (img.imageHeight / img.imageWidth)
    img.hAlign = "CENTER"
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle("Title", parent=styles["Title"],
    fontSize=18, leading=24, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"))

subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
    fontSize=11, leading=14, spaceAfter=16, alignment=TA_CENTER,
    textColor=colors.HexColor("#444444"))

h1_style = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"), borderPad=0)

h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=12, leading=15, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor("#16213e"))

body_style = ParagraphStyle("Body", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)

caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
    fontSize=8.5, leading=11, spaceAfter=8, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique")

box_style = ParagraphStyle("Box", parent=styles["Normal"],
    fontSize=9.5, leading=14, spaceAfter=4, alignment=TA_LEFT,
    leftIndent=10, rightIndent=10)

def p(text, style=None): return Paragraph(text, style or body_style)
def sp(n=0.3):           return Spacer(1, n * cm)
def hr():                return HRFlowable(width="100%", thickness=0.5,
                                           color=colors.HexColor("#cccccc"),
                                           spaceAfter=6, spaceBefore=6)


def section_box(content_rows, bg=colors.HexColor("#f5f7ff"),
                border=colors.HexColor("#3a3a8c")):
    """Draw a lightly shaded box around a group of flowables."""
    tdata = [[c] for c in content_rows]
    t = Table(tdata, colWidths=[W - 2 * MARGIN - 0.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.8, border),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return t


def failure_table():
    data = [
        [p("<b>Pattern</b>", box_style), p("<b>Diagnosis</b>", box_style)],
        [p("w′ᵢ high,  r̃_d(i) low",  box_style),
         p("Generator relies on a low-ranked document → <b>Retriever failure</b>", box_style)],
        [p("w′ᵢ low,  r̃_d(i) high",  box_style),
         p("Generator ignores the top-ranked document → <b>Generator failure</b>", box_style)],
        [p("w′ᵢ high,  r̃_d(i) high", box_style),
         p("Retriever and generator agree → <b>Aligned (healthy)</b>", box_style)],
        [p("w′ᵢ low,  r̃_d(i) low",  box_style),
         p("Unit irrelevant to both modules → <b>Noise unit</b>", box_style)],
    ]
    cw = [(W - 2 * MARGIN - 0.4 * cm) * f for f in (0.38, 0.62)]
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#dde3f5")),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f9faff")),
        ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor("#3a3a8c")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaacc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


def scenario_table():
    data = [
        [p("<b>Scenario</b>", box_style),
         p("<b>Signals Used</b>", box_style),
         p("<b>What It Reveals</b>", box_style)],
        [p("A — Baseline", box_style),
         p("Response dissimilarity w′ᵢ", box_style),
         p("Generator reliance on each context unit", box_style)],
        [p("B — Retriever-Weighted", box_style),
         p("w′ᵢ × softmax retrieval score r̃_d(i)", box_style),
         p("Retriever–generator alignment and failure modes", box_style)],
        [p("C — Confidence Fusion", box_style),
         p("[α·w′ᵢ + (1−α)·Δcᵢ] × r̃_d(i)", box_style),
         p("Tri-modal: answer change + confidence drop + retrieval relevance", box_style)],
    ]
    cw = [(W - 2 * MARGIN - 0.4 * cm) * f for f in (0.20, 0.35, 0.45)]
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#dde3f5")),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f9faff")),
        ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor("#3a3a8c")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaacc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Build document
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "report_rag_importance.pdf"
)

doc = SimpleDocTemplate(
    out_path,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=2.5 * cm, bottomMargin=2.5 * cm,
)

story = []

# ── Title block ──────────────────────────────────────────────────────────────
story += [
    sp(0.5),
    p("End-to-End Importance Scoring for<br/>Retrieval-Augmented Generation Systems", title_style),
    p("A Technical Report on Generator-Side, Retriever-Weighted,<br/>and Confidence-Fused Explanation Frameworks", subtitle_style),
    hr(),
    sp(0.2),
]

# ── Abstract ─────────────────────────────────────────────────────────────────
story += [
    p("<b>Abstract</b>", h1_style),
    p(
        "Explainability methods for Retrieval-Augmented Generation (RAG) systems have "
        "predominantly focused on the generator module in isolation, measuring which parts "
        "of the provided context most influence the generated answer. This report formalises "
        "three complementary importance-scoring schemes of increasing expressiveness: "
        "(A) a baseline perturbation-based scheme faithful to the original RAG-Ex formulation, "
        "(B) a retriever-weighted extension that jointly scores context units by their "
        "generative importance and the retrieval relevance of the document they originate from, "
        "and (C) a confidence-fused three-way scheme that additionally incorporates the "
        "model's generation confidence as a signal of genuine reliance. Together, these "
        "schemes enable diagnosis of retriever failures, generator failures, and "
        "retriever–generator misalignment that purely generator-intrinsic methods cannot detect."
    ),
    sp(0.3),
    hr(),
]

# ── Section 1: Background ─────────────────────────────────────────────────────
story += [
    p("1. Background and Motivation", h1_style),
    p(
        "A standard RAG pipeline for open-domain question answering proceeds as follows. "
        "Given a user question <i>q</i>, a retriever selects the <i>K</i> most relevant "
        "documents {d₁, …, d_K} from a corpus. These documents are concatenated to form "
        "a context <i>C</i>, which together with the question constitutes the prompt fed to "
        "a large language model (LLM). The LLM then produces an answer <i>A₀</i>."
    ),
    p(
        "Existing post-hoc explanation methods for RAG treat the retrieved context as a "
        "monolithic input and measure only the generator's dependence on sub-units of that "
        "context. This ignores an important question in deployed systems: <i>are the units "
        "that the generator relies upon also the units that the retriever judged most relevant?</i> "
        "A gap between these two rankings reveals a structural failure in the RAG pipeline "
        "that purely generator-intrinsic explanations cannot surface."
    ),
    p(
        "This report introduces a unified formalism covering three scenarios of increasing "
        "expressiveness, each building on the previous one."
    ),
    sp(0.2),
]

# ── Section 2: Notation ───────────────────────────────────────────────────────
story += [
    p("2. Notation and Setup", h1_style),
    p("The following notation is used throughout:"),
    sp(0.1),
    section_box([
        p("<b>q</b>  — user question (fixed throughout all perturbations)", box_style),
        p("<b>C</b>  — context formed by concatenating the K retrieved documents", box_style),
        p("<b>{u₁, …, u_n}</b>  — context units obtained by splitting C at the chosen granularity (word / sentence / paragraph)", box_style),
        p("<b>A₀</b>  — baseline answer generated from the original prompt", box_style),
        p("<b>A_i^(j)</b>  — answer generated after applying the j-th perturbation to unit uᵢ", box_style),
        p("<b>sim(·,·)</b>  — comparator function ∈ [0, 1]; higher = more similar (e.g. SBERT cosine similarity)", box_style),
        p("<b>m</b>  — number of perturbations applied per unit", box_style),
        p("<b>r_d</b>  — raw retrieval relevance score of document d returned by the retriever", box_style),
        p("<b>c₀, c_i^(j)</b>  — generation confidence (exp of mean token log-probability) for A₀ and A_i^(j)", box_style),
        p("<b>α ∈ [0,1]</b>  — mixture parameter balancing response change vs confidence drop", box_style),
    ]),
    sp(0.3),
]

# ── Section 3: Baseline (Scenario A) ─────────────────────────────────────────
story += [
    p("3. Scenario A — Baseline Generator Importance", h1_style),
    p(
        "For each context unit uᵢ, a set of m perturbed inputs is constructed by applying "
        "a perturbation strategy (e.g. leave-one-out, entity replacement, synonym injection) "
        "to the unit within the full context text. Each perturbed context is paired with the "
        "original question to form a new prompt, from which a perturbed answer A_i^(j) is "
        "generated. The question is never perturbed."
    ),
    sp(0.15),
    p("<b>Step 1 — Baseline generation</b>", h2_style),
    p("Construct the baseline prompt using the context C and question q, then generate the reference answer:"),
    math_img(r"P_0 = [ \, C\;||\;q \, ], \quad A_0 = \mathrm{LLM}(P_0)", fontsize=13),
    sp(0.15),
    p("<b>Step 2 — Perturbation and re-generation</b>", h2_style),
    p("For each unit uᵢ and each perturbation variant j, form a perturbed context C_i^(j) and re-generate:"),
    math_img(r"P_i^{(j)} = [ \, C_i^{(j)}\;||\;q \, ], \quad A_i^{(j)} = \mathrm{LLM}(P_i^{(j)})", fontsize=13),
    p("where C_i^(j) is the full context with unit uᵢ replaced or removed according to perturbation j."),
    sp(0.15),
    p("<b>Step 3 — Raw dissimilarity</b>", h2_style),
    math_img(r"w'_i = 1 - \frac{1}{m}\sum_{j=1}^{m} \mathrm{sim}(A_i^{(j)}, \,  A_0)", fontsize=13),
    p(
        "Interpretation: if perturbing uᵢ produces answers that differ greatly from A₀, "
        "then w′ᵢ → 1 (unit is important). If the answer is unchanged, w′ᵢ → 0."
    ),
    sp(0.15),
    p("<b>Step 4 — Normalisation (paper formula)</b>", h2_style),
    math_img(r"w_i = \frac{w'_i}{\max_{j=1}^{n} w'_j}", fontsize=13),
    p(
        "The most important unit receives score 1.0; all others are scaled proportionally. "
        "The least important unit retains its non-zero relative score, correctly reflecting "
        "partial influence."
    ),
    sp(0.2),
]

# ── Section 4: Scenario B ─────────────────────────────────────────────────────
story += [
    p("4. Scenario B — Retriever-Weighted Importance", h1_style),
    p(
        "The baseline score wᵢ is a purely generator-side signal. It cannot distinguish "
        "whether the important unit came from a highly relevant or an irrelevant document. "
        "Scenario B multiplies the generator importance by the retrieval relevance of the "
        "source document, producing a joint score."
    ),
    sp(0.15),
    p("<b>Step 1 — Softmax normalisation of retrieval scores</b>", h2_style),
    p(
        "Let {r₁, …, r_K} be the raw retrieval scores of the K retrieved documents "
        "(e.g. cosine similarity from a dense retriever). Apply softmax:"
    ),
    math_img(r"\tilde{r}_d = \frac{e^{r_d}}{\sum_{k=1}^{K} e^{r_k}}", fontsize=13),
    p("This ensures retrieval weights sum to 1 and are strictly positive."),
    sp(0.15),
    p("<b>Step 2 — Unit-to-document mapping</b>", h2_style),
    p(
        "Each unit uᵢ is assigned to the document d(i) whose text contains it "
        "(determined by substring matching). The retrieval weight of that document "
        "is r̃_d(i)."
    ),
    sp(0.15),
    p("<b>Step 3 — Joint importance score</b>", h2_style),
    math_img(r"w_i^B = w_i \;\cdot\; \tilde{r}_{d(i)}", fontsize=13),
    math_img(r"\hat{w}_i^B = \frac{w_i^B}{\max_{j=1}^{n} w_j^B}", fontsize=13),
    sp(0.15),
    p("<b>Diagnostic failure modes exposed by Scenario B:</b>", h2_style),
    failure_table(),
    sp(0.2),
    p(
        "This scenario enables <i>retriever–generator alignment analysis</i>: a system is "
        "well-calibrated if the units with the highest ŵᵢᴮ scores are both heavily used by "
        "the generator and originate from highly relevant retrieved documents. Systematic "
        "misalignment indicates either retriever failures (low r̃, high w′) or generator "
        "failures (high r̃, low w′)."
    ),
    sp(0.2),
]

# ── Section 5: Scenario C ─────────────────────────────────────────────────────
story += [
    p("5. Scenario C — Confidence-Fused Tri-Modal Importance", h1_style),
    p(
        "Generation confidence provides a complementary signal to response dissimilarity. "
        "A unit for which the model's confidence drops sharply under perturbation was not "
        "merely statistically associated with the answer — the model was genuinely uncertain "
        "without it. Scenario C fuses three signals: the response change (w′ᵢ), the "
        "confidence drop (Δcᵢ), and the retrieval relevance (r̃_d(i))."
    ),
    sp(0.15),
    p("<b>Confidence measurement</b>", h2_style),
    p(
        "Generation confidence is defined as the exponential of the mean token "
        "log-probability over the generated sequence:"
    ),
    math_img(r"c = \exp(\frac{1}{T}\sum_{t=1}^{T} \log P(\hat{y}_t \mid \hat{y}_{<t}, \,  x)) \;\;\in (0, 1]", fontsize=12),
    p("where T is the number of generated tokens and x is the input prompt."),
    sp(0.15),
    p("<b>Step 1 — Confidence drop per unit</b>", h2_style),
    math_img(r"\Delta c_i = \max(0,\; c_0 - \frac{1}{m}\sum_{j=1}^{m} c_i^{(j)})", fontsize=13),
    p(
        "The max(0, ·) clipping ensures that a confidence increase under perturbation "
        "(meaning the unit was noise) contributes zero rather than a negative value. "
        "c₀ is the baseline confidence on A₀."
    ),
    sp(0.15),
    p("<b>Step 2 — Fused raw score</b>", h2_style),
    math_img(r"w_i^C = [\alpha \cdot w'_i \;+\; (1-\alpha)\cdot \Delta c_i] \;\cdot\; \tilde{r}_{d(i)}", fontsize=13),
    p(
        "The bracket fuses the two generator-side signals: response dissimilarity w′ᵢ "
        "and confidence drop Δcᵢ, controlled by the mixture parameter α ∈ [0, 1]. "
        "The result is then gated by the retrieval relevance r̃_d(i), which discounts "
        "importance scores for units from poorly-ranked documents."
    ),
    sp(0.1),
    section_box([
        p("<b>Parameter α interpretation:</b>", box_style),
        p("α = 1.0  →  Scenario C reduces to Scenario B (response-change only, retrieval-weighted)", box_style),
        p("α = 0.0  →  importance driven entirely by confidence drop, scaled by retrieval relevance", box_style),
        p("α = 0.5  →  equal weight to answer change and confidence drop (recommended default)", box_style),
    ], bg=colors.HexColor("#fff8e7"), border=colors.HexColor("#b8860b")),
    sp(0.15),
    p("<b>Step 3 — Normalisation</b>", h2_style),
    math_img(r"\hat{w}_i^C = \frac{w_i^C}{\max_{j=1}^{n} w_j^C}", fontsize=13),
    sp(0.2),
    p("<b>Research questions addressed by Scenario C:</b>"),
    p(
        "1. When the most important unit (high ŵᵢᶜ) is perturbed, does model confidence "
        "drop in proportion to the answer change, or does confidence remain high while the "
        "answer changes? The latter suggests the model is generating plausibly but not reliably."
    ),
    p(
        "2. Does the confidence-drop signal Δcᵢ correlate with the response-dissimilarity "
        "signal w′ᵢ? A low correlation would suggest these are complementary signals, "
        "justifying the fusion rather than using either alone."
    ),
    p(
        "3. Do units from high-relevance documents (high r̃_d(i)) show larger confidence "
        "drops under perturbation, validating the assumption that the retriever's ranking "
        "correlates with the generator's genuine reliance?"
    ),
    sp(0.2),
]

# ── Section 6: Comparison ─────────────────────────────────────────────────────
story += [
    p("6. Comparison of the Three Scenarios", h1_style),
    scenario_table(),
    sp(0.25),
    p(
        "The three scenarios form a hierarchy. Scenario A is a faithful implementation of "
        "the original perturbation-based RAG explanation. Scenario B extends it by "
        "incorporating the retriever's perspective, enabling joint retriever–generator "
        "analysis. Scenario C further adds generation confidence, providing the richest "
        "signal at the cost of an additional forward pass per perturbation."
    ),
    sp(0.2),
]

# ── Section 7: Notation Summary ───────────────────────────────────────────────
story += [
    p("7. Formula Reference", h1_style),

    p("<b>Scenario A</b>", h2_style),
    math_img(r"w'_i = 1 - \frac{1}{m}\sum_{j=1}^{m}\mathrm{sim}(A_i^{(j)}, A_0), \qquad w_i^A = \frac{w'_i}{\max_j w'_j}", fontsize=12, width_cm=15),

    p("<b>Scenario B</b>", h2_style),
    math_img(r"\tilde{r}_d = \frac{e^{r_d}}{\sum_k e^{r_k}}, \qquad w_i^B = w_i^A \cdot \tilde{r}_{d(i)}, \qquad \hat{w}_i^B = \frac{w_i^B}{\max_j w_j^B}", fontsize=12, width_cm=15),

    p("<b>Scenario C</b>", h2_style),
    math_img(r"\Delta c_i = \max(0,\; c_0 - \frac{1}{m}\sum_j c_i^{(j)})", fontsize=12, width_cm=13),
    math_img(r"w_i^C = [\alpha \cdot w'_i + (1-\alpha)\cdot\Delta c_i]\cdot\tilde{r}_{d(i)}, \qquad \hat{w}_i^C = \frac{w_i^C}{\max_j w_j^C}", fontsize=12, width_cm=15),

    sp(0.2),
]

# ── Section 8: Discussion ─────────────────────────────────────────────────────
story += [
    p("8. Discussion", h1_style),
    p(
        "<b>On the role of the question.</b> The question q is held fixed across all "
        "perturbations in all three scenarios. Only the context is modified. This is by "
        "design: the goal is to explain which parts of the retrieved context drove the "
        "answer to a specific question, not to measure question sensitivity."
    ),
    p(
        "<b>On the choice of perturbation strategy.</b> The leave-one-out strategy removes "
        "an entire unit from the full context string, operating at the unit level of the "
        "chosen granularity. Other strategies (entity replacement, synonym injection, "
        "antonym injection, reordering) modify the unit in place and reassemble the context. "
        "Leave-one-out is the most interpretable for Scenarios B and C because it provides "
        "a clean counterfactual: what happens if this unit disappears entirely?"
    ),
    p(
        "<b>On softmax vs. raw retrieval scores.</b> Softmax normalisation is preferred over "
        "min-max normalisation for retrieval scores because it preserves the relative "
        "magnitudes between documents and is differentiable, which is relevant if the "
        "framework is extended to gradient-based methods in future work."
    ),
    p(
        "<b>On the α parameter.</b> The mixture parameter α in Scenario C can be treated "
        "as a hyperparameter tuned on a held-out validation set, or kept at the default "
        "α = 0.5. An ablation over α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} is recommended to "
        "understand the relative contribution of the two generator-side signals."
    ),
    sp(0.2),
]

# ── Footer marker ─────────────────────────────────────────────────────────────
story += [
    hr(),
    p("End of Technical Report", caption_style),
]

doc.build(story)
print(f"PDF written to: {out_path}")
