"""Branch-Solve-Merge explainer figures (mechanics only, no editorial).

Data: real cached labels from outputs/tables/labeler_scores_sample.csv
Implementation reference: LLM-Labeler/simple_bsm.py, llm-hmm labeler_scoring.py
"""
import csv
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- style
C_BRANCH = "#3B82F6"   # blue
C_SOLVE = "#F59E0B"    # amber
C_MERGE = "#10B981"    # green
C_BULL = "#059669"
C_BEAR = "#DC2626"
C_NEUT = "#6B7280"
C_INK = "#1F2937"
C_PAPER = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": C_INK,
    "axes.edgecolor": C_INK,
    "figure.facecolor": C_PAPER,
    "axes.facecolor": C_PAPER,
    "savefig.facecolor": C_PAPER,
})


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.02):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0, zorder=2)
    ax.add_patch(b)
    return b


def arrow(ax, x0, y0, x1, y1, color=C_INK, lw=1.6, style="-|>", ms=14):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w))


def label_color(sent):
    return {"Bullish": C_BULL, "Bearish": C_BEAR, "Neutral": C_NEUT}[sent]


def score_chip_text(s):
    return f"{s:+d}"


# ------------------------------------------------- generic pipeline figure
def draw_pipeline(fname, title, headline, criteria, stage_note=True,
                  footer=False, mean_val=None, sent=None, figw=16.4, figh=8.6):
    """criteria: list of (name, score_or_None). mean_val/sent for examples."""
    n = len(criteria)
    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=17,
            fontweight="bold")

    # stage headers
    hy = 0.875
    if stage_note:
        ax.text(0.315, hy, "1 \u00b7 BRANCH  (plan)", ha="center", fontsize=13.5,
                fontweight="bold", color=C_BRANCH)
        ax.text(0.615, hy, "2 \u00b7 SOLVE  (score each criterion)", ha="center",
                fontsize=13.5, fontweight="bold", color=C_SOLVE)
        ax.text(0.885, hy, "3 \u00b7 MERGE  (average \u2192 label)", ha="center",
                fontsize=13.5, fontweight="bold", color=C_MERGE)

    # input box (left)
    ix, iy, iw, ih = 0.015, 0.40, 0.155, 0.30
    box(ax, ix, iy, iw, ih, "#EEF2FF", C_BRANCH)
    ax.text(ix + iw / 2, iy + ih - 0.045, "INPUT", ha="center", fontsize=11,
            fontweight="bold", color=C_BRANCH)
    ax.text(ix + iw / 2, iy + ih / 2 - 0.02, wrap(f'"{headline}"', 26),
            ha="center", va="center", fontsize=10.5, style="italic")
    ax.text(ix + iw / 2, iy + 0.045, "ticker: SPY \u00b7 temperature 0",
            ha="center", fontsize=8.5, color="#475569")

    # branch column: criteria boxes
    bx, bw = 0.235, 0.165
    bh = min(0.145, 0.62 / n - 0.015)
    gap = (0.72 - n * bh) / max(n - 1, 1) if n > 1 else 0
    y0 = 0.80 - bh
    cy = []
    for i, (name, _s) in enumerate(criteria):
        y = y0 - i * (bh + gap)
        cy.append(y + bh / 2)
        box(ax, bx, y, bw, bh, "#EFF6FF", C_BRANCH)
        ax.text(bx + bw / 2, y + bh - 0.022, f"criterion {i + 1}", ha="center",
                fontsize=9, fontweight="bold", color=C_BRANCH)
        ax.text(bx + bw / 2, y + (bh - 0.022) / 2, wrap(name, 30), ha="center",
                va="center", fontsize=9.5)
        arrow(ax, ix + iw, 0.55, bx, y + bh / 2, color=C_BRANCH, lw=1.2)
    # call-count annotation under branch column
    if stage_note:
        ax.text(bx + bw / 2, y0 - n * (bh + gap) + gap - 0.015,
                "1 LLM call \u2192 JSON list\nof {name, rubric} pairs",
                ha="center", va="top", fontsize=9.5, color="#475569")

    # solve column: score chips
    sx, sw = 0.545, 0.115
    for i, (name, s) in enumerate(criteria):
        y = cy[i]
        if s is None:
            txt, fc, ec = "score\n\u22125 \u2026 +5", "#FFFBEB", C_SOLVE
        else:
            txt = score_chip_text(s)
            fc = "#FEF3C7" if abs(s) <= 1 else (
                "#DCFCE7" if s > 0 else "#FEE2E2")
            ec = C_SOLVE
        box(ax, sx, y - 0.055, sw, 0.11, fc, ec)
        ax.text(sx + sw / 2, y, txt, ha="center", va="center",
                fontsize=15 if s is not None else 11, fontweight="bold")
        arrow(ax, bx + bw, y, sx, y, color=C_SOLVE, lw=1.2)
    if stage_note:
        ax.text(sx + sw / 2, y0 - n * (bh + gap) + gap - 0.015,
                "4 parallel LLM calls\neach sees ONLY its\ncriterion \u00b7 integer score\n+ rationale",
                ha="center", va="top", fontsize=9.5, color="#475569")

    # merge box
    mx, my, mw, mh = 0.815, 0.44, 0.16, 0.24
    box(ax, mx, my, mw, mh, "#ECFDF5", C_MERGE, lw=1.8)
    ax.text(mx + mw / 2, my + mh - 0.035, "MERGE", ha="center", fontsize=11,
            fontweight="bold", color=C_MERGE)
    if mean_val is None:
        ax.text(mx + mw / 2, my + mh / 2 - 0.02,
                "avg = mean(scores)\n\n\u2265 +1.5  \u2192 Bullish\n\u2264 \u22121.5  \u2192 Bearish\nelse     \u2192 Neutral",
                ha="center", va="center", fontsize=10.5)
    else:
        ax.text(mx + mw / 2, my + mh - 0.085, f"avg = {mean_val:+.2f}",
                ha="center", fontsize=16, fontweight="bold", color=C_MERGE)
        ax.text(mx + mw / 2, my + 0.06,
                "mean of branch scores\n(classified unrounded)",
                ha="center", va="center", fontsize=9.5, color="#475569")
    for y in cy:
        arrow(ax, sx + sw, y, mx, 0.56, color=C_MERGE, lw=1.1)

    # label chip
    ly = 0.175
    if sent is not None:
        lc = label_color(sent)
        box(ax, mx, ly, mw, 0.115, "white", lc, lw=2.2)
        ax.text(mx + mw / 2, ly + 0.0575, sent.upper(), ha="center",
                va="center", fontsize=15, fontweight="bold", color=lc)
        arrow(ax, mx + mw / 2, my, mx + mw / 2, ly + 0.115, color=lc, lw=1.8)
        ax.text(mx + mw / 2, ly - 0.035,
                {"Bullish": "avg \u2265 +1.5", "Bearish": "avg \u2264 \u22121.5",
                 "Neutral": "\u22121.5 < avg < +1.5"}[sent],
                ha="center", fontsize=9.5, color="#475569")
    else:
        ax.text(mx + mw / 2, ly + 0.06, "Bullish / Neutral / Bearish",
                ha="center", fontsize=11, color="#475569")

    if footer:
        ax.text(0.5, 0.012,
                "per-article outputs:  avg_impact \u2208 [\u22125, +5] \u00b7 S_raw = avg/5 \u2208 [\u22121, +1] \u00b7 "
                "confidence = 1 \u2212 std(scores)/5 \u2208 [0.25, 1]      \u2192  aggregated daily  \u2192  HMM input feature",
                ha="center", fontsize=9.5, color="#334155",
                bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#CBD5E1"))

    fig.savefig(os.path.join(OUT, fname), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------- fig 1
draw_pipeline(
    "bsm_01_flow.png",
    "Branch \u2192 Solve \u2192 Merge: one headline, one label",
    "news headline about the market / ticker",
    [("Criterion 1  (LLM also writes\nits own rubric)", None),
     ("Criterion 2", None),
     ("Criterion 3", None),
     ("Criterion 4", None)],
    footer=True,
)

# ------------------------------------------------- real examples from cache
bull = ("US Inflation Eases More Than Expected In October, Cements Expectations "
        "For End Of Interest Rate Hikes",
        [("Inflation Surprise Magnitude", 3), ("Fed Policy Expectations", 3),
         ("Equity Risk Premium & Valuation Sensitivity", 3),
         ("Risk Sentiment and Market Breadth", 4)], 3.25, "Bullish")
draw_pipeline("bsm_02_example_bullish.png",
              "Worked example \u00b7 real cached labels \u00b7 Bullish case",
              bull[0], bull[1], mean_val=bull[2], sent=bull[3])

bear = ("SVB Financial Realigns Portfolio -- and Blows Up the Banking Sector",
        [("Banking Sector Contagion", -4), ("Systemic Risk & Credit Conditions", -4),
         ("Market Sentiment and Volatility", -3),
         ("Policy Response & Safe-Haven Flows", -3)], -3.50, "Bearish")
draw_pipeline("bsm_03_example_bearish.png",
              "Worked example \u00b7 real cached labels \u00b7 Bearish case",
              bear[0], bear[1], mean_val=bear[2], sent=bear[3])

mixed = ("Tech stocks outperform Tuesday, Nasdaq Composite gains",
         [("Sector Rotation Impact", -1), ("Market Breadth Signal", -1),
          ("Investor Sentiment Spillover", 2),
          ("Index Composition Sensitivity", 2)], 0.50, "Neutral")
draw_pipeline("bsm_04_example_mixed.png",
              "Worked example \u00b7 real cached labels \u00b7 branches disagree, merge decides",
              mixed[0], mixed[1], mean_val=mixed[2], sent=mixed[3])

# fig 5: number line + example cards (sorted by mean so connectors never cross)
fig, ax = plt.subplots(figsize=(15.5, 7.2))
ax.set_xlim(-5.6, 5.6); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.97, "MERGE rule: average the branch scores, then threshold at \u00b11.5",
        ha="left", va="top", fontsize=17, fontweight="bold")

# number line with bands
yl = 0.70
bh = 0.10
ax.add_patch(plt.Rectangle((-5, yl), 3.5, bh, fc="#FEE2E2", ec="none"))
ax.add_patch(plt.Rectangle((-1.5, yl), 3.0, bh, fc="#E5E7EB", ec="none"))
ax.add_patch(plt.Rectangle((1.5, yl), 3.5, bh, fc="#D1FAE5", ec="none"))
ax.plot([-5.15, 5.15], [yl, yl], color=C_INK, lw=1.2)
for v in range(-5, 6):
    ax.plot([v, v], [yl - 0.018, yl + 0.018], color=C_INK, lw=1.2)
    ax.text(v, yl + bh + 0.012, f"{v:+d}", ha="center", va="bottom", fontsize=11)
for v in (-1.5, 1.5):
    ax.plot([v, v], [yl - 0.05, yl + bh + 0.04], color=C_INK, lw=1.6,
            linestyle=(0, (4, 3)))
ax.text(-3.25, yl + bh + 0.055, "Bearish  (avg \u2264 \u22121.5)", ha="center",
        fontsize=12.5, fontweight="bold", color=C_BEAR)
ax.text(0.0, yl + bh + 0.055, "Neutral  (\u22121.5 < avg < +1.5)", ha="center",
        fontsize=12.5, fontweight="bold", color=C_NEUT)
ax.text(3.25, yl + bh + 0.055, "Bullish  (avg \u2265 +1.5)", ha="center",
        fontsize=12.5, fontweight="bold", color=C_BULL)

examples = sorted([
    ([3, 3, 3, 4], "CPI eases (real)"),
    ([4, 4, 3, 3], "all-positive branches"),
    ([2, 1, 0, 0], "mildly positive"),
    ([-1, -1, 2, 2], "tech rally (real)"),
    ([-1, -1, -2, 2], "two-and-two split"),
    ([-4, -4, -3, -3], "SVB collapse (real)"),
], key=lambda e: sum(e[0]) / len(e[0]))

# diamonds on the line; value labels alternate depth so close means never collide
for i, (vec, name) in enumerate(examples):
    m = sum(vec) / len(vec)
    c = C_BEAR if m <= -1.5 else (C_BULL if m >= 1.5 else C_NEUT)
    ax.plot([m], [yl], marker="D", ms=10, color=c, zorder=3)
    depth = 0.055 + 0.055 * (i % 2)
    ax.text(m, yl - depth, f"{m:+.2f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c)
    ax.plot([m, m], [yl - depth + 0.02, yl - 0.012], color=c, lw=0.8, alpha=0.6)

# example cards below, evenly spaced left-to-right in the same order
cw, ch, cy0 = 1.72, 0.155, 0.30
step = 10.0 / len(examples)
for i, (vec, name) in enumerate(examples):
    m = sum(vec) / len(vec)
    c = C_BEAR if m <= -1.5 else (C_BULL if m >= 1.5 else C_NEUT)
    cx = -5 + step * (i + 0.5)
    box(ax, cx - cw / 2, cy0, cw, ch, "white", c, lw=1.8)
    vec_txt = "  ".join(f"{s:+d}" for s in vec)
    ax.text(cx, cy0 + ch - 0.038, wrap(name, 22), ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_INK)
    ax.text(cx, cy0 + 0.062, f"[ {vec_txt} ]", ha="center", va="center",
            fontsize=10.5, family="monospace")
    ax.text(cx, cy0 + 0.026, f"mean {m:+.2f}  \u2192  "
            f"{'Bearish' if m <= -1.5 else ('Bullish' if m >= 1.5 else 'Neutral')}",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=c)
    # connector card -> diamond
    ax.plot([cx, cx, m, m], [cy0 + ch + 0.01, yl - 0.24, yl - 0.24, yl - 0.05],
            color=c, lw=0.9, alpha=0.45, linestyle=(0, (2, 2)))

fig.text(0.5, 0.015,
         "score vectors \u2192 mean \u2192 label.   Real rows are cached labels from "
         "outputs/tables/labeler_scores_sample.csv (LLM-labeler, temperature 0).",
         ha="center", fontsize=10, color="#475569")
fig.savefig(os.path.join(OUT, "bsm_05_merge_rules.png"), dpi=160,
            bbox_inches="tight")
plt.close(fig)
print("wrote bsm_05_merge_rules.png")

# ---------------------------------------------------------------- fig 6
fig, ax = plt.subplots(figsize=(15.5, 7.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.97, "Call structure, caching, and failure handling",
        ha="center", va="top", fontsize=17, fontweight="bold")

# left panel: fan-out
box(ax, 0.03, 0.13, 0.30, 0.72, "#F8FAFC", "#CBD5E1")
ax.text(0.18, 0.80, "LLM calls per article", ha="center", fontsize=13,
        fontweight="bold")
steps = [("BRANCH (plan)", "1 call", C_BRANCH),
         ("SOLVE \u00d7 3\u20134 criteria", "3\u20134 calls", C_SOLVE),
         ("MERGE", "0 calls (arithmetic)", C_MERGE)]
sy = 0.66
for name, cnt, col in steps:
    box(ax, 0.055, sy, 0.25, 0.10, "white", col, lw=1.6)
    ax.text(0.07, sy + 0.05, name, ha="left", va="center", fontsize=10.5)
    ax.text(0.29, sy + 0.05, cnt, ha="right", va="center", fontsize=10.5,
            fontweight="bold", color=col)
    sy -= 0.135
ax.text(0.18, 0.30, "total  \u2248 4\u20135 calls / article", ha="center",
        fontsize=11.5, fontweight="bold")
ax.text(0.18, 0.215, "corpus: 2,117 articles  \u2192  \u224810k LLM calls",
        ha="center", fontsize=10.5, color="#475569")
ax.text(0.18, 0.165, "calls are thread-pooled (8 workers),\ntemperature 0, retries on 429/5xx",
        ha="center", fontsize=9.5, color="#475569")

# middle panel: cache anatomy
box(ax, 0.36, 0.13, 0.30, 0.72, "#F8FAFC", "#CBD5E1")
ax.text(0.51, 0.80, "Cache: one JSON record\nper LLM call", ha="center",
        va="top", fontsize=13, fontweight="bold")
ax.text(0.51, 0.665,
        "data/cache/llm_labeler_scores.jsonl\n(45,272 records on disk)",
        ha="center", fontsize=10, color="#475569")
rec = ('{"key": "sha1(headline)|backend|model|v2",\n '
       '"stage": "solve",\n '
       '"criterion": "Fed Policy Expectations",\n '
       '"score": 3,\n '
       '"rationale": "..."}')
box(ax, 0.385, 0.44, 0.25, 0.19, "white", "#94A3B8")
ax.text(0.51, 0.535, rec, ha="center", va="center", fontsize=9,
        family="monospace")
ax.text(0.51, 0.365, "appended per call \u00b7 crash-resumable \u00b7 loaded last-wins",
        ha="center", fontsize=10)
ax.text(0.51, 0.27, "key includes backend + model + pipeline version,\nso a model switch never mixes label spaces",
        ha="center", fontsize=9.5, color="#475569")
ax.text(0.51, 0.175, "plan / solve / final are cached separately:\na cached solve is reused even if the\nplan stage re-runs",
        ha="center", fontsize=9.5, color="#475569")

# right panel: failure handling
box(ax, 0.67, 0.13, 0.30, 0.72, "#F8FAFC", "#CBD5E1")
ax.text(0.82, 0.80, "Failure handling", ha="center", fontsize=13,
        fontweight="bold")
fails = [("plan unparseable \u00d72",
          "fixed 3-criterion\nFALLBACK_RUBRIC used;\nresult never cached", "#F59E0B"),
         ("solve unparseable \u00d72",
          "neutral (0) used in-memory\nonly; never cached;\narticle retried next run", "#EF4444"),
         ("degraded labels downstream",
          "surface as NaN, not 0,\nso coverage gates\nstay honest", "#6B7280")]
fy = 0.615
for trig, act, col in fails:
    box(ax, 0.69, fy - 0.015, 0.26, 0.145, "white", col, lw=1.4)
    ax.text(0.70, fy + 0.105, trig, ha="left", va="center", fontsize=10,
            fontweight="bold", color=col)
    ax.text(0.70, fy + 0.045, act, ha="left", va="center", fontsize=9)
    fy -= 0.175

fig.savefig(os.path.join(OUT, "bsm_06_calls_cache.png"), dpi=160,
            bbox_inches="tight")
plt.close(fig)
print("wrote bsm_06_calls_cache.png")

# ---------------------------------------------------------------- fig 7
rows = list(csv.DictReader(open(os.path.join(OUT, "..", "..", "outputs",
                                             "tables", "labeler_scores_sample.csv"))))
avgs = [float(r["avg_impact"]) for r in rows]
sents = [r["sentiment"] for r in rows]
crit_scores = []
for r in rows:
    for part in r["criteria_scores"].split(";"):
        if "=" in part:
            crit_scores.append(int(part.rsplit("=", 1)[1]))
ncrit = [len([p for p in r["criteria_scores"].split(";") if "=" in p]) for r in rows]

fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
fig.suptitle("What the pipeline emits on the real corpus (sample of 30 cached labels)",
             fontsize=16, fontweight="bold", y=1.00)

ax = axs[0]
bins = [-5 + 0.5 * i for i in range(21)]
ax.hist(avgs, bins=bins, color="#94A3B8", edgecolor="white")
for sp, c, lab in ((-1.5, C_BEAR, ""), (1.5, C_BULL, "")):
    ax.axvline(sp, color=c, lw=1.6, linestyle=(0, (4, 3)))
ax.axvspan(-5, -1.5, color="#FEE2E2", alpha=0.45, zorder=0)
ax.axvspan(1.5, 5, color="#D1FAE5", alpha=0.45, zorder=0)
ax.set_xlabel("avg_impact per article (\u22125 \u2026 +5)")
ax.set_ylabel("articles")
ax.set_title("merged article-level scores", fontsize=12)
ax.text(0.03, 0.95, "thresholds \u00b11.5", transform=ax.transAxes, fontsize=9.5,
        color="#475569", va="top")

ax = axs[1]
counts = {s: sents.count(s) for s in ("Bullish", "Neutral", "Bearish")}
ax.bar(counts.keys(), counts.values(),
       color=[label_color(s) for s in counts], width=0.62)
for k, v in counts.items():
    ax.text(k, v + 0.25, str(v), ha="center", fontsize=11, fontweight="bold")
ax.set_title("merged label counts", fontsize=12)
ax.set_ylabel("articles")

ax = axs[2]
vals, cnts = zip(*sorted((v, crit_scores.count(v)) for v in set(crit_scores)))
ax.bar([str(v) for v in vals], cnts, color="#F59E0B", width=0.62)
ax.set_title("individual branch (criterion) scores", fontsize=12)
ax.set_xlabel("score")
ax.set_ylabel("branches")

fig.text(0.5, 0.005,
         f"n = {len(rows)} articles \u00b7 {sum(ncrit)} branches \u00b7 "
         f"{sum(ncrit)/len(rows):.2f} branches/article \u00b7 "
         f"from outputs/tables/labeler_scores_sample.csv",
         ha="center", fontsize=10, color="#475569")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(os.path.join(OUT, "bsm_07_real_outputs.png"), dpi=160,
            bbox_inches="tight")
plt.close(fig)
print("wrote bsm_07_real_outputs.png")
print("done ->", OUT)
