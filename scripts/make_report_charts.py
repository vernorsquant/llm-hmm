"""Generate the README summary charts: score compression + all-models Sharpe."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment.config import FIG_DIR
from hmm_sentiment.labeler_scoring import load_labeler_cache, _text_key
from hmm_sentiment.news_corpus import load_corpus

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def chart_score_compression(cache, news, path):
    """Bar chart: how directional is each scorer? mean |avg score| + % directional."""
    scorers = [
        ("llama-3.1-8b", "openrouter", "meta-llama/llama-3.1-8b-instruct", "#17becf"),
        ("deepseek-v4.1", "openrouter", "deepseek/deepseek-v4.1-flash-1017", "#ff7f0e"),
        ("glm-5.3-flash", "glm", "glm-5.3-flash", "#d62728"),
    ]
    names, mean_abs, pct_dir, colors = [], [], [], []
    for label, be, mid, color in scorers:
        vals = []
        for h in news["headline"]:
            f = cache["final"].get(_text_key(str(h), be, mid))
            if f and f.get("scores"):
                vals.append(np.mean(f["scores"]))
        v = np.array(vals)
        names.append(label)
        mean_abs.append(float(np.abs(v).mean()))
        pct_dir.append(100.0 * float(np.mean(np.abs(v) >= 1.5)))
        colors.append(color)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    bars = ax.bar(names, mean_abs, color=colors, alpha=0.9)
    for b, v in zip(bars, mean_abs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(1.5, color="gray", ls="--", lw=1)
    ax.text(2.45, 1.55, "directional cut (±1.5)", fontsize=8, color="gray", ha="right")
    ax.set_ylabel("mean |avg rubric score|  (0–5)")
    ax.set_title("Score compression: big models hedge toward neutral")
    ax.grid(alpha=0.3, axis="y")

    bars2 = ax2.bar(names, pct_dir, color=colors, alpha=0.9)
    for b, v in zip(bars2, pct_dir):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% of articles scored directional")
    ax2.set_title("% articles with |avg| ≥ 1.5")
    ax2.set_ylim(0, 85)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Why bigger models hurt: they compress scores toward neutral (n = 2,117 articles)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def chart_sharpe_all(path):
    """One master chart: OOS switch Sharpe for every configuration tested."""
    rows = [
        ("M3 FinBERT sentiment", 2.04, "#2ca02c", "best overall"),
        ("M13 Hybrid FinBERT+8b", 1.96, "#9467bd", "between parents"),
        ("M15 DeepSeek PCA twin", 2.00, "#f2b134", ""),
        ("M9 Labeler 8b", 1.87, "#17becf", "best labeler"),
        ("M1 Price-only HMM", 1.82, "#7f7f7f", ""),
        ("Buy & Hold SPY", 1.79, "black", ""),
        ("M14 FinBERT+DeepSeek", 1.71, "#ff7f0e", ""),
        ("M12 DeepSeek labeler", 1.58, "#d62728", "worst labeler"),
        ("M12 GLM labeler", 1.53, "#8b0000", "worst labeler"),
    ]
    rows_sorted = sorted(rows, key=lambda r: r[1])
    names = [r[0] for r in rows_sorted]
    vals = [r[1] for r in rows_sorted]
    colors = [r[2] for r in rows_sorted]
    notes = [r[3] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, vals, color=colors, alpha=0.92)
    ax.axvline(1.79, color="black", ls="--", lw=1.2)
    ax.text(1.795, len(names) - 0.4, "B&H 1.79", fontsize=9, va="top")
    for bar, v, note in zip(bars, vals, notes):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}" + (f"  ({note})" if note else ""), va="center", fontsize=9.5)
    ax.set_xlim(0, 2.45)
    ax.set_xlabel("OOS walk-forward switch Sharpe, 2023, net of 5 bps")
    ax.set_title("Every configuration tested — nothing significantly beats FinBERT sentiment\n"
                 "(all pairwise differences n.s.; MDE 0.32–1.88 at n = 250 days)", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    news = load_corpus()
    cache = load_labeler_cache()
    print(chart_score_compression(cache, news, f"{OUT_DIR}/labeler_score_compression.png"))
    print(chart_sharpe_all(f"{OUT_DIR}/labeler_sharpe_all_models.png"))
    print("CHARTS_DONE")
