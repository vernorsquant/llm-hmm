"""Publication-grade (300 DPI) figures for the 4-model benchmark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR
from .metrics import equity_curve

MODEL_LABELS = {
    "M1_price_hmm": "Model 1: Price HMM",
    "M2_price_pca_hmm": "Model 2: Price PCA-HMM",
    "M3_sent_hmm": "Model 3: Sentiment HMM",
    "M4_sent_pca_hmm": "Model 4: Sentiment PCA-HMM",
}
COLORS = {
    "M1_price_hmm": "#1f77b4",
    "M2_price_pca_hmm": "#2ca02c",
    "M3_sent_hmm": "#d62728",
    "M4_sent_pca_hmm": "#9467bd",
}


def plot_cumulative_returns(curves, spy_bh, path=None):
    """curves: {label: returns Series} for all model/framework variants."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, rets in curves.items():
        eq = equity_curve(rets)
        ax.plot(eq.index, eq.values, lw=1.4, label=label)
    eq = equity_curve(spy_bh)
    ax.plot(eq.index, eq.values, lw=2.0, color="black", ls="--", label="Buy & Hold SPY")
    ax.set_title("Cumulative Wealth: 4 Regime Models x {Switch, MVO} vs Benchmarks (2023, net of 5 bps)")
    ax.set_ylabel("Wealth ($1 start)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = path or f"{FIG_DIR}/cumulative_returns_all_models.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_regime_comparison(spy_price, model_states, path=None):
    """model_states: {model: states Series} with 0=bull, 1=bear (values in {0,1})."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (name, states) in zip(axes.ravel(), model_states.items()):
        ax.plot(spy_price.index, spy_price.values, lw=0.9, color="black")
        bear = states.values == 1
        ax.fill_between(
            spy_price.index, spy_price.min(), spy_price.max(),
            where=bear, color="#d62728", alpha=0.18, label="Bear/High-Vol state",
        )
        ax.set_title(MODEL_LABELS.get(name, name), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("SPY")
    fig.suptitle("Regime Classification: SPY with Bear-State Shading (2023)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = path or f"{FIG_DIR}/regime_classification_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_mvo_weights(weights_m2, weights_m4, p_bull_m4=None, path=None):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (name, w) in zip(axes, [("Model 2: Price PCA-HMM (MVO)", weights_m2), ("Model 4: Sentiment PCA-HMM (MVO)", weights_m4)]):
        ax.fill_between(w.index, 0, w["SPY"], color="#1f77b4", alpha=0.55, label="SPY weight")
        ax.fill_between(w.index, w["SPY"], 1, color="#ff7f0e", alpha=0.55, label="IEF weight")
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    if p_bull_m4 is not None:
        axes[1].twinx().plot(p_bull_m4.index, p_bull_m4.values, color="black", lw=0.8, alpha=0.6)
        axes[1].twinx().set_ylabel("P(bull)")
    fig.suptitle("Regime-Conditioned MVO: Dynamic Equity/Bond Weights")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = path or f"{FIG_DIR}/mvo_weights_over_time.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_transition_lead_time(leads, path=None):
    """leads: {model: {'lead_days': int|None, 'entry_date': ts|None,
    'carried_in': bool}} — bars show lead days; entry date annotated."""
    names = list(leads.keys())
    vals = [leads[n]["lead_days"] if leads[n]["lead_days"] is not None else 0 for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    labels = []
    for n in names:
        info = leads[n]
        note = "no flip"
        if info["lead_days"] is not None:
            if info["carried_in"]:
                note = "bear since window start"
            elif info["entry_date"] is not None:
                note = f"entered {pd.Timestamp(info['entry_date']).date()}"
        labels.append(f"{MODEL_LABELS.get(n, n)}\n({note})")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    for name, v, bar in zip(names, vals, bars):
        txt = f"{v:+d}d" if leads[name]["lead_days"] is not None else "n/a"
        off = 0.15 if v >= 0 else -0.15
        ax.text(bar.get_width() + off, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=9)
    ax.set_xlabel("Trading days of lead before 2023-03-13 (SVB stress realized in prices)")
    ax.set_title("Regime Transition Timing into the March 2023 Banking Crisis")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = path or f"{FIG_DIR}/transition_lead_time_svb.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
