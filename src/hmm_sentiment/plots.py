"""Publication-grade (300 DPI) figures for the 4-model benchmark."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR
from .metrics import equity_curve


def _datefmt(ax):
    """Clean, non-overlapping date ticks."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    return ax

MODEL_LABELS = {
    "M1_price_hmm": "Model 1: Price HMM",
    "M2_price_pca_hmm": "Model 2: Price PCA-HMM",
    "M3_sent_hmm": "Model 3: Sentiment HMM",
    "M4_sent_pca_hmm": "Model 4: Sentiment PCA-HMM",
    "M5_vader_hmm": "Model 5: VADER HMM",
    "M6_vader_pca_hmm": "Model 6: VADER PCA-HMM",
}
COLORS = {
    "M1_price_hmm": "#1f77b4",
    "M2_price_pca_hmm": "#2ca02c",
    "M3_sent_hmm": "#d62728",
    "M4_sent_pca_hmm": "#9467bd",
    "M5_vader_hmm": "#ff7f0e",
    "M6_vader_pca_hmm": "#8c564b",
    "M7_fingpt_hmm": "#e377c2",
    "M8_fingpt_pca_hmm": "#7f7f7f",
    "M9_lbl_hmm": "#17becf",
    "M10_lbl_pca_hmm": "#bcbd22",
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
    n = len(model_states)
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4 * nrows), sharex=True, squeeze=False)
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
    for ax in axes.ravel()[n:]:
        ax.axis("off")
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


# ------------------------------------------------- LLM-Labeler comparison

def plot_labeler_correlation(daily, path=None):
    """daily: DataFrame indexed by trading day with columns
    finbert_sent_w, labeler_sent_w, finbert_n, labeler_n.

    Three readable panels: (top, full width) the two daily series overlaid with
    the SVB stress window shaded; (bottom left) scatter with 45-degree line and
    correlation annotations; (bottom right) rolling 21-day Pearson correlation.
    """
    from scipy.stats import spearmanr

    fb = daily["finbert_sent_w"]
    lb = daily["labeler_sent_w"]
    r_all = float(fb.corr(lb))
    rho_all = float(spearmanr(fb, lb).statistic)
    covered = (daily["finbert_n"] > 0) & (daily["labeler_n"] > 0)
    r_cov = float(fb[covered].corr(lb[covered])) if covered.sum() > 2 else np.nan
    rho_cov = float(spearmanr(fb[covered], lb[covered]).statistic) if covered.sum() > 2 else np.nan

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(fb.index, fb.values, lw=1.1, color="#d62728", label="FinBERT daily sentiment (sent_w)", alpha=0.9)
    ax.plot(lb.index, lb.values, lw=1.1, color="#17becf", label="LLM-Labeler impact score (sent_w_lbl)", alpha=0.9)
    ax.axvspan(pd.Timestamp("2023-03-08"), pd.Timestamp("2023-03-20"), color="gray", alpha=0.25)
    ax.text(pd.Timestamp("2023-03-14"), ax.get_ylim()[1] * 0.92, "SVB", ha="center", fontsize=9, color="gray")
    ax.set_ylabel("Daily sentiment level (raw scales differ)")
    ax.set_title("LLM-Labeler (rubric-decomposed impact) vs FinBERT sentiment — daily series, 2023")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    _datefmt(ax)

    ax2 = fig.add_subplot(gs[1, 0])
    # scatter on z-scored daily series: the two scorers live on different raw
    # scales (rubric-score average vs probability difference), so a raw 45°
    # line would imply a scale equivalence that does not exist (audit MINOR-5)
    fb_z = (fb - fb.mean()) / fb.std()
    lb_z = (lb - lb.mean()) / lb.std()
    lim = max(2.8, float(np.nanmax(np.abs(np.concatenate([fb_z.values, lb_z.values])))) * 1.1)
    ax2.plot([-lim, lim], [-lim, lim], color="gray", lw=1, ls="--", label="45°")
    ax2.scatter(fb_z, lb_z, s=18, alpha=0.65, color="#1f77b4", edgecolor="none")
    ax2.set_xlabel("FinBERT sent_w (z-score)")
    ax2.set_ylabel("Labeler sent_w_lbl (z-score)")
    ax2.set_title("Daily scatter (z-scored)")
    note = (f"Pearson r = {r_all:.2f} | Spearman ρ = {rho_all:.2f}  (all days)\n"
            f"Pearson r = {r_cov:.2f} | Spearman ρ = {rho_cov:.2f}  (both-covered days)\n"
            f"raw std: FinBERT {fb.std():.3f}, Labeler {lb.std():.3f}")
    ax2.text(0.03, 0.97, note, transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    rolling = fb.rolling(21).corr(lb)
    ax3.plot(rolling.index, rolling.values, lw=1.3, color="#1f77b4")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylim(-1, 1)
    ax3.set_title("Rolling 21-day Pearson correlation")
    ax3.set_ylabel("r(t)")
    ax3.grid(alpha=0.3)
    _datefmt(ax3)
    ax3.text(0.02, 0.06, "brief ±1 dips = 21d windows over low-variation (ffill-stale) stretches",
             transform=ax3.transAxes, fontsize=8, color="dimgray")

    fig.tight_layout()
    out = path or f"{FIG_DIR}/labeler_vs_finbert_correlation.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_labeler_zoo(curves, sharpes, path=None, order=None, display=None, title=None):
    """curves: {label: OOS switch net returns}; sharpes: {label: sharpe} used
    for legend annotation. Panel A: 2023 wealth curves; Panel B: net Sharpe bars.
    order/display override the default model lineup for variant comparisons."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1.0]})

    default_order = ["Buy & Hold SPY", "M1_price_hmm", "M3_sent_hmm", "M7_fingpt_hmm", "M9_lbl_hmm"]
    default_display = {
        "Buy & Hold SPY": "Buy & Hold SPY",
        "M1_price_hmm": "Price-only HMM",
        "M3_sent_hmm": "FinBERT sentiment HMM",
        "M7_fingpt_hmm": "FinGPT one-shot LLM HMM",
        "M9_lbl_hmm": "LLM-Labeler rubric HMM (new)",
    }
    order = order or default_order
    display = display or default_display
    palette = ["black", "#1f77b4", "#d62728", "#e377c2", "#17becf",
               "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b", "#7f7f7f"]
    styles = {}
    for i, label in enumerate(order):
        if label == "Buy & Hold SPY":
            styles[label] = dict(color="black", ls="--", lw=1.8)
        else:
            styles[label] = dict(color=palette[i % len(palette)], lw=2.0 if "M9" in label or "M13" in label else 1.4)
    for label in order:
        if label not in curves:
            continue
        eq = equity_curve(curves[label])
        sr = sharpes.get(label)
        leg = f"{display[label]}  (SR {sr:.2f})" if sr is not None else display[label]
        ax.plot(eq.index, eq.values, label=leg, **styles[label])
    ax.set_title("Walk-forward OOS regime switch, 2023, net of 5 bps")
    ax.set_ylabel("Wealth ($1 start)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    names = [n for n in order if n in sharpes]
    vals = [sharpes[n] for n in names]
    cols = [styles[n].get("color", "gray") for n in names]
    bars = ax2.bar(range(len(names)), vals, color=cols, alpha=0.88)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([display[n] for n in names], rotation=20, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                 f"{v:.2f}", ha="center", fontsize=9)
    ax2.set_ylabel("Net Sharpe (2023 OOS)")
    ax2.set_title("Net Sharpe by model")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(title or "LLM-Labeler pipeline vs sentiment baselines — regime-switch strategies")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = path or f"{FIG_DIR}/labeler_vs_sentiment_zoo.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
