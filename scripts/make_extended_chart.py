"""Extended-window (2017-2026) figure: FinBERT vs VADER vs price-only vs B&H.

Refits the same monthly walk-forward used by run_extended_zoo.py and plots
cumulative wealth of the causal walk-forward regime switches + B&H, with final
Sharpe annotations. This is the chart for the 'does FinBERT beat VADER with
real power?' README section.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import FIG_DIR
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import equity_curve, perf_stats
from run_extended_zoo import MODELS, START, build_extended_features
from run_long_backtest import month_posterior

OUT = os.path.join(FIG_DIR, "extended_window_finbert_vs_vader.png")


def main():
    market, feat, _ = build_extended_features()
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    end = feat.index.max()
    months = [m for m in pd.date_range("2017-02-01", end, freq="MS")
              if feat[feat.index < m].shape[0] >= 120]
    print(f"[ext-chart] {len(months)} monthly walk-forward refits x {len(MODELS)} models", flush=True)

    curves = {"Buy & Hold SPY": bt.buy_and_hold(rets)}
    sharpes = {"Buy & Hold SPY": perf_stats(curves["Buy & Hold SPY"])["sharpe"]}
    labels = {
        "M1_price_hmm": "Price-only HMM",
        "M3_finbert_hmm": "FinBERT sentiment HMM",
        "M5_vader_hmm": "VADER sentiment HMM",
    }
    colors = {"M1_price_hmm": "#7f7f7f", "M3_finbert_hmm": "#2ca02c", "M5_vader_hmm": "#ff7f0e"}
    for name, (cols, use_pca) in MODELS.items():
        parts = []
        for m_start in months:
            m_end = m_start + pd.offsets.MonthEnd(0)
            train = feat[feat.index < m_start]
            month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
            if len(month) == 0 or len(train) < 120:
                continue
            parts.append(month_posterior(train, month, cols, use_pca, 42))
        p_bull = pd.concat(parts).sort_index().dropna()
        r, _ = bt.run_binary_switch(p_bull, rets, cost_bps=bt.COST_BPS)
        curves[labels[name]] = r
        sharpes[labels[name]] = perf_stats(r)["sharpe"]
        print(f"[ext-chart] {name} done: Sharpe {sharpes[labels[name]]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for label, r in curves.items():
        eq = equity_curve(r)
        sr = sharpes[label]
        lw = 2.2 if "FinBERT" in label or label == "Buy & Hold SPY" else 1.5
        ls = "--" if label == "Buy & Hold SPY" else "-"
        ax.plot(eq.index, eq.values, lw=lw, ls=ls,
                color={"Price-only HMM": "#7f7f7f", "FinBERT sentiment HMM": "#2ca02c",
                       "VADER sentiment HMM": "#ff7f0e"}.get(label, "black"),
                label=f"{label}  (Sharpe {sr:.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("Wealth of $1 (log scale)")
    ax.set_title("2017–2026, monthly walk-forward refits, causal posteriors, net of 5 bps:\n"
                 "FinBERT vs VADER vs price-only regime switches and buy-and-hold", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"[ext-chart] saved {OUT}", flush=True)
    print("EXT_CHART_DONE", flush=True)


if __name__ == "__main__":
    main()
