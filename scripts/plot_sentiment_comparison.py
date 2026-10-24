"""Focused comparison: FinBERT+HMM (M3/M4) vs VADER+HMM (M5/M6) vs price-only
baseline HMM (M1) — cumulative wealth (Switch) + Sharpe bars (Switch/MVO)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, EVAL_END, EVAL_START, FIG_DIR
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import equity_curve, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.plots import MODEL_LABELS
from hmm_sentiment.regime_models import FEATURE_SETS, _fit_one
from hmm_sentiment.vader_scoring import vader_scores

FOCUS = ["M1_price_hmm", "M3_sent_hmm", "M4_sent_pca_hmm", "M5_vader_hmm", "M6_vader_pca_hmm"]
LINE_COLORS = {
    "M1_price_hmm": "black",
    "M3_sent_hmm": "#d62728",
    "M4_sent_pca_hmm": "#9467bd",
    "M5_vader_hmm": "#ff7f0e",
    "M6_vader_pca_hmm": "#8c564b",
}


def main():
    market = fetch_market_data()
    news = load_corpus()
    scored = score_articles(news)
    scored["importance"] = rate_importance(scored["headline"].tolist())
    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_scored = vader_scores(news)
    v_scored["importance"] = rate_importance(v_scored["headline"].tolist())
    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"}
    )
    feat = market.join(daily, how="left").join(
        v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left"
    )

    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})

    models = {n: _fit_one(feat, cols, pca, 42) for n, (cols, pca) in FEATURE_SETS.items()}

    switch_curves = {}
    mvo_curves = {}
    stats_rows = []
    for name in FOCUS:
        p_bull = models[name]["posterior_bull"].loc[eval_idx]
        r_sw, _ = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
        r_mvo, _ = bt.run_mvo(p_bull, rets, models[name]["states"], cost_bps=COST_BPS)
        switch_curves[name] = equity_curve(r_sw)
        mvo_curves[name] = equity_curve(r_mvo)
        stats_rows.append({
            "model": MODEL_LABELS[name],
            "switch_sharpe": perf_stats(r_sw)["sharpe"],
            "mvo_sharpe": perf_stats(r_mvo)["sharpe"],
            "switch_ann_return": perf_stats(r_sw)["ann_return"],
        })
        print(f"{name}: switch={stats_rows[-1]['switch_sharpe']:.3f} mvo={stats_rows[-1]['mvo_sharpe']:.3f}")

    bh = bt.buy_and_hold(rets).loc[eval_idx]
    stats_rows.append({"model": "Buy & Hold SPY", "switch_sharpe": perf_stats(bh)["sharpe"],
                       "mvo_sharpe": perf_stats(bh)["sharpe"], "switch_ann_return": perf_stats(bh)["ann_return"]})
    pd.DataFrame(stats_rows).to_csv(f"{FIG_DIR.replace('/figures', '/tables')}/focus_comparison.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})

    eq_bh = equity_curve(bh)
    ax1.plot(eq_bh.index, eq_bh.values, lw=1.6, color="gray", ls="--", label="Buy & Hold SPY")
    for name in FOCUS:
        eq = switch_curves[name]
        lw = 2.2 if name == "M1_price_hmm" else 1.6
        ls = (0, (4, 2)) if "pca" in name else "-"
        ax1.plot(eq.index, eq.values, lw=lw, ls=ls, color=LINE_COLORS[name], label=MODEL_LABELS[name])
    ax1.set_title("Switch strategies: cumulative wealth (2023, net of 5 bps)")
    ax1.set_ylabel("Wealth ($1 start)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    width = 0.38
    xs = np.arange(len(stats_rows))
    ax2.bar(xs - width / 2, [r["switch_sharpe"] for r in stats_rows], width, label="Switch", color="#1f77b4")
    ax2.bar(xs + width / 2, [r["mvo_sharpe"] for r in stats_rows], width, label="MVO", color="#ff7f0e")
    labels = ["Price HMM\n(baseline)", "FinBERT\nHMM", "FinBERT\nPCA-HMM", "VADER\nHMM", "VADER\nPCA-HMM", "B&H\nSPY"]
    ax2.set_xticks(xs, labels, fontsize=8)
    ax2.axhline(perf_stats(bh)["sharpe"], color="gray", lw=1, ls=":")
    ax2.set_title("Net Sharpe by model and framework")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("LLM-weighted sentiment (FinBERT) vs VADER vs price-only baseline HMM — 2023", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{FIG_DIR}/sentiment_vs_baseline_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("SAVED", out)


if __name__ == "__main__":
    main()
