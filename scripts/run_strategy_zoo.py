"""Strategy zoo: 8 sentiment/price HMM models x 5 allocation strategies.

Strategies (all net of 5 bps on SPY-leg turnover, all lagged — signal at t-1
positions day t):
  switch       100% SPY if p_bull>0.5 else 100% IEF
  mvo          regime-conditioned blended max-Sharpe MVO
  prob_weight  w_SPY = p_bull(t-1), continuous
  hysteresis   enter bull at p>0.6, exit at p<0.4 (whipsaw damper)
  vol_target   w_SPY = p_bull(t-1) * clip(0.10 / rv(t-1), 0, 1)

Outputs: outputs/tables/strategy_zoo.csv, outputs/figures/strategy_zoo_heatmap.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, EVAL_END, EVAL_START, FIG_DIR, TABLE_DIR
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.fingpt_scoring import fingpt_scores
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.regime_models import FEATURE_SETS, _fit_one
from hmm_sentiment.vader_scoring import vader_scores

MODEL_ORDER = [
    "M1_price_hmm", "M2_price_pca_hmm",
    "M3_sent_hmm", "M4_sent_pca_hmm",
    "M5_vader_hmm", "M6_vader_pca_hmm",
    "M7_fingpt_hmm", "M8_fingpt_pca_hmm",
]
STRATEGIES = ["switch", "mvo", "prob_weight", "hysteresis", "vol_target"]


def build_features():
    market = fetch_market_data()
    news = load_corpus()

    scored = score_articles(news)
    scored["importance"] = rate_importance(scored["headline"].tolist())

    v_scored = vader_scores(news)
    v_scored["importance"] = rate_importance(v_scored["headline"].tolist())

    print("[zoo] FinGPT scoring (8B on CPU — the long stage)", flush=True)
    f_scored = fingpt_scores(news)
    f_scored["importance"] = rate_importance(f_scored["headline"].tolist())

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"}
    )
    f_merged = map_articles_to_trading_days(f_scored, market.index)
    f_daily = build_daily_news_features(f_merged, market.index).rename(
        columns={"sent_w": "sent_w_f", "sent_w_ema5": "sent_w_ema5_f", "sent_disp": "sent_disp_f"}
    )
    feat = market.join(daily, how="left")
    feat = feat.join(v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left")
    feat = feat.join(f_daily[["sent_w_f", "sent_w_ema5_f", "sent_disp_f"]], how="left")
    return market, feat


def weight_net_returns(w, rets, cost_bps=COST_BPS):
    """Net returns for a continuous w_SPY series (w for SPY, 1-w for IEF)."""
    gross = w * rets["SPY"].reindex(w.index).fillna(0.0) + (1 - w) * rets["IEF"].reindex(w.index).fillna(0.0)
    turn = w.diff().abs().fillna(0.0)
    return gross - (cost_bps / 1e4) * turn


def main():
    market, feat = build_features()
    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})

    models = {n: _fit_one(feat, cols, pca, 42) for n, (cols, pca) in FEATURE_SETS.items()}

    rows = []
    heat = np.zeros((len(MODEL_ORDER), len(STRATEGIES)))
    for i, name in enumerate(MODEL_ORDER):
        p_bull = models[name]["posterior_bull"].loc[eval_idx]
        states = models[name]["states"]
        for j, strat in enumerate(STRATEGIES):
            if strat == "switch":
                r, _ = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
            elif strat == "mvo":
                r, _ = bt.run_mvo(p_bull, rets, states, cost_bps=COST_BPS)
            elif strat == "prob_weight":
                w = bt.prob_weighted_weights(p_bull)
                r = weight_net_returns(w, rets)
            elif strat == "hysteresis":
                pos = bt.hysteresis_positions(p_bull)
                r = weight_net_returns(pos, rets)
            else:  # vol_target
                w = bt.vol_target_switch_weights(p_bull, feat["realized_vol"], target_vol=0.10)
                r = weight_net_returns(w, rets)
            s = perf_stats(r, f"{name}|{strat}")
            heat[i, j] = s["sharpe"]
            rows.append({
                "model": name, "strategy": strat,
                "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
                "calmar": s["calmar"],
            })
        print(f"[zoo] {name} done", flush=True)

    # benchmarks
    bh = bt.buy_and_hold(rets).loc[eval_idx]
    b6040 = bt.static_60_40(rets, cost_bps=COST_BPS).loc[eval_idx]
    rows.append({"model": "BENCHMARK", "strategy": "buy_hold_spy", **{k: v for k, v in perf_stats(bh).items() if k != "label"}})
    rows.append({"model": "BENCHMARK", "strategy": "static_60_40", **{k: v for k, v in perf_stats(b6040).items() if k != "label"}})

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLE_DIR}/strategy_zoo.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=2.3)
    ax.set_xticks(range(len(STRATEGIES)), STRATEGIES, rotation=20, fontsize=9)
    ax.set_yticks(range(len(MODEL_ORDER)), [m.replace("_", " ") for m in MODEL_ORDER], fontsize=9)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(STRATEGIES)):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Net Sharpe heatmap: model x strategy (2023, 5 bps costs)")
    fig.colorbar(im, label="Sharpe")
    fig.tight_layout()
    out = f"{FIG_DIR}/strategy_zoo_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    best = df[df["model"] != "BENCHMARK"].nlargest(5, "sharpe")
    print(df.to_string(index=False), flush=True)
    print("\nTOP 5:", best[["model", "strategy", "sharpe"]].to_string(index=False))
    print("SAVED", out)


if __name__ == "__main__":
    main()
