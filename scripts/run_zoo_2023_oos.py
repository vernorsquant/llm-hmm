"""2023-only, apples-to-apples zoo: walk-forward (OOS) posteriors for ALL 8
models x 5 strategies, plus no-HMM price baselines (canonical SMA 50/200,
12-1 momentum, vol-target, B&H, 60/40) on identical dates.

Every model is refit monthly on PRIOR data only — the in-sample optimism that
inflated the single-fit numbers is eliminated. Significance: JK-Memmel +
block bootstrap with Holm correction across the sentiment-vs-baseline family.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, EVAL_END, EVAL_START, TABLE_DIR
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.fingpt_scoring import fingpt_scores
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import max_drawdown, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.price_strategies import sma_cross_positions
from hmm_sentiment.regime_models import FEATURE_SETS
from hmm_sentiment.vader_scoring import vader_scores
from run_long_backtest import month_posterior

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
    f_scored = fingpt_scores(news)
    f_scored["importance"] = rate_importance(f_scored["headline"].tolist())

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"})
    f_merged = map_articles_to_trading_days(f_scored, market.index)
    f_daily = build_daily_news_features(f_merged, market.index).rename(
        columns={"sent_w": "sent_w_f", "sent_w_ema5": "sent_w_ema5_f", "sent_disp": "sent_disp_f"})
    feat = market.join(daily, how="left")
    feat = feat.join(v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left")
    feat = feat.join(f_daily[["sent_w_f", "sent_w_ema5_f", "sent_disp_f"]], how="left")
    return market, feat


def weight_net_returns(w, rets, cost_bps=COST_BPS):
    gross = w * rets["SPY"].reindex(w.index).fillna(0.0) + (1 - w) * rets["IEF"].reindex(w.index).fillna(0.0)
    return gross - (cost_bps / 1e4) * w.diff().abs().fillna(0.0)


def main():
    market, feat = build_features()
    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})

    # walk-forward OOS posteriors for all 8 models (monthly refits, prior-data-only)
    print("[oos-zoo] walk-forward refits (12 months x 8 models)...", flush=True)
    oos = {}
    for m_start in pd.date_range(EVAL_START, EVAL_END, freq="MS"):
        m_end = m_start + pd.offsets.MonthEnd(0)
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        for name, (cols, use_pca) in FEATURE_SETS.items():
            oos.setdefault(name, {})[m_start] = month_posterior(train, month, cols, use_pca, 42)
    oos = {name: pd.concat(list(d.values())).sort_index() for name, d in oos.items()}

    rows = []
    heat = np.zeros((len(MODEL_ORDER), len(STRATEGIES)))
    for i, name in enumerate(MODEL_ORDER):
        p_bull = oos[name].loc[eval_idx]
        states = np.where(p_bull >= 0.5, 0, 1)
        for j, strat in enumerate(STRATEGIES):
            if strat == "switch":
                r, _ = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
            elif strat == "mvo":
                r, _ = bt.run_mvo(p_bull, rets.loc[eval_idx], states, cost_bps=COST_BPS)
            elif strat == "prob_weight":
                r = weight_net_returns(bt.prob_weighted_weights(p_bull), rets)
            elif strat == "hysteresis":
                r = weight_net_returns(bt.hysteresis_positions(p_bull), rets)
            else:
                r = weight_net_returns(bt.vol_target_switch_weights(p_bull, feat["realized_vol"], 0.10), rets)
            s = perf_stats(r, f"{name}|{strat}")
            heat[i, j] = s["sharpe"]
            rows.append({"model": name, "strategy": strat, "ann_return": s["ann_return"],
                         "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                         "max_drawdown": max_drawdown(r), "calmar": s["calmar"], "oos": True})
        print(f"[oos-zoo] {name} done", flush=True)

    # no-HMM, no-sentiment price baselines (canonical SMA 50/200; 10/40 as
    # parameter-sensitivity footnote) — same dates, same costs
    bh = bt.buy_and_hold(rets).loc[eval_idx]
    b6040 = bt.static_60_40(rets, cost_bps=COST_BPS).loc[eval_idx]
    for name, pos in [("SMA 50/200 (no HMM)", sma_cross_positions(market["SPY_close"], 50, 200)),
                      ("SMA 10/40 (param-sensitivity)", sma_cross_positions(market["SPY_close"], 10, 40)),
                      ("Vol-target 10% (no HMM)", bt.vol_target_switch_weights(
                          pd.Series(1.0, index=eval_idx), feat["realized_vol"], 0.10))]:
        w = pos.loc[eval_idx]
        r = weight_net_returns(w, rets)
        s = perf_stats(r, name)
        rows.append({"model": "NO-HMM", "strategy": name, "ann_return": s["ann_return"],
                     "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                     "max_drawdown": max_drawdown(r), "calmar": s["calmar"], "oos": True})
    rows.append({"model": "BENCHMARK", "strategy": "buy_hold_spy", "ann_return": perf_stats(bh)["ann_return"],
                 "ann_vol": perf_stats(bh)["ann_vol"], "sharpe": perf_stats(bh)["sharpe"],
                 "max_drawdown": max_drawdown(bh), "calmar": perf_stats(bh)["calmar"], "oos": True})
    rows.append({"model": "BENCHMARK", "strategy": "static_60_40", "ann_return": perf_stats(b6040)["ann_return"],
                 "ann_vol": perf_stats(b6040)["ann_vol"], "sharpe": perf_stats(b6040)["sharpe"],
                 "max_drawdown": max_drawdown(b6040), "calmar": perf_stats(b6040)["calmar"], "oos": True})

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLE_DIR}/zoo_2023_oos.csv", index=False)

    # significance (OOS returns): sentiment models vs price-HMM, vs B&H
    sig_rows = []
    r_m1 = bt.run_binary_switch(oos["M1_price_hmm"].loc[eval_idx], rets, cost_bps=COST_BPS)[0]
    pairs = [(n, "M1_price_hmm") for n in ["M3_sent_hmm", "M5_vader_hmm", "M7_fingpt_hmm"]] + \
            [("M3_sent_hmm", None), ("M1_price_hmm", None)]
    p_adj = []
    raw_ps = []
    for a, b in pairs:
        ra = bt.run_binary_switch(oos[a].loc[eval_idx], rets, cost_bps=COST_BPS)[0]
        rb = bh if b is None else bt.run_binary_switch(oos[b].loc[eval_idx], rets, cost_bps=COST_BPS)[0]
        jk = jk_memmel = __import__("hmm_sentiment.stats_tests", fromlist=["jk_memmel_sharpe_diff"]).jk_memmel_sharpe_diff(ra.values, rb.values)
        boot = __import__("hmm_sentiment.stats_tests", fromlist=["bootstrap_sharpe_diff"]).bootstrap_sharpe_diff(ra.values, rb.values, n_boot=5000, block=10, seed=7)
        label = f"{a} vs {b or 'B&H SPY'}"
        raw_ps.append(jk["p_value"])
        sig_rows.append({"pair": label, "sr_diff_daily": jk["sr_diff"], "jk_z": jk["z"],
                         "jk_p": jk["p_value"], "boot_p": boot["p_value"]})
    from hmm_sentiment.stats_tests import holm_correction
    holm = holm_correction(raw_ps)
    for r_, h in zip(sig_rows, holm):
        r_["holm_p"] = h
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{TABLE_DIR}/zoo_2023_oos_significance.csv", index=False)

    print(df.sort_values("sharpe", ascending=False).round(3).to_string(index=False), flush=True)
    print("\nSIGNIFICANCE (2023 OOS):\n", sig_df.to_string(index=False), flush=True)
    print("ZOO_OOS_DONE", flush=True)


if __name__ == "__main__":
    main()
