"""Baseline audit + walk-forward (OOS) robustness + statistical significance.

1. Verifies the price-only baseline (M1/M2) contains NO sentiment inputs and that
   its switch returns reproduce through an independent recomputation path.
2. Walk-forward refit: for each 2023 month, each model is re-fit ONLY on prior
   data and predicts that month's states — eliminating in-sample optimism.
3. Significance: JK-Memmel + block-bootstrap Sharpe-difference tests (Holm-
   corrected), DM on return differentials, stress-proxy F1 + McNemar.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import (
    COST_BPS,
    EVAL_END,
    EVAL_START,
    FEATURE_START,
    SVB_REF_DATE,
    SVB_WINDOW_START,
    TABLE_DIR,
)
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import max_drawdown, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.regime_models import FEATURE_SETS, _fit_one
from hmm_sentiment.stats_tests import (
    bootstrap_sharpe_diff,
    dm_test,
    holm_correction,
    jk_memmel_sharpe_diff,
    mcnemar_test,
    stress_f1,
)
from hmm_sentiment.vader_scoring import vader_scores


def build_features():
    market = fetch_market_data()
    news = load_corpus()
    scored = score_articles(news)
    scored["importance"] = rate_importance(scored["headline"].tolist())

    v_scored = vader_scores(news)
    v_scored["importance"] = rate_importance(v_scored["headline"].tolist())

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"}
    )
    feat = market.join(daily, how="left").join(
        v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left"
    )
    return market, feat


def month_posterior(train_feat, month_feat, cols, use_pca, random_state=42):
    """Fit on train, then run a CAUSAL filtered forward pass over the month.

    Replaced the original predict_proba smoothing (forward-BACKWARD pass over
    the whole month frame), which let day-t posteriors depend on days
    t+1..month-end — a lookahead channel in every walk-forward path. See
    regime_models.filtered_month_posterior.
    """
    from hmm_sentiment.regime_models import filtered_month_posterior

    return filtered_month_posterior(train_feat, month_feat, cols, use_pca, random_state)


def walk_forward(feat, eval_idx):
    """OOS posteriors: for each month, fit on data before the month, predict it."""
    oos = {}
    months = pd.date_range(EVAL_START, EVAL_END, freq="MS")
    for m_start in months:
        m_end = (m_start + pd.offsets.MonthEnd(0))
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        if len(month) == 0:
            continue
        for name, (cols, use_pca) in FEATURE_SETS.items():
            post = month_posterior(train, month, cols, use_pca)
            oos.setdefault(name, {})[m_start] = post
    return {name: pd.concat(list(d.values())).sort_index() for name, d in oos.items()}


def switch_perf(p_bull, rets, eval_idx):
    r, pos = bt.run_binary_switch(p_bull.loc[eval_idx], rets, cost_bps=COST_BPS)
    return r


def main():
    market, feat = build_features()
    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})

    # ---------- 1. baseline audit ----------
    m1_cols = FEATURE_SETS["M1_price_hmm"][0]
    m2_cols = FEATURE_SETS["M2_price_pca_hmm"][0]
    sent_cols = [c for c in m1_cols + m2_cols if "sent" in c or "imp" in c]
    audit = {"m1_cols": m1_cols, "m2_cols": m2_cols, "sentiment_in_price_features": sent_cols}
    assert not sent_cols, f"SENTIMENT LEAKED INTO PRICE BASELINE: {sent_cols}"

    models = {n: _fit_one(feat, cols, pca, 42) for n, (cols, pca) in FEATURE_SETS.items()}
    p1 = models["M1_price_hmm"]["posterior_bull"]
    rets_is, _ = bt.run_binary_switch(p1.loc[eval_idx], rets, cost_bps=COST_BPS)
    # independent recomputation path
    pos = (p1.loc[eval_idx] >= 0.5).astype(float).shift(1).fillna(0.0)
    manual = pos * rets["SPY"].reindex(eval_idx).fillna(0) + (1 - pos) * rets["IEF"].reindex(eval_idx).fillna(0)
    turn = pos.diff().abs().fillna(0.0)
    manual -= (COST_BPS / 1e4) * turn
    audit["independent_recompute_max_abs_diff"] = float((manual - rets_is).abs().max())
    audit["trades_2023"] = int((pos.diff().abs() > 0).sum())
    assert audit["independent_recompute_max_abs_diff"] < 1e-12

    spy_2023 = market.loc[EVAL_START:EVAL_END, "SPY_close"]
    audit["spy_2023_total_return"] = float(spy_2023.iloc[-1] / spy_2023.iloc[0] - 1)
    print("AUDIT:", json.dumps(audit, indent=2))

    # ---------- 2. walk-forward OOS ----------
    print("[stats] walk-forward refits (72 fits)...", flush=True)
    oos_post = walk_forward(feat, eval_idx)
    perf_rows = []
    oos_rets = {}
    is_rets = {}
    for name in FEATURE_SETS:
        p_is = models[name]["posterior_bull"].loc[eval_idx]
        r_is, _ = bt.run_binary_switch(p_is, rets, cost_bps=COST_BPS)
        r_oos = switch_perf(oos_post[name], rets, eval_idx)
        oos_rets[name] = r_oos
        is_rets[name] = r_is
        s_is, s_oos = perf_stats(r_is, name), perf_stats(r_oos, name)
        perf_rows.append(
            {"model": name, "framework": "Switch",
             "is_sharpe": s_is["sharpe"], "oos_sharpe": s_oos["sharpe"],
             "is_ann_return": s_is["ann_return"], "oos_ann_return": s_oos["ann_return"],
             "is_max_dd": s_is["max_drawdown"], "oos_max_dd": s_oos["max_drawdown"]}
        )
        print(f"  {name}: IS {s_is['sharpe']:.2f} -> OOS {s_oos['sharpe']:.2f}", flush=True)

    bh = bt.buy_and_hold(rets).loc[eval_idx]
    perf_rows.append({"model": "Buy & Hold SPY", "framework": "-", "is_sharpe": perf_stats(bh)["sharpe"],
                      "oos_sharpe": perf_stats(bh)["sharpe"], "is_ann_return": perf_stats(bh)["ann_return"],
                      "oos_ann_return": perf_stats(bh)["ann_return"], "is_max_dd": max_drawdown(bh),
                      "oos_max_dd": max_drawdown(bh)})
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(f"{TABLE_DIR}/oos_robustness.csv", index=False)

    # ---------- 3. significance ----------
    rows = []
    pairs = [
        ("M3_sent_hmm", "M1_price_hmm"),
        ("M4_sent_pca_hmm", "M2_price_pca_hmm"),
        ("M5_vader_hmm", "M1_price_hmm"),
        ("M3_sent_hmm", "M5_vader_hmm"),
        ("M4_sent_pca_hmm", "M6_vader_pca_hmm"),
        ("M3_sent_hmm", None),  # vs B&H
    ]
    for a, b in pairs:
        for mode, source in [("IS", is_rets), ("OOS", oos_rets)]:
            ra = source[a]
            rb = bh if b is None else source[b]
            jk = jk_memmel_sharpe_diff(ra.values, rb.values)
            boot = bootstrap_sharpe_diff(ra.values, rb.values, n_boot=5000, block=10, seed=7)
            dm = dm_test(ra.values, rb.values, lags=5)
            rows.append({
                "mode": mode, "pair": f"{a} vs {b or 'B&H SPY'}",
                "sr_diff": jk["sr_diff"], "jk_z": jk["z"], "jk_p": jk["p_value"],
                "boot_p": boot["p_value"], "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"],
                "dm_p": dm["p_value"],
            })
    stats_df = pd.DataFrame(rows)
    for mode in ("IS", "OOS"):
        mask = stats_df["mode"] == mode
        stats_df.loc[mask, "holm_p"] = holm_correction(stats_df.loc[mask, "jk_p"].values)
    stats_df.to_csv(f"{TABLE_DIR}/stats_summary.csv", index=False)

    # stress proxy: VIX > 25
    proxy = (feat.loc[eval_idx, "VIX_close"] > 25).astype(int)
    bear_flags = {}
    for name in FEATURE_SETS:
        bear_flags[name] = (oos_post[name].loc[eval_idx] < 0.5).astype(int)
    f1_rows = []
    for name, flag in bear_flags.items():
        f1_rows.append({"model": name, "f1_vix25": stress_f1(proxy.values, flag.values),
                        "bear_days": int(flag.sum()),
                        "proxy_days": int(proxy.sum())})
    m1_b, m3_b = bear_flags["M1_price_hmm"], bear_flags["M3_sent_hmm"]
    # McNemar over ALL eval days (conditioning on the proxy voids the
    # marginal-homogeneity null — audit finding). Proxy-conditional flag rates
    # are reported descriptively below.
    b_m3_only = int(((m3_b == 1) & (m1_b == 0)).sum())
    c_m1_only = int(((m1_b == 1) & (m3_b == 0)).sum())
    mc = mcnemar_test(b=b_m3_only, c=c_m1_only)
    stress_df = pd.DataFrame(f1_rows)
    stress_df["m3_bear_on_proxy_days"] = [int(((bear_flags["M3_sent_hmm"] == 1) & (proxy == 1)).sum()),
                                          int(((bear_flags["M1_price_hmm"] == 1) & (proxy == 1)).sum())]
    stress_df.to_csv(f"{TABLE_DIR}/stress_f1.csv", index=False)
    print("STRESS F1 vs VIX>25:\n", stress_df.to_string(index=False))
    print("McNemar (all-days discordant bear flags, M3-only vs M1-only):", mc)

    print("\nIS/OOS table:\n", perf.to_string(index=False))
    print("\nSignificance (2023 OOS only — IS rows are diagnostic only and are "
          "excluded from significance output, audit finding):\n",
          stats_df[stats_df["mode"] == "OOS"].to_string(index=False))


if __name__ == "__main__":
    main()
