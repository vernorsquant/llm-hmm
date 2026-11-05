"""FinBERT vs VADER over the EXTENDED corpus (2017-2026): does FinBERT win
once there is real statistical power?

Monthly walk-forward (prior-data-only refits, causal filtered posteriors) for
  M1 price-only · M3 FinBERT · M5 VADER
plus B&H, over every month from corpus start through today. Significance:
JK-Memmel + stationary block bootstrap + Holm across the 4-pair family.
Also: year-by-year switch Sharpe and the daily-series correlation FinBERT~VADER.

Importance ratings (Qwen) are SKIPPED for the extended corpus: with
WEIGHTED_SENTIMENT=False the rating does not enter any HMM feature
(importance only feeds hi_imp_volume, which is excluded from model inputs and
not part of this comparison) — disclosed.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, TABLE_DIR
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import max_drawdown, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.stats_tests import (
    bootstrap_sharpe_diff,
    holm_correction,
    jk_memmel_sharpe_diff,
    stress_f1,
)
from hmm_sentiment.vader_scoring import vader_scores
from run_long_backtest import month_posterior

START = "2016-06-01"          # market warmup before the first corpus month
END = pd.Timestamp.today().strftime("%Y-%m-%d")
EXT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw", "news_gdelt_extended.csv")
STRATEGIES = ["switch", "mvo", "prob_weight", "hysteresis", "vol_target"]
MODELS = {
    "M1_price_hmm": (["SPY_ret", "realized_vol", "VIX_close"], False),
    "M3_finbert_hmm": (["SPY_ret", "realized_vol", "VIX_close", "sent_w", "sent_w_ema5", "sent_disp"], False),
    "M5_vader_hmm": (["SPY_ret", "realized_vol", "VIX_close", "sent_w_v", "sent_w_ema5_v", "sent_disp_v"], False),
}


def build_extended_features():
    market = fetch_market_data(start=START, end=END, use_cache=False)
    news23 = load_corpus()
    ext_path = EXT_CSV
    frames = [news23]
    if os.path.exists(ext_path):
        ext = pd.read_csv(ext_path)
        ext["timestamp_utc"] = pd.to_datetime(ext["timestamp_utc"], utc=True)
        frames.append(ext)
        print(f"[ext-zoo] 2023 corpus {len(news23)} + extended {len(ext)} rows", flush=True)
    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["headline", "window"]).reset_index(drop=True)
    print(f"[ext-zoo] combined corpus (deduped on headline+window): {len(news)}", flush=True)

    scored = score_articles(news)
    v = vader_scores(news)
    # importance ratings skipped for the extended corpus: with unweighted
    # aggregation they enter no HMM feature (disclosed); hi_imp_volume is not
    # part of this comparison.
    scored["importance"] = 0.5
    v["importance"] = 0.5

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_merged = map_articles_to_trading_days(v, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"})
    feat = market.join(daily, how="left").join(
        v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left")
    daily_series = pd.DataFrame({
        "finbert_sent_w": daily["sent_w"],
        "vader_sent_w": v_daily["sent_w_v"],
        "finbert_n": daily["n_labeled"],
        "vader_n": v_daily["n_labeled"],
    })
    return market, feat, daily_series


def weight_net_returns(w, rets, cost_bps=COST_BPS):
    gross = w * rets["SPY"].reindex(w.index).fillna(0.0) + (1 - w) * rets["IEF"].reindex(w.index).fillna(0.0)
    return gross - (cost_bps / 1e4) * w.diff().abs().fillna(0.0)


def main():
    market, feat, daily_series = build_extended_features()
    daily_series.to_csv(f"{TABLE_DIR}/extended_daily_series.csv")

    eval_idx = feat.index[(feat.index >= "2017-02-01") & (feat.index <= END)]
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    bh = bt.buy_and_hold(rets).loc[eval_idx]

    months = pd.date_range("2017-02-01", END, freq="MS")
    months = [m for m in months if m <= pd.Timestamp(END)]
    print(f"[ext-zoo] walk-forward refits: {len(months)} months x {len(MODELS)} models", flush=True)

    oos = {}
    for k, m_start in enumerate(months, start=1):
        m_end = m_start + pd.offsets.MonthEnd(0)
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        if len(month) == 0 or len(train) < 120:
            continue
        for name, (cols, use_pca) in MODELS.items():
            try:
                post = month_posterior(train, month, cols, use_pca, 42)
                oos.setdefault(name, {})[m_start] = post
            except Exception as e:
                print(f"[ext-zoo] {name} {m_start.date()} failed: {e}", flush=True)
        if k % 12 == 0:
            print(f"[ext-zoo] {k}/{len(months)} months done", flush=True)
    oos = {name: pd.concat(list(dd.values())).sort_index() for name, dd in oos.items()}

    rows, curves = [], {}
    for name in MODELS:
        p_bull = oos[name].loc[eval_idx].dropna()
        idx = p_bull.index
        states = np.where(p_bull >= 0.5, 0, 1)
        for strat in STRATEGIES:
            if strat == "switch":
                r, pos = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
                turnover = float(pos.diff().abs().mean())
            elif strat == "mvo":
                r, w = bt.run_mvo(p_bull, rets.loc[idx], states, cost_bps=COST_BPS)
                turnover = float(w["SPY"].diff().abs().mean())
            elif strat == "prob_weight":
                w = bt.prob_weighted_weights(p_bull)
                r = weight_net_returns(w, rets)
                turnover = float(w.diff().abs().mean())
            elif strat == "hysteresis":
                w = bt.hysteresis_positions(p_bull)
                r = weight_net_returns(w, rets)
                turnover = float(w.diff().abs().mean())
            else:
                w = bt.vol_target_switch_weights(p_bull, feat["realized_vol"], 0.10)
                r = weight_net_returns(w, rets)
                turnover = float(w.diff().abs().mean())
            s = perf_stats(r, f"{name}|{strat}")
            rows.append({"model": name, "strategy": strat, "ann_return": s["ann_return"],
                         "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                         "max_drawdown": max_drawdown(r), "calmar": s["calmar"],
                         "turnover_daily": turnover, "n_days": len(r)})
            if strat == "switch":
                curves[name] = r
        print(f"[ext-zoo] {name} done", flush=True)

    s_bh = perf_stats(bh, "buy_hold_spy")
    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLE_DIR}/extended_zoo.csv", index=False)

    # year-by-year switch Sharpe
    yearly = []
    for name, r in {**curves, "buy_hold_spy": bh}.items():
        for year, seg in r.groupby(r.index.year):
            yearly.append({"model": name, "year": int(year),
                           "sharpe": perf_stats(seg)["sharpe"],
                           "ann_return": perf_stats(seg)["ann_return"]})
    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(f"{TABLE_DIR}/extended_yearly.csv", index=False)

    # significance family (switch returns, full extended window)
    pairs = [("M3_finbert_hmm", "M5_vader_hmm"), ("M3_finbert_hmm", "M1_price_hmm"),
             ("M5_vader_hmm", "M1_price_hmm"), ("M3_finbert_hmm", None),
             ("M5_vader_hmm", None), ("M1_price_hmm", None)]
    sig_rows, raw_ps = [], []
    for a, b in pairs:
        ra = curves[a].values
        rb = bh.values if b is None else curves[b].values
        jk = jk_memmel_sharpe_diff(ra, rb)
        label = f"{a} vs {b or 'B&H SPY'}"
        seed = int(__import__("hashlib").sha1(label.encode()).hexdigest()[:8], 16) % (2 ** 31)
        boot = bootstrap_sharpe_diff(ra, rb, n_boot=5000, block=10, seed=seed)
        raw_ps.append(jk["p_value"])
        sig_rows.append({"pair": label, "sr_diff_daily": jk["sr_diff"], "jk_z": jk["z"],
                         "jk_p": jk["p_value"], "boot_p": boot["p_value"], "n_days": len(ra)})
    for r_, h in zip(sig_rows, holm_correction(raw_ps)):
        r_["holm_p"] = h
    sig_df = pd.DataFrame(sig_rows)

    def _mde(row):
        se = abs(row["sr_diff_daily"] / row["jk_z"]) if abs(row["jk_z"]) > 1e-12 else np.nan
        return float(2.80 * se * np.sqrt(252)) if np.isfinite(se) and se > 0 else np.nan

    sig_df["mde_ann_80pct"] = sig_df.apply(_mde, axis=1)
    sig_df.to_csv(f"{TABLE_DIR}/extended_significance.csv", index=False)

    # stress F1 vs VIX>25 (all eval days)
    proxy = (feat.loc[eval_idx, "VIX_close"] > 25).astype(int)
    f1 = {n: stress_f1(proxy.values, (oos[n].loc[eval_idx] < 0.5).astype(int).values) for n in MODELS}
    pd.DataFrame([{"model": n, "f1_vix25": f1[n]} for n in f1]).to_csv(
        f"{TABLE_DIR}/extended_stress_f1.csv", index=False)

    corr = float(daily_series["finbert_sent_w"].corr(daily_series["vader_sent_w"]))
    with open(f"{TABLE_DIR}/extended_diagnostics.json", "w") as f:
        json.dump({
            "window": [str(eval_idx.min().date()), str(eval_idx.max().date())],
            "n_days": int(len(eval_idx)),
            "n_months_walkforward": len(months),
            "finbert_vader_daily_corr": round(corr, 4),
            "stress_f1_vix25": {k: round(v, 3) for k, v in f1.items()},
            "importance_ratings_skipped": True,
        }, f, indent=2)

    print(df.sort_values("sharpe", ascending=False).round(3).to_string(index=False), flush=True)
    print("\nSIGNIFICANCE (switch, extended window):\n", sig_df.round(4).to_string(index=False), flush=True)
    print("\nFinBERT~VADER daily corr:", round(corr, 4), flush=True)
    print("\nYEARLY:\n", yearly_df.pivot_table(index="year", columns="model", values="sharpe").round(2).to_string(), flush=True)
    print("EXTENDED_ZOO_DONE", flush=True)


if __name__ == "__main__":
    main()
