"""Labeler variants zoo: bigger/reasoning model + FinBERT x Labeler hybrids.

Variants (all walk-forward OOS, identical 2023 windows, 5 bps one-sided costs):
  M9_lbl_hmm        llama-3.1-8b rubric labels (from run_labeler_zoo)
  M12_lblbig_hmm    deepseek-v4.1-flash (newer reasoning hybrid) rubric labels
  M13_hybrid_hmm    FinBERT sentiment features + 8b labeler features TOGETHER
  M14_hybridbig_hmm FinBERT sentiment features + big-model labeler features
  M15/M16           PCA twins of M12/M13
plus M1 price-only and M3 FinBERT as anchors.

Outputs: tables/labeler_variants_{oos,insample,significance,daily_big}.csv,
figures/labeler_variants_zoo.png, labeler_big_vs_finbert_correlation.png.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import (
    COST_BPS,
    EVAL_END,
    EVAL_START,
    SVB_REF_DATE,
    SVB_WINDOW_START,
    TABLE_DIR,
)
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.labeler_scoring import (
    LABELER_VERSION,
    _model_tag,
    labeler_scores,
)
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import bear_entry_lead, max_drawdown, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.plots import plot_labeler_correlation, plot_labeler_zoo
from hmm_sentiment.regime_models import MARKET_FEATURES, SENTIMENT_FEATURES, fit_regime_models
from hmm_sentiment.stats_tests import (
    bootstrap_sharpe_diff,
    holm_correction,
    jk_memmel_sharpe_diff,
    stress_f1,
)
from run_labeler_zoo import strategy_return
from run_long_backtest import month_posterior

BIG_MODEL = os.getenv("LLM_LABELER_BIG_MODEL", "deepseek/deepseek-v4.1-flash-1017")
BIG_BACKEND = os.getenv("LLM_LABELER_BIG_BACKEND", "openrouter")
SMALL_MODEL = "meta-llama/llama-3.1-8b-instruct"
# SKIP_BIG=1 runs the hybrid/PCA arms only (big-model labels unavailable, e.g.
# OpenRouter credits exhausted). Feature sets for missing variants are dropped.
SKIP_BIG = os.getenv("VARIANTS_SKIP_BIG", "0") == "1"

BIG_COLS = ["sent_w_lbig", "sent_w_ema5_lbig", "sent_disp_lbig"]
FEATURE_SETS_VAR = {
    "M1_price_hmm": (MARKET_FEATURES, False),
    "M3_sent_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES, False),
    "M9_lbl_hmm": (MARKET_FEATURES + ["sent_w_lbl", "sent_w_ema5_lbl", "sent_disp_lbl"], False),
    "M12_lblbig_hmm": (MARKET_FEATURES + BIG_COLS, False),
    "M13_hybrid_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES + ["sent_w_lbl", "sent_w_ema5_lbl", "sent_disp_lbl"], False),
    "M14_hybridbig_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES + BIG_COLS, False),
    "M15_lblbig_pca_hmm": (MARKET_FEATURES + BIG_COLS, True),
    "M16_hybrid_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES + ["sent_w_lbl", "sent_w_ema5_lbl", "sent_disp_lbl"], True),
}
if SKIP_BIG:
    FEATURE_SETS_VAR = {k: v for k, v in FEATURE_SETS_VAR.items()
                        if "big" not in k and "M15" not in k}
MODEL_ORDER = list(FEATURE_SETS_VAR.keys())
STRATEGIES = ["switch", "mvo", "prob_weight", "hysteresis", "vol_target"]


def build_features():
    market = fetch_market_data()
    news = load_corpus()

    scored = score_articles(news)
    scored["importance"] = rate_importance(scored["headline"].tolist())
    lbl8 = labeler_scores(news, backend="openrouter", model=SMALL_MODEL, max_workers=16)
    lbl8["importance"] = rate_importance(lbl8["headline"].tolist())
    lblbig = None
    if not SKIP_BIG:
        lblbig = labeler_scores(news, backend=BIG_BACKEND, model=BIG_MODEL, max_workers=16)
        lblbig["importance"] = rate_importance(lblbig["headline"].tolist())

    # drop degenerate articles (degraded/missing labels -> NaN) BEFORE daily
    # mapping, so they cannot taint aggregation; cov records PRE-drop coverage
    # (post-drop coverage is 1.0 by construction — audit round-2 gate hole)
    bad8 = lbl8["avg_impact"].isna()
    cov = {"lbl8": 1.0 - float(bad8.mean())}
    if bad8.any():
        print(f"[variants] dropping {int(bad8.sum())} degenerate 8b article(s)", flush=True)
        lbl8 = lbl8.dropna(subset=["avg_impact"])
    if lblbig is not None:
        badbig = lblbig["avg_impact"].isna()
        cov["lblbig"] = 1.0 - float(badbig.mean())
        if badbig.any():
            print(f"[variants] dropping {int(badbig.sum())} degenerate big-model article(s)", flush=True)
            lblbig = lblbig.dropna(subset=["avg_impact"])

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    l8 = map_articles_to_trading_days(lbl8, market.index)
    d8 = build_daily_news_features(l8, market.index).rename(
        columns={"sent_w": "sent_w_lbl", "sent_w_ema5": "sent_w_ema5_lbl", "sent_disp": "sent_disp_lbl"})

    feat = market.join(daily[["sent_w", "sent_w_ema5", "sent_disp", "n_labeled"]], how="left")
    feat = feat.join(d8[["sent_w_lbl", "sent_w_ema5_lbl", "sent_disp_lbl"]], how="left")

    daily_big = pd.DataFrame({
        "finbert_sent_w": daily["sent_w"],
        "finbert_n": daily["n_labeled"],
    })
    if not SKIP_BIG:
        lbig = map_articles_to_trading_days(lblbig, market.index)
        dbig = build_daily_news_features(lbig, market.index).rename(
            columns={"sent_w": "sent_w_lbig", "sent_w_ema5": "sent_w_ema5_lbig", "sent_disp": "sent_disp_lbig"})
        feat = feat.join(dbig[["sent_w_lbig", "sent_w_ema5_lbig", "sent_disp_lbig"]], how="left")
        daily_big["labelerbig_sent_w"] = dbig["sent_w_lbig"]
        daily_big["labelerbig_n"] = dbig["n_labeled"]
    return market, feat, daily_big, cov


def main():
    market, feat, daily_big, cov = build_features()
    # coverage gate: abort if any variant's PRE-drop coverage < 0.995
    for name, c in cov.items():
        if c < 0.995:
            print(f"[variants] FATAL: {name} coverage {c:.4f} < 0.995 — finish labeling first.", flush=True)
            sys.exit(1)
    dropped = {k: round(1.0 - v, 4) for k, v in cov.items() if v < 1.0}
    if dropped:
        print(f"[variants] degenerate-article drop rate (disclosed): {dropped}", flush=True)
    print(f"[variants] coverage OK: {cov}", flush=True)

    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    bh = bt.buy_and_hold(rets).loc[eval_idx]
    daily_big.loc[EVAL_START:EVAL_END].to_csv(f"{TABLE_DIR}/labeler_variants_daily_big.csv")

    print("[variants] walk-forward refits (12 months x 8 models)...", flush=True)
    oos = {}
    for m_start in pd.date_range(EVAL_START, EVAL_END, freq="MS"):
        m_end = m_start + pd.offsets.MonthEnd(0)
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        for name, (cols, use_pca) in FEATURE_SETS_VAR.items():
            oos.setdefault(name, {})[m_start] = month_posterior(train, month, cols, use_pca, 42)
    oos = {name: pd.concat(list(d.values())).sort_index() for name, d in oos.items()}

    proxy = (feat.loc[eval_idx, "VIX_close"] > 25).astype(int)
    rows, switch_curves, switch_sharpes, leads, f1s = [], {}, {}, {}, {}
    for name in MODEL_ORDER:
        p_bull = oos[name].loc[eval_idx]
        states = np.where(p_bull >= 0.5, 0, 1)
        for strat in STRATEGIES:
            r, turnover = strategy_return(strat, p_bull, states, rets, feat["realized_vol"])
            s = perf_stats(r, f"{name}|{strat}")
            rows.append({"model": name, "strategy": strat, "oos": True,
                         "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                         "sharpe": s["sharpe"], "max_drawdown": max_drawdown(r),
                         "calmar": s["calmar"], "turnover_daily": turnover})
            if strat == "switch":
                switch_curves[name] = r
                switch_sharpes[name] = s["sharpe"]
                leads[name] = bear_entry_lead(1.0 - p_bull, SVB_REF_DATE, SVB_WINDOW_START)
                f1s[name] = stress_f1(proxy.values, (p_bull < 0.5).astype(int).values)
        print(f"[variants] OOS {name} done", flush=True)

    s_bh = perf_stats(bh)
    rows.append({"model": "BENCHMARK", "strategy": "buy_hold_spy", "oos": True,
                 "ann_return": s_bh["ann_return"], "ann_vol": s_bh["ann_vol"],
                 "sharpe": s_bh["sharpe"], "max_drawdown": max_drawdown(bh),
                 "calmar": s_bh["calmar"], "turnover_daily": np.nan})
    pd.DataFrame(rows).to_csv(f"{TABLE_DIR}/labeler_variants_oos.csv", index=False)

    # significance family (pre-registered): big vs small, big/hybrid vs FinBERT,
    # hybrid vs FinBERT, hybrid vs 8b labeler, hybrid vs price. Pairs referencing
    # skipped variants are dropped (SKIP_BIG mode).
    pairs = [("M12_lblbig_hmm", "M9_lbl_hmm"), ("M12_lblbig_hmm", "M3_sent_hmm"),
             ("M13_hybrid_hmm", "M3_sent_hmm"), ("M13_hybrid_hmm", "M9_lbl_hmm"),
             ("M13_hybrid_hmm", "M1_price_hmm"), ("M14_hybridbig_hmm", "M3_sent_hmm")]
    pairs = [(a, b) for a, b in pairs if a in switch_curves and (b is None or b in switch_curves)]
    sig_rows, raw_ps = [], []
    for a, b in pairs:
        ra = switch_curves[a].values
        rb = bh.values if b is None else switch_curves[b].values
        jk = jk_memmel_sharpe_diff(ra, rb)
        label = f"{a} vs {b or 'B&H SPY'}"
        seed = int(__import__("hashlib").sha1(label.encode()).hexdigest()[:8], 16) % (2 ** 31)
        boot = bootstrap_sharpe_diff(ra, rb, n_boot=5000, block=10, seed=seed)
        raw_ps.append(jk["p_value"])
        sig_rows.append({"pair": label, "sr_diff_daily": jk["sr_diff"], "jk_z": jk["z"],
                         "jk_p": jk["p_value"], "boot_p": boot["p_value"]})
    for r_, h in zip(sig_rows, holm_correction(raw_ps)):
        r_["holm_p"] = h
    sig_df = pd.DataFrame(sig_rows)

    def _mde(row):
        se = abs(row["sr_diff_daily"] / row["jk_z"]) if abs(row["jk_z"]) > 1e-12 else np.nan
        return float(2.80 * se * np.sqrt(252)) if np.isfinite(se) and se > 0 else np.nan

    sig_df["mde_ann_80pct"] = sig_df.apply(_mde, axis=1)
    sig_df.to_csv(f"{TABLE_DIR}/labeler_variants_significance.csv", index=False)

    # diagnostics
    diag = {
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
        "labeler_version": LABELER_VERSION,
        "coverage": cov,
        "posterior_corr_oos": {
            f"{a}::{b}": round(float(oos[a].loc[eval_idx].corr(oos[b].loc[eval_idx])), 4)
            for a, b in [(x, y) for x, y in [("M12_lblbig_hmm", "M9_lbl_hmm"), ("M12_lblbig_hmm", "M3_sent_hmm"),
                                             ("M13_hybrid_hmm", "M3_sent_hmm"), ("M13_hybrid_hmm", "M9_lbl_hmm"),
                                             ("M14_hybridbig_hmm", "M13_hybrid_hmm")]
                         if x in oos and y in oos]
        },
        "svb_lead_days": {n: leads[n]["lead_days"] for n in leads},
        "stress_f1_vix25": {n: round(float(f1s[n]), 3) for n in f1s},
    }
    with open(f"{TABLE_DIR}/labeler_variants_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # figures
    if not SKIP_BIG and "labelerbig_sent_w" in daily_big.columns:
        plot_labeler_correlation(
            daily_big.rename(columns={"labelerbig_sent_w": "labeler_sent_w",
                                      "labelerbig_n": "labeler_n"}).loc[EVAL_START:EVAL_END],
            path=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/outputs/figures/labeler_big_vs_finbert_correlation.png")
    curves = {"Buy & Hold SPY": bh, **switch_curves}
    sharpes = {"Buy & Hold SPY": s_bh["sharpe"], **switch_sharpes}
    plot_labeler_zoo(
        curves, sharpes,
        path=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/outputs/figures/labeler_variants_zoo.png",
        order=["Buy & Hold SPY", "M1_price_hmm", "M3_sent_hmm", "M9_lbl_hmm",
               "M12_lblbig_hmm", "M13_hybrid_hmm", "M14_hybridbig_hmm"],
        display={
            "Buy & Hold SPY": "Buy & Hold SPY",
            "M1_price_hmm": "Price-only HMM",
            "M3_sent_hmm": "FinBERT sentiment",
            "M9_lbl_hmm": "Labeler 8b",
            "M12_lblbig_hmm": "Labeler deepseek-v4.1 (reasoning)",
            "M13_hybrid_hmm": "Hybrid FinBERT+8b",
            "M14_hybridbig_hmm": "Hybrid FinBERT+big",
        },
        title="Labeler variants: bigger reasoning model + FinBERT hybrids")

    print(df_print := pd.DataFrame(rows).sort_values("sharpe", ascending=False).head(14).round(3).to_string(index=False), flush=True)
    print("\nSIGNIFICANCE:\n", sig_df.round(4).to_string(index=False), flush=True)
    print("\nSVB lead / F1:", {n: (leads[n]["lead_days"], round(f1s[n], 2)) for n in leads}, flush=True)
    print("VARIANTS_ZOO_DONE", flush=True)


if __name__ == "__main__":
    main()
