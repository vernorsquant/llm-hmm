"""LLM-Labeler vs sentiment baselines: full zoo on identical 2023 windows.

Adds two feature sets fed by the LLM-Labeler rubric-decomposition pipeline
(GLM backend, see hmm_sentiment/labeler_scoring.py):
  M9_lbl_hmm       market + labeler daily features
  M10_lbl_pca_hmm  same with PCA(0.95)

Outputs (tables/):
  labeler_zoo_2023_oos.csv   walk-forward OOS: all models x 5 strategies
  labeler_zoo_insample.csv   single-fit companion (disclosed in-sample)
  labeler_significance.csv   JK-Memmel + block bootstrap + Holm family
  labeler_daily_series.csv   daily FinBERT vs Labeler sentiment (+ coverage)
  labeler_diagnostics.json   transmats, fallback rates, posterior correlations
Figures (figures/):
  labeler_vs_finbert_correlation.png
  labeler_vs_sentiment_zoo.png
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
from hmm_sentiment.fingpt_scoring import fingpt_scores
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.labeler_scoring import (
    BACKEND,
    LABELER_VERSION,
    _model_tag,
    load_labeler_cache,
    labeler_scores,
)
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import bear_entry_lead, max_drawdown, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.plots import plot_labeler_correlation, plot_labeler_zoo
from hmm_sentiment.regime_models import FEATURE_SETS, MARKET_FEATURES, fit_regime_models
from hmm_sentiment.stats_tests import (
    bootstrap_sharpe_diff,
    holm_correction,
    jk_memmel_sharpe_diff,
    stress_f1,
)
from hmm_sentiment.vader_scoring import vader_scores
from run_long_backtest import month_posterior

LABELER_COLS = ["sent_w_lbl", "sent_w_ema5_lbl", "sent_disp_lbl"]
FEATURE_SETS_LBL = dict(FEATURE_SETS)
FEATURE_SETS_LBL["M9_lbl_hmm"] = (MARKET_FEATURES + LABELER_COLS, False)
FEATURE_SETS_LBL["M10_lbl_pca_hmm"] = (MARKET_FEATURES + LABELER_COLS, True)

MODEL_ORDER = [
    "M1_price_hmm", "M2_price_pca_hmm",
    "M3_sent_hmm", "M4_sent_pca_hmm",
    "M5_vader_hmm", "M6_vader_pca_hmm",
    "M7_fingpt_hmm", "M8_fingpt_pca_hmm",
    "M9_lbl_hmm", "M10_lbl_pca_hmm",
]
STRATEGIES = ["switch", "mvo", "prob_weight", "hysteresis", "vol_target"]

FALLBACK_RUBRIC_NAMES = {
    "Earnings & Fundamentals", "Macro & Policy", "Sentiment & Risk",
}


def build_features():
    market = fetch_market_data()
    news = load_corpus()

    scored = score_articles(news)
    scored["importance"] = rate_importance(scored["headline"].tolist())
    v_scored = vader_scores(news)
    v_scored["importance"] = rate_importance(v_scored["headline"].tolist())
    f_scored = fingpt_scores(news)
    f_scored["importance"] = rate_importance(f_scored["headline"].tolist())

    lbl_scored = labeler_scores(news)
    lbl_scored["importance"] = rate_importance(lbl_scored["headline"].tolist())

    merged = map_articles_to_trading_days(scored, market.index)
    daily = build_daily_news_features(merged, market.index)
    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index).rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"})
    f_merged = map_articles_to_trading_days(f_scored, market.index)
    f_daily = build_daily_news_features(f_merged, market.index).rename(
        columns={"sent_w": "sent_w_f", "sent_w_ema5": "sent_w_ema5_f", "sent_disp": "sent_disp_f"})
    l_merged = map_articles_to_trading_days(lbl_scored, market.index)
    l_daily = build_daily_news_features(l_merged, market.index).rename(
        columns={"sent_w": "sent_w_lbl", "sent_w_ema5": "sent_w_ema5_lbl", "sent_disp": "sent_disp_lbl"})

    feat = market.join(daily, how="left")
    feat = feat.join(v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left")
    feat = feat.join(f_daily[["sent_w_f", "sent_w_ema5_f", "sent_disp_f"]], how="left")
    feat = feat.join(l_daily[LABELER_COLS], how="left")

    daily_series = pd.DataFrame({
        "finbert_sent_w": daily["sent_w"],
        "labeler_sent_w": l_daily["sent_w_lbl"],
        "finbert_n": daily["n_labeled"],
        "labeler_n": l_daily["n_labeled"],
        "finbert_n_mapped": daily["n_articles"],
        "labeler_n_mapped": l_daily["n_articles"],
    })
    return market, feat, daily_series, lbl_scored


def strategy_return(strat, p_bull, states, rets, realized_vol):
    if strat == "switch":
        r, pos = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
        turnover = float(pos.diff().abs().mean())
        return r, turnover
    if strat == "mvo":
        # rets MUST be sliced to the posterior's index: states/weights are
        # eval-sized, and an unsliced full-history rets frame misaligns the
        # expanding-moment estimation (documented pitfall in run_zoo_2023_oos).
        r, w = bt.run_mvo(p_bull, rets.loc[p_bull.index], states, cost_bps=COST_BPS)
        return r, float(w["SPY"].diff().abs().mean())
    if strat == "prob_weight":
        w = bt.prob_weighted_weights(p_bull)
    elif strat == "hysteresis":
        w = bt.hysteresis_positions(p_bull)
    else:  # vol_target
        w = bt.vol_target_switch_weights(p_bull, realized_vol, 0.10)
    gross = w * rets["SPY"].reindex(w.index).fillna(0.0) + (1 - w) * rets["IEF"].reindex(w.index).fillna(0.0)
    r = gross - (COST_BPS / 1e4) * w.diff().abs().fillna(0.0)
    return r, float(w.diff().abs().mean())


def main():
    market, feat, daily_series, lbl_scored = build_features()
    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    bh = bt.buy_and_hold(rets).loc[eval_idx]

    # ---------------- coverage gate (audit MAJOR-3): never run on a partially
    # labeled series. Degraded/missing labels surface as NaN and are DROPPED
    # with disclosure when the deficit is tiny (< 0.5%); larger deficits abort.
    bad_mask = lbl_scored["avg_impact"].isna()
    lbl_coverage = 1.0 - float(bad_mask.mean())
    if lbl_coverage < 0.995:
        print(f"[labeler-zoo] FATAL: labeler coverage {lbl_coverage:.4f} < 0.995 — "
              f"{int(bad_mask.sum())} articles unlabeled/degraded. "
              "Rerun scripts/label_corpus.py to complete labeling.", flush=True)
        sys.exit(1)
    if bad_mask.any():
        print(f"[labeler-zoo] dropping {int(bad_mask.sum())} degenerate article(s):",
              flush=True)
        for h in lbl_scored.loc[bad_mask, "headline"].tolist():
            print(f"  - {h[:90]}", flush=True)
        lbl_scored = lbl_scored.dropna(subset=["avg_impact"]).copy()
    diag_note_coverage = round(lbl_coverage, 4)

    daily_series.loc[EVAL_START:EVAL_END].to_csv(f"{TABLE_DIR}/labeler_daily_series.csv")

    # ---------------- labeler health diagnostics (coverage, provenance)
    cache = load_labeler_cache()
    n_final = len(cache["final"])
    lbl_labeled = lbl_scored["avg_impact"].notna().sum()
    fb_ser = daily_series["finbert_sent_w"].loc[EVAL_START:EVAL_END]
    lb_ser = daily_series["labeler_sent_w"].loc[EVAL_START:EVAL_END]
    diag = {
        "backend": {"backend": BACKEND, "model": _model_tag(), "labeler_version": LABELER_VERSION},
        "articles_total": int(len(lbl_scored)) + int(bad_mask.sum()),
        "articles_used": int(len(lbl_scored)),
        "label_coverage_after_drop": round(1.0 - float(bad_mask.mean()) if bad_mask.any() else 1.0, 4),
        "dropped_degenerate_articles": int(bad_mask.sum()),
        "cache_finals_total_all_backends": n_final,
        "mean_criteria_per_article": round(float(np.mean([f.get("n_criteria", 0) for f in cache["final"].values()])), 2),
        # distribution stats: the labeler's rubric-score scale compresses vs
        # FinBERT's probability difference (audit MINOR-5) — record the shapes
        "finbert_sent_w_std": round(float(fb_ser.std()), 4),
        "labeler_sent_w_std": round(float(lb_ser.std()), 4),
        "finbert_abs_gt_0p5_frac": round(float((fb_ser.abs() > 0.5).mean()), 4),
        "labeler_abs_gt_0p5_frac": round(float((lb_ser.abs() > 0.5).mean()), 4),
        "finbert_days_covered": int((daily_series["finbert_n"] > 0).sum()),
        "labeler_days_covered": int((daily_series["labeler_n"] > 0).sum()),
    }
    print(f"[labeler-zoo] labeler health: {diag}", flush=True)

    # ---------------- walk-forward OOS posteriors (monthly refit, prior-only)
    print("[labeler-zoo] walk-forward refits (12 months x 10 models)...", flush=True)
    oos = {}
    for m_start in pd.date_range(EVAL_START, EVAL_END, freq="MS"):
        m_end = m_start + pd.offsets.MonthEnd(0)
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        for name, (cols, use_pca) in FEATURE_SETS_LBL.items():
            oos.setdefault(name, {})[m_start] = month_posterior(train, month, cols, use_pca, 42)
    oos = {name: pd.concat(list(d.values())).sort_index() for name, d in oos.items()}

    # ---------------- OOS zoo
    proxy = (feat.loc[eval_idx, "VIX_close"] > 25).astype(int)
    rows = []
    switch_curves = {}
    switch_sharpes = {}
    leads = {}
    f1s = {}
    for name in MODEL_ORDER:
        p_bull = oos[name].loc[eval_idx]
        states = np.where(p_bull >= 0.5, 0, 1)
        for strat in STRATEGIES:
            r, turnover = strategy_return(strat, p_bull, states, rets, feat["realized_vol"])
            s = perf_stats(r, f"{name}|{strat}")
            rows.append({
                "model": name, "strategy": strat, "oos": True,
                "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                "sharpe": s["sharpe"], "max_drawdown": max_drawdown(r),
                "calmar": s["calmar"], "turnover_daily": turnover,
            })
            if strat == "switch":
                switch_curves[name] = r
                switch_sharpes[name] = s["sharpe"]
                leads[name] = bear_entry_lead(1.0 - p_bull, SVB_REF_DATE, SVB_WINDOW_START)
                flag = (p_bull < 0.5).astype(int)
                f1s[name] = stress_f1(proxy.values, flag.values)
        print(f"[labeler-zoo] OOS {name} done", flush=True)
    for label, r in [("buy_hold_spy", bh), ("static_60_40", bt.static_60_40(rets, cost_bps=COST_BPS).loc[eval_idx])]:
        s = perf_stats(r, label)
        rows.append({"model": "BENCHMARK", "strategy": label, "oos": True,
                     "ann_return": s["ann_return"], "ann_vol": s["ann_vol"], "sharpe": s["sharpe"],
                     "max_drawdown": max_drawdown(r), "calmar": s["calmar"],
                     "turnover_daily": np.nan})
    oos_df = pd.DataFrame(rows)
    oos_df.to_csv(f"{TABLE_DIR}/labeler_zoo_2023_oos.csv", index=False)

    # ---------------- in-sample companion (single fit, disclosed)
    print("[labeler-zoo] in-sample companion fit...", flush=True)
    models_is = fit_regime_models(feat, feature_sets=FEATURE_SETS_LBL, random_state=42)
    is_rows = []
    for name, res in models_is.items():
        p_bull = res["posterior_bull"].loc[eval_idx]
        states = np.asarray(res["states"])[feat.index.get_indexer(eval_idx)]
        for strat in STRATEGIES:
            r, turnover = strategy_return(strat, p_bull, states, rets, feat["realized_vol"])
            s = perf_stats(r, f"{name}|{strat}")
            is_rows.append({"model": name, "strategy": strat, "oos": False,
                            "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                            "sharpe": s["sharpe"], "max_drawdown": max_drawdown(r),
                            "calmar": s["calmar"], "turnover_daily": turnover})
    pd.DataFrame(is_rows).to_csv(f"{TABLE_DIR}/labeler_zoo_insample.csv", index=False)

    # ---------------- significance family (OOS switch returns)
    print("[labeler-zoo] significance battery...", flush=True)
    pairs = [("M9_lbl_hmm", "M1_price_hmm"), ("M9_lbl_hmm", "M3_sent_hmm"),
             ("M9_lbl_hmm", "M7_fingpt_hmm"), ("M3_sent_hmm", "M1_price_hmm"),
             ("M9_lbl_hmm", None)]
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
    holm = holm_correction(raw_ps)
    for r_, h in zip(sig_rows, holm):
        r_["holm_p"] = h
    sig_df = pd.DataFrame(sig_rows)

    def _mde_ann(row):
        se = abs(row["sr_diff_daily"] / row["jk_z"]) if abs(row["jk_z"]) > 1e-12 else np.nan
        return float(2.80 * se * np.sqrt(252)) if np.isfinite(se) and se > 0 else np.nan

    sig_df["mde_ann_80pct"] = sig_df.apply(_mde_ann, axis=1)
    sig_df.to_csv(f"{TABLE_DIR}/labeler_significance.csv", index=False)

    # ---------------- diagnostics JSON
    for name, res in models_is.items():
        if name in ("M9_lbl_hmm", "M10_lbl_pca_hmm"):
            diag[f"transmat::{name}"] = res["model"].transmat_.tolist()
            diag[f"converged::{name}"] = bool(res["model"].monitor_.converged)
    pb = {n: res["posterior_bull"].loc[eval_idx] for n, res in models_is.items()}
    oos_pb = {n: oos[n].loc[eval_idx] for n in MODEL_ORDER}
    for a, b in [("M9_lbl_hmm", "M3_sent_hmm"), ("M9_lbl_hmm", "M7_fingpt_hmm"),
                 ("M9_lbl_hmm", "M1_price_hmm"), ("M3_sent_hmm", "M7_fingpt_hmm")]:
        diag[f"posterior_corr::{a}::{b}"] = round(float(pb[a].corr(pb[b])), 4)
        # same pair on the OOS walk-forward paths (audit NOTE-10: diagnostics
        # should not come only from the in-sample fit)
        diag[f"posterior_corr_oos::{a}::{b}"] = round(float(oos_pb[a].corr(oos_pb[b])), 4)
    diag["svb_lead_days"] = {n: (leads[n]["lead_days"]) for n in leads}
    diag["svb_lead_carried_in"] = {n: bool(leads[n]["carried_in"]) for n in leads}
    diag["stress_f1_vix25"] = {n: round(float(f1s[n]), 3) for n in f1s}
    with open(f"{TABLE_DIR}/labeler_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # ---------------- figures
    plot_labeler_correlation(daily_series.loc[EVAL_START:EVAL_END])
    curves = {"Buy & Hold SPY": bh, **switch_curves}
    sharpes = {"Buy & Hold SPY": perf_stats(bh)["sharpe"], **switch_sharpes}
    plot_labeler_zoo(curves, sharpes)

    print(oos_df.sort_values("sharpe", ascending=False).head(20).round(3).to_string(index=False), flush=True)
    print("\nSIGNIFICANCE (OOS switch):\n", sig_df.round(4).to_string(index=False), flush=True)
    print("\nSVB lead / F1:", {n: (leads[n]["lead_days"], round(f1s[n], 2)) for n in leads}, flush=True)
    print("LABELER_ZOO_DONE", flush=True)


if __name__ == "__main__":
    main()
