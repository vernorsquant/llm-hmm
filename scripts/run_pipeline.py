"""End-to-end pipeline: data -> scoring -> features -> 4 models -> backtests -> artifacts."""

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
    FIG_DIR,
    SVB_REF_DATE,
    SVB_WINDOW_END,
    SVB_WINDOW_START,
    TABLE_DIR,
)
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.llm_importance import rate_importance
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import bear_entry_lead, perf_stats
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.plots import (
    plot_cumulative_returns,
    plot_mvo_weights,
    plot_regime_comparison,
    plot_transition_lead_time,
)
from hmm_sentiment.regime_models import FEATURE_SETS, fit_regime_models
from hmm_sentiment.vader_scoring import vader_scores

STEPS_LOG = f"{TABLE_DIR}/pipeline_log.json"


def log_step(msg):
    print(f"[pipeline] {msg}", flush=True)


def build_features():
    log_step("market data")
    market = fetch_market_data()
    log_step(f"market rows={len(market)} {market.index.min().date()} -> {market.index.max().date()}")

    log_step("news corpus")
    news = load_corpus()
    log_step(f"articles={len(news)}")

    log_step("FinBERT scoring")
    scored = score_articles(news)
    log_step("LLM importance rating")
    scored["importance"] = rate_importance(scored["headline"].tolist())

    log_step("VADER scoring (baseline)")
    v_scored = vader_scores(news)
    v_scored["importance"] = rate_importance(v_scored["headline"].tolist())  # all cached

    log_step("close-safeguard mapping + daily features")
    merged = map_articles_to_trading_days(scored, market.index)
    daily_news = build_daily_news_features(merged, market.index)

    v_merged = map_articles_to_trading_days(v_scored, market.index)
    v_daily = build_daily_news_features(v_merged, market.index)
    v_daily = v_daily.rename(
        columns={"sent_w": "sent_w_v", "sent_w_ema5": "sent_w_ema5_v", "sent_disp": "sent_disp_v"}
    )

    feat = market.join(daily_news, how="left")
    feat = feat.join(v_daily[["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]], how="left")
    log_step(f"features rows={len(feat)} cols={list(feat.columns)}")
    return market, feat, merged, scored


def run_all():
    market, feat, merged, scored = build_features()

    log_step("fitting 4 HMM models (in-sample, disclosed)")
    models = fit_regime_models(feat, random_state=42)

    eval_idx = feat.loc[EVAL_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})

    curves = {}
    stats_rows = []
    leads = {}
    mvo_weights = {}
    states_by_model = {}
    p_bull_by_model = {}

    bench_spy = bt.buy_and_hold(rets).loc[eval_idx]
    bench_6040 = bt.static_60_40(rets, cost_bps=COST_BPS).loc[eval_idx]
    curves["60/40"] = bench_6040

    stats_rows.append({"label": "Buy & Hold SPY", **perf_stats(bench_spy, "Buy & Hold SPY")})
    stats_rows.append({"label": "60/40 Static", **perf_stats(bench_6040, "60/40 Static")})

    for name, res in models.items():
        p_bull_full = res["posterior_bull"]
        p_bull_eval = p_bull_full.loc[eval_idx]
        p_bear_eval = 1.0 - p_bull_eval
        states_full = pd.Series(res["states"], index=feat.index)

        sw_ret, _ = bt.run_binary_switch(p_bull_eval, rets, cost_bps=COST_BPS)
        curves[f"{name} [Switch]"] = sw_ret

        moments = bt.regime_conditioned_moments(rets.values, res["states"])
        mvo_ret, w = bt.run_mvo(p_bull_eval, rets, res["states"], cost_bps=COST_BPS)
        curves[f"{name} [MVO]"] = mvo_ret
        mvo_weights[name] = w

        label = name
        stats_rows.append({"label": f"{label} [Switch]", **perf_stats(sw_ret, f"{label} [Switch]")})
        stats_rows.append({"label": f"{label} [MVO]", **perf_stats(mvo_ret, f"{label} [MVO]")})

        leads[name] = bear_entry_lead(
            p_bear_eval, SVB_REF_DATE, SVB_WINDOW_START
        )
        states_by_model[name] = states_full.loc[eval_idx]
        p_bull_by_model[name] = p_bull_eval
        log_step(f"{name}: switch sharpe={stats_rows[-2]['sharpe']:.2f} mvo sharpe={stats_rows[-1]['sharpe']:.2f} lead={leads[name]}")
    # ---- artifacts
    log_step("figures")
    plot_cumulative_returns(curves, bench_spy)
    plot_regime_comparison(feat.loc[eval_idx, "SPY_close"], states_by_model)
    plot_mvo_weights(
        mvo_weights["M2_price_pca_hmm"], mvo_weights["M4_sent_pca_hmm"],
        p_bull_by_model["M4_sent_pca_hmm"],
    )
    plot_transition_lead_time(leads)

    log_step("metrics table")
    metrics = pd.DataFrame(stats_rows)
    metrics["transition_lead_days"] = metrics["label"].map(
        {f"{n} [Switch]": leads[n]["lead_days"] for n in leads}
        | {f"{n} [MVO]": leads[n]["lead_days"] for n in leads}
    )
    metrics.to_csv(f"{TABLE_DIR}/metrics_summary.csv", index=False)

    # diagnostics: transition matrices + regime stats + cross-model agreement
    diag = {}
    pb = {n: res["posterior_bull"] for n, res in models.items()}
    for name, res in models.items():
        diag[name] = {
            "transmat": res["model"].transmat_.tolist(),
            "converged": bool(res["model"].monitor_.converged),
            "pca_explained_variance": res["pca_explained_variance"],
            "lead_days_svb": leads[name]["lead_days"],
            "lead_entry_date": str(leads[name]["entry_date"].date()) if leads[name]["entry_date"] is not None else None,
            "lead_carried_in": leads[name]["carried_in"],
            "bear_days_in_2023": int((states_by_model[name] == 1).sum()),
        }
    for a, b in [("M3_sent_hmm", "M5_vader_hmm"), ("M4_sent_pca_hmm", "M6_vader_pca_hmm"), ("M1_price_hmm", "M3_sent_hmm")]:
        if a in pb and b in pb:
            diag[f"posterior_corr::{a}::{b}"] = round(float(pb[a].corr(pb[b])), 4)
    with open(f"{TABLE_DIR}/model_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # scored articles sample for the README
    cols = ["timestamp_utc", "source", "headline", "S_raw", "confidence", "importance"]
    scored[cols].to_csv(f"{TABLE_DIR}/article_scores_sample.csv", index=False)

    log_step("DONE")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run_all()
