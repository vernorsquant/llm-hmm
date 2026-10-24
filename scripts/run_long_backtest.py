"""Longer-lookback study for the price-only HMM baseline (M1/M2).

Extends the evaluation to 2015-2023 (~2,260 trading days) with monthly
walk-forward refits, so Sharpe-difference tests have real statistical power.
Sentiment models are excluded here: the GDELT corpus covers 2023 only.

Outputs: outputs/tables/long_backtest_years.csv (per-year OOS Sharpe),
outputs/tables/long_backtest_summary.csv (overall IS/OOS/benchmarks),
outputs/tables/long_backtest_significance.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, EVAL_END, TABLE_DIR
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import max_drawdown, perf_stats
from hmm_sentiment.regime_models import FEATURE_SETS, _fit_one
from hmm_sentiment.stats_tests import (
    bootstrap_sharpe_diff,
    jk_memmel_sharpe_diff,
    stress_f1,
)
from run_stats import month_posterior

LONG_START = "2015-01-01"
WARMUP_START = "2010-01-01"
TRAIN_LOOKBACKS = {"full": None, "5y": 5 * 252, "2y": 2 * 252}


def switch_perf(p_bull, rets, eval_idx):
    r, _ = bt.run_binary_switch(p_bull.loc[eval_idx], rets, cost_bps=COST_BPS)
    return r


def main():
    market = fetch_market_data(start=WARMUP_START, end=EVAL_END, use_cache=True)
    feat = market.copy()  # price-only features
    eval_idx = feat.loc[LONG_START:EVAL_END].index
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    print(f"[long] rows={len(feat)} eval days={len(eval_idx)}", flush=True)

    months = pd.date_range(LONG_START, EVAL_END, freq="MS")
    cols, use_pca = FEATURE_SETS["M1_price_hmm"]

    results = {}
    for lookback_name, lookback_days in TRAIN_LOOKBACKS.items():
        oos_parts = {}
        for m_start in months:
            m_end = m_start + pd.offsets.MonthEnd(0)
            train = feat[feat.index < m_start]
            if lookback_days is not None:
                train = train.iloc[-lookback_days:]
            if len(train) < 252:
                continue
            month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
            post = month_posterior(train, month, cols, use_pca, random_state=42)
            oos_parts[m_start] = post
        oos = pd.concat(list(oos_parts.values())).sort_index()
        r = switch_perf(oos, rets, eval_idx)
        results[lookback_name] = r
        s = perf_stats(r, f"M1_OOS_{lookback_name}")
        print(f"[long] lookback={lookback_name}: sharpe={s['sharpe']:.3f} "
              f"ann_ret={s['ann_return']:.3f} maxDD={s['max_drawdown']:.3f}", flush=True)

    # in-sample (full-window fit)
    p_is = _fit_one(feat, cols, use_pca, 42)["posterior_bull"]
    r_is = switch_perf(p_is, rets, eval_idx)
    bh = bt.buy_and_hold(rets).loc[eval_idx]
    bh6040 = bt.static_60_40(rets, cost_bps=COST_BPS).loc[eval_idx]

    rows = []
    for label, r in [("M1_OOS_lookback_full", results["full"]),
                     ("M1_OOS_lookback_5y", results["5y"]),
                     ("M1_OOS_lookback_2y", results["2y"]),
                     ("M1_IS_fullsample", r_is),
                     ("Buy & Hold SPY", bh),
                     ("60/40 Static", bh6040)]:
        s = perf_stats(r, label)
        s["max_drawdown"] = max_drawdown(r)
        rows.append(s)
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{TABLE_DIR}/long_backtest_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    # per-year OOS Sharpe
    yearly = []
    for year in range(2015, 2024):
        yidx = eval_idx[eval_idx.year == year]
        row = {"year": year}
        for label, r in [("M1_OOS_full", results["full"]), ("B&H", bh), ("60/40", bh6040)]:
            row[label] = round(float(r[yidx].mean() / r[yidx].std() * np.sqrt(252)), 2) if r[yidx].std() > 0 else np.nan
        yearly.append(row)
    years_df = pd.DataFrame(yearly)
    years_df.to_csv(f"{TABLE_DIR}/long_backtest_years.csv", index=False)
    print(years_df.to_string(index=False), flush=True)

    # significance with real power (~2260 obs)
    rows_sig = []
    for label, r in [("M1_OOS_lookback_full", results["full"]),
                     ("M1_OOS_lookback_5y", results["5y"]),
                     ("M1_OOS_lookback_2y", results["2y"]),
                     ("M1_IS_fullsample", r_is)]:
        jk = jk_memmel_sharpe_diff(r.values, bh.values)
        boot = bootstrap_sharpe_diff(r.values, bh.values, n_boot=5000, block=10, seed=7)
        rows_sig.append({"strategy": label, "sr_diff_vs_bh": jk["sr_diff"],
                         "jk_z": jk["z"], "jk_p": jk["p_value"],
                         "boot_p": boot["p_value"],
                         "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"]})
        print(f"[sig] {label}: sr_diff={jk['sr_diff']:.4f} jk_p={jk['p_value']:.4f} boot_p={boot['p_value']:.4f}", flush=True)
    pd.DataFrame(rows_sig).to_csv(f"{TABLE_DIR}/long_backtest_significance.csv", index=False)

    # stress-flag quality over the long window (VIX > 25)
    proxy = (feat.loc[eval_idx, "VIX_close"] > 25).astype(int)
    print(f"[long] VIX>25 proxy days: {int(proxy.sum())} / {len(proxy)}", flush=True)
    print("LONG BACKTEST DONE", flush=True)


if __name__ == "__main__":
    main()
