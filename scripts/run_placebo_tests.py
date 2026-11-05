"""Placebo tests: is the FinBERT regime signal real?

A. Daily-shuffle permutation (N=60): permute the daily FinBERT feature rows
   across dates (preserving each day's values and intra-day structure,
   destroying date alignment), rerun the identical monthly walk-forward, and
   build the null distribution of the switch Sharpe.
   p = (1 + #{perm >= real}) / (N + 1).
B. Time-shift placebo: push every article's mapped trading date 5 sessions
   forward (news arrives after the trades). The edge must collapse; if it
   doesn't, the pipeline leaks.

Methodology identical to the real run: monthly prior-only refits, causal
filtered posteriors, t-1 -> t positioning, one-sided 5 bps costs.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, TABLE_DIR
from hmm_sentiment.daily_features import build_daily_news_features, map_articles_to_trading_days
from hmm_sentiment.finbert_scoring import score_articles
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import perf_stats
from hmm_sentiment.news_corpus import load_corpus
from run_extended_zoo import EXT_CSV, MODELS, START, build_extended_features

N_PERM = int(os.getenv("PLACEBO_N", "30"))
SHIFT_DAYS = 5
M3_COLS = MODELS["M3_finbert_hmm"][0]


def _monthly_walkforward(feat, rets, months):
    """Return the causal walk-forward switch return series for M3 columns."""
    oos = {}
    for m_start in months:
        m_end = m_start + pd.offsets.MonthEnd(0)
        train = feat[feat.index < m_start]
        month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
        if len(month) == 0 or len(train) < 120:
            continue
        from run_long_backtest import month_posterior
        post = month_posterior(train, month, M3_COLS, False, 42)
        oos.setdefault(m_start, post)
    if not oos:
        return None
    return pd.concat(list(oos.values())).sort_index()


def _switch_sharpe(p_bull, rets):
    r, _ = bt.run_binary_switch(p_bull, rets, cost_bps=COST_BPS)
    return float(perf_stats(r)["sharpe"]), r


def _perm_worker(args):
    seed, feat, months, rets = args
    rng = np.random.default_rng(seed)
    feat_p = feat.copy()
    cols = ["sent_w", "sent_w_ema5", "sent_disp"]
    vals = feat[cols].values
    perm_idx = rng.permutation(len(feat))
    feat_p[cols] = vals[perm_idx]
    # rebuild EMA consistently with the permuted level (EMA of the permuted level)
    feat_p["sent_w_ema5"] = feat_p["sent_w"].ewm(span=5, adjust=False).mean()
    p = _monthly_walkforward(feat_p, rets, months)
    if p is None or p.dropna().empty:
        return seed, np.nan
    return seed, _switch_sharpe(p, rets)[0]


def main():
    market, feat, _ = build_extended_features()
    rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    months = [m for m in pd.date_range("2017-02-01", feat.index.max(), freq="MS")
              if feat[feat.index < m].shape[0] >= 120]
    print(f"[placebo] {len(months)} walk-forward months", flush=True)

    # real benchmark
    p_real = _monthly_walkforward(feat, rets, months)
    sharpe_real, r_real = _switch_sharpe(p_real, rets)
    print(f"[placebo] REAL FinBERT switch Sharpe: {sharpe_real:.4f}", flush=True)

    # A. daily-shuffle permutations (parallel)
    jobs = [(42 + i, feat, months, rets) for i in range(N_PERM)]
    with Pool(min(6, os.cpu_count() or 4)) as pool:
        results = pool.map(_perm_worker, jobs)
    perm_sharpes = [s for _, s in results if np.isfinite(s)]
    n_ge = sum(1 for s in perm_sharpes if s >= sharpe_real)
    p_perm = (1 + n_ge) / (1 + len(perm_sharpes))

    # B. time-shift +5 sessions
    shifted = feat.copy()
    shift = pd.Series(shifted.index, index=shifted.index).shift(-SHIFT_DAYS)
    valid = shift.notna()
    shifted.index = pd.DatetimeIndex(shift.loc[valid].values)
    p_shift = _monthly_walkforward(shifted, rets, months)
    sharpe_shift = _switch_sharpe(p_shift, rets)[0] if p_shift is not None and not p_shift.dropna().empty else np.nan

    out = {
        "real_sharpe": round(sharpe_real, 4),
        "n_permutations": len(perm_sharpes),
        "perm_sharpe_mean": round(float(np.mean(perm_sharpes)), 4),
        "perm_sharpe_p95": round(float(np.quantile(perm_sharpes, 0.95)), 4),
        "perm_p_value": round(p_perm, 4),
        "shifted5d_sharpe": round(float(sharpe_shift), 4) if np.isfinite(sharpe_shift) else None,
        "real_ann_return": round(float(perf_stats(r_real)["ann_return"]), 4),
    }
    with open(f"{TABLE_DIR}/placebo_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame({"perm_seed": [s for s, _ in results], "perm_sharpe": perm_sharpes}).to_csv(
        f"{TABLE_DIR}/placebo_permutations.csv", index=False)
    print("[placebo] RESULT:", json.dumps(out), flush=True)
    print("PLACEBO_DONE", flush=True)


if __name__ == "__main__":
    main()
