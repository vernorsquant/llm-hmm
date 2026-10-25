"""Significance: walk-forward price-HMM OOS vs the SMA 10/40 rule (2015-2023)."""
import os
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.price_strategies import sma_cross_positions
from hmm_sentiment.regime_models import FEATURE_SETS
from hmm_sentiment.stats_tests import bootstrap_sharpe_diff, jk_memmel_sharpe_diff
from run_long_backtest import month_posterior

market = fetch_market_data(start="2010-01-01", end="2023-12-31")
feat = market.copy()
rets = feat[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
eval_idx = feat.loc["2015-01-01":"2023-12-31"].index
cols, use_pca = FEATURE_SETS["M1_price_hmm"]

parts = {}
for m_start in pd.date_range("2015-01-01", "2023-12-01", freq="MS"):
    m_end = m_start + pd.offsets.MonthEnd(0)
    train = feat[feat.index < m_start]
    month = feat[(feat.index >= m_start) & (feat.index <= m_end)]
    parts[m_start] = month_posterior(train, month, cols, use_pca, 42)
p_oos = pd.concat(list(parts.values())).sort_index()

r_hmm, _ = bt.run_binary_switch(p_oos.loc[eval_idx], rets, cost_bps=COST_BPS)
pos_sma = sma_cross_positions(market["SPY_close"], 10, 40).loc[eval_idx]
r_sma = (pos_sma * rets["SPY"].loc[eval_idx].fillna(0)
         + (1 - pos_sma) * rets["IEF"].loc[eval_idx].fillna(0)
         - (COST_BPS / 1e4) * pos_sma.diff().abs().fillna(0))

jk = jk_memmel_sharpe_diff(r_hmm.values, r_sma.values)
boot = bootstrap_sharpe_diff(r_hmm.values, r_sma.values, n_boot=5000, block=10, seed=7)
print(f"HMM-OOS vs SMA10/40 (2015-2023): sr_diff={jk['sr_diff']:.4f} daily | "
      f"jk_p={jk['p_value']:.4f} | boot_p={boot['p_value']:.4f} | "
      f"CI=({boot['ci_low']:.4f}, {boot['ci_high']:.4f})")
