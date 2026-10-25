"""Price-only strategy zoo: NO HMM, NO sentiment. SMA crossovers, 12-1
momentum, vol-targeted B&H, plain B&H and 60/40 — on 2023 and 2015-2023.

These strategies have no fitted parameters (no in-sample optimism possible),
so they are the honest floor the HMM/sentiment complex must beat.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment import backtest as bt
from hmm_sentiment.config import COST_BPS, EVAL_END, TABLE_DIR
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.metrics import max_drawdown, perf_stats
from hmm_sentiment.price_strategies import (
    momentum_positions,
    sma_cross_positions,
    vol_target_positions,
)
from hmm_sentiment.stats_tests import bootstrap_sharpe_diff, jk_memmel_sharpe_diff

WARMUP_START = "2010-01-01"
LONG_START = "2015-01-01"


def run():
    market = fetch_market_data(start=WARMUP_START, end=EVAL_END, use_cache=True)
    rets = market[["SPY_ret", "IEF_ret"]].rename(columns={"SPY_ret": "SPY", "IEF_ret": "IEF"})
    close = market["SPY_close"]

    strategies = {
        "SMA 10/40 cross": sma_cross_positions(close, 10, 40),
        "SMA 50/200 cross": sma_cross_positions(close, 50, 200),
        "12-1 momentum": momentum_positions(close, 252, 21),
        "Vol-target 10% (no model)": vol_target_positions(market["realized_vol"], 0.10),
    }

    rows = []
    for window_name, widx in [
        ("2023", market.loc["2023-01-01":EVAL_END].index),
        ("2015-2023", market.loc[LONG_START:EVAL_END].index),
    ]:
        bh = bt.buy_and_hold(rets).loc[widx]
        b6040 = bt.static_60_40(rets, cost_bps=COST_BPS).loc[widx]
        for name, pos in strategies.items():
            pos_w = pos.loc[widx]
            gross = pos_w * rets["SPY"].loc[widx].fillna(0) + (1 - pos_w) * rets["IEF"].loc[widx].fillna(0)
            net = gross - (COST_BPS / 1e4) * pos_w.diff().abs().fillna(0.0)
            s = perf_stats(net, name)
            s.update({"window": window_name, "strategy": name, "max_drawdown": max_drawdown(net),
                      "trades": int(pos_w.diff().abs().gt(0).sum())})
            rows.append(s)
        for name, r in [("Buy & Hold SPY", bh), ("60/40 Static", b6040)]:
            s = perf_stats(r, name)
            s.update({"window": window_name, "strategy": name, "max_drawdown": max_drawdown(r), "trades": 0})
            rows.append(s)

        # significance vs B&H for the two crossover rules (same window)
        for name, pos in list(strategies.items())[:3]:
            pos_w = pos.loc[widx]
            gross = pos_w * rets["SPY"].loc[widx].fillna(0) + (1 - pos_w) * rets["IEF"].loc[widx].fillna(0)
            net = gross - (COST_BPS / 1e4) * pos_w.diff().abs().fillna(0.0)
            jk = jk_memmel_sharpe_diff(net.values, bh.values)
            boot = bootstrap_sharpe_diff(net.values, bh.values, n_boot=5000, block=10, seed=7)
            rows.append({"window": window_name, "strategy": f"SIG {name} vs B&H",
                         "ann_return": None, "ann_vol": None, "sr_diff_daily": jk["sr_diff"],
                         "max_drawdown": None, "calmar": None, "n_days": jk["n"],
                         "trades": None, "jk_p": jk["p_value"], "boot_p": boot["p_value"]})

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLE_DIR}/price_only_zoo.csv", index=False)
    show = [c for c in ["window", "strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown", "trades", "jk_p", "boot_p"] if c in df.columns]
    print(df[show].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
