"""Performance metrics: annualized return/vol, Sharpe, max drawdown, Calmar,
and regime transition lead time."""

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def equity_curve(returns):
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns):
    eq = equity_curve(returns)
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def perf_stats(returns, label=""):
    r = returns.dropna()
    n = len(r)
    ann_ret = float((1.0 + r).prod() ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if r.std() > 0 else 0.0
    mdd = max_drawdown(r)
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    return {
        "label": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "n_days": n,
    }


def bear_entry_lead(p_bear, ref_date, window_start):
    """Crisis-entry timing: the LAST bull->bear entry (p_bear crosses 0.5)
    at or before `ref_date`, scanning from `window_start`.

    Returns {lead_days, entry_date, carried_in}:
      - genuine flip inside window: lead = trading days entry -> ref (>0 early)
      - already bear at window_start (no entry inside): carried_in=True,
        entry_date=None, lead = trading days window_start -> ref
      - never bear before ref: lead_days=None
    """
    p = p_bear.dropna()
    p = p[(p.index >= pd.Timestamp(window_start)) & (p.index <= pd.Timestamp(ref_date))]
    if p.empty:
        return {"lead_days": None, "entry_date": None, "carried_in": False}

    is_bear = p > 0.5
    if not is_bear.any():
        return {"lead_days": None, "entry_date": None, "carried_in": False}

    # fill with the first value so a bear-at-window-start day 0 is NOT
    # counted as an entry observed inside the window
    entries = is_bear & ~is_bear.shift(1, fill_value=bool(is_bear.iloc[0]))
    if entries.any():
        entry_date = entries[entries].index[-1]
        i_entry = p.index.get_loc(entry_date)
        i_ref = len(p) - 1  # ref is the last element by construction
        return {"lead_days": int(i_ref - i_entry), "entry_date": entry_date, "carried_in": False}

    # bear the whole window: carried in from before
    return {"lead_days": int(len(p) - 1), "entry_date": None, "carried_in": True}
