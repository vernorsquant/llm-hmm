"""Backtesting: binary regime switch, regime-conditioned MVO, benchmarks, costs.

Timing convention (no lookahead): the regime signal known at the close of day
t-1 determines the position held during day t. Transaction costs of 5 bps are
charged per unit of one-sided turnover: cost_t = 5e-4 * sum(|w_t - w_{t-1}|.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import COST_BPS, TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------- signals

def positions_from_signal(p_bull):
    """Position for day t from posterior known at t-1: 1.0 = 100% SPY, 0.0 = 100% IEF.

    NaN posteriors (warmup) propagate backward-fill of last known position;
    leading NaNs -> 0.0 (start defensively in cash-like IEF).
    """
    sig = (p_bull >= 0.5).astype(float)
    sig[p_bull.isna()] = np.nan
    pos = sig.shift(1)
    return pos.fillna(0.0)


def strategy_returns_from_positions(pos, asset_returns, weights):
    """Gross day-t return = sum_i w_i(t) * r_i(t).

    `weights`: dict asset -> scalar or Series aligned to pos.index.
    """
    out = pd.Series(0.0, index=pos.index)
    for asset, w in weights.items():
        r = asset_returns[asset].reindex(pos.index).fillna(0.0)
        if isinstance(w, pd.Series):
            w = w.reindex(pos.index).fillna(0.0)
        out = out + w * r
    return out


def transaction_cost(w_prev, w_new, cost_bps=COST_BPS):
    """cost = cost_bps/1e4 * sum(|w_new - w_prev|) (one-sided turnover)."""
    w_prev = np.asarray(w_prev, dtype=float)
    w_new = np.asarray(w_new, dtype=float)
    return (cost_bps / 1e4) * float(np.abs(w_new - w_prev).sum())


# ---------------------------------------------------------------- MVO

def blend_moments(mu0, cov0, mu1, cov1, p):
    """mu_blend = p*mu_0 + (1-p)*mu_1 ; Sigma_blend likewise."""
    mu_b = p * np.asarray(mu0, float) + (1.0 - p) * np.asarray(mu1, float)
    s_b = p * np.asarray(cov0, float) + (1.0 - p) * np.asarray(cov1, float)
    return mu_b, s_b


def solve_max_sharpe_weights(mu, cov, cost_bps=None):
    """Long-only, fully-invested max-Sharpe (rf=0) for 2 assets via SLSQP.

    Robust to singular covariance (adds a small ridge). Returns w (len 2).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 1e-10:
        cov = cov + np.eye(len(mu)) * (max(1e-6, 0.01 * abs(eigvals.max()) if eigvals.max() > 0 else 1e-6))

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -ret / vol

    res = minimize(
        neg_sharpe,
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(mu),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    if not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-6:
        w = np.array([0.5, 0.5])
    else:
        w = w / w.sum()
    return w


def regime_conditioned_moments(asset_returns, states, ann=TRADING_DAYS_PER_YEAR):
    """mu_k (annualized mean vector) and Sigma_k (annualized covariance) per state."""
    rets = np.asarray(asset_returns, dtype=float)
    states = np.asarray(states)
    moments = {}
    for k in np.unique(states):
        if k < 0:
            continue
        X = rets[states == k]
        mu = X.mean(axis=0) * ann
        cov = np.cov(X, rowvar=False) * ann
        cov = np.atleast_2d(cov) + np.eye(cov.shape[0]) * 1e-10
        moments[int(k)] = (mu, cov)
    return moments


def mvo_weight_series(p_bull, moments):
    """Daily MVO weights from lagged blended moments. Returns (weights_df, w_prev0)."""
    mu0, s0 = moments[0]
    mu1, s1 = moments[1]
    idx = p_bull.index
    rows = np.zeros((len(idx), 2))
    w_prev = np.array([0.0, 1.0])  # start defensively
    for i, (t, p) in enumerate(p_bull.items()):
        if np.isnan(p):
            w = w_prev.copy()
        else:
            mu_b, s_b = blend_moments(mu0, s0, mu1, s1, float(p))
            w = solve_max_sharpe_weights(mu_b, s_b)
        rows[i] = w
        w_prev = w
    return pd.DataFrame(rows, index=idx, columns=["SPY", "IEF"])


# ---------------------------------------------------------------- backtests

def run_binary_switch(p_bull, rets, cost_bps=COST_BPS):
    """Regime 0 -> 100% SPY; Regime 1 -> 100% IEF. Returns net daily returns."""
    pos = positions_from_signal(p_bull)
    gross = pos * rets["SPY"].reindex(pos.index).fillna(0.0) + (1.0 - pos) * rets["IEF"].reindex(pos.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = (cost_bps / 1e4) * turnover
    return gross - cost, pos


def run_mvo(p_bull, rets, states, cost_bps=COST_BPS):
    """Regime-conditioned blended MVO. `states` over rets index. Net daily returns."""
    moments = regime_conditioned_moments(rets[["SPY", "IEF"]].values, states)
    weights = mvo_weight_series(p_bull, moments)
    gross = (weights["SPY"] * rets["SPY"].reindex(weights.index).fillna(0.0)
             + weights["IEF"] * rets["IEF"].reindex(weights.index).fillna(0.0))
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = (cost_bps / 1e4) * dw
    return gross - cost, weights


def buy_and_hold(rets, asset="SPY"):
    return rets[asset].fillna(0.0).copy()


def static_60_40(rets, cost_bps=COST_BPS):
    """Monthly-rebalanced 60/40 with transaction costs."""
    w_spy = pd.Series(0.6, index=rets.index)
    months = rets.index.to_period("M")
    is_rebal = pd.Series(months != pd.Series(months).shift(1)).values
    w_spy_prev = w_spy.shift(1).fillna(0.6)
    w_iec_prev = 1.0 - w_spy_prev
    gross = 0.6 * rets["SPY"].fillna(0.0) + 0.4 * rets["IEF"].fillna(0.0)
    turn = np.where(is_rebal, np.abs(0.6 - w_spy_prev) + np.abs(0.4 - w_iec_prev), 0.0)
    cost = pd.Series(turn, index=rets.index) * (cost_bps / 1e4)
    return gross - cost
