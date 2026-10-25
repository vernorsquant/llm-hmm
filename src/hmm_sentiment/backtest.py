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


def mvo_weight_series(p_bull, moments=None, rets=None, states=None,
                      ann=TRADING_DAYS_PER_YEAR, min_state_obs=10):
    """Daily MVO weights, strictly no lookahead.

    Weights for day t come from the posterior known at close t-1 (p_bull is
    shifted one day). Regime-conditioned moments are estimated from an
    EXPANDING window of returns/states strictly before day t (never from the
    full sample). `moments` (precomputed) may be passed for the legacy
    full-sample path; if omitted, `rets`/`states` drive expanding moments.
    """
    idx = p_bull.index
    shifted = p_bull.shift(1)
    rows = np.zeros((len(idx), 2))
    w_prev = np.array([0.0, 1.0])  # start defensively
    have_moments = moments is not None
    cur_moments = moments
    rets_arr = np.asarray(rets, dtype=float) if rets is not None else None
    states_arr = np.asarray(states) if states is not None else None

    for i, t in enumerate(idx):
        p = shifted.iloc[i]
        if have_moments:
            cur_moments = moments
        elif rets_arr is not None and i >= 1:
            # expanding: use rows strictly before day t
            sub_rets = rets_arr[:i]
            sub_states = states_arr[:i]
            counts = {int(k): int((sub_states == k).sum()) for k in (0, 1) if (sub_states == k).sum() >= min_state_obs}
            if len(counts) == 2:
                try:
                    cur_moments = regime_conditioned_moments(sub_rets, sub_states)
                except Exception:
                    pass  # keep last valid moments
        if cur_moments is not None and not np.isnan(p):
            mu_b, s_b = blend_moments(cur_moments[0][0], cur_moments[0][1],
                                      cur_moments[1][0], cur_moments[1][1], float(p))
            w = solve_max_sharpe_weights(mu_b, s_b)
        else:
            w = w_prev.copy()
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


def run_mvo(p_bull, rets, states, cost_bps=COST_BPS, expanding=True):
    """Regime-conditioned blended MVO, strictly no lookahead: weights for day t
    use the posterior from t-1 and moments estimated from an expanding window
    of returns/states strictly before t (never full-sample). Net daily returns.

    Cost convention (one-sided, consistent with run_binary_switch and
    weight_net_returns): 5 bps x |delta w_SPY| per day. w_IEF = 1 - w_SPY, so
    charging both legs would double-count a fully-invested two-asset switch.
    """
    if expanding:
        weights = mvo_weight_series(p_bull, rets=rets[["SPY", "IEF"]].values, states=states)
    else:
        moments = regime_conditioned_moments(rets[["SPY", "IEF"]].values, states)
        weights = mvo_weight_series(p_bull, moments=moments)
    gross = (weights["SPY"] * rets["SPY"].reindex(weights.index).fillna(0.0)
             + weights["IEF"] * rets["IEF"].reindex(weights.index).fillna(0.0))
    dw = weights["SPY"].diff().abs().fillna(weights["SPY"].abs())
    cost = (cost_bps / 1e4) * dw
    return gross - cost, weights


def buy_and_hold(rets, asset="SPY"):
    return rets[asset].fillna(0.0).copy()


def static_60_40(rets, cost_bps=COST_BPS):
    """Monthly-rebalanced 60/40 with transaction costs on drifted weights.

    Tracks the drifted equity weight between rebalances and charges the true
    trade size at each month-end rebalance (5 bps per unit of one-sided
    turnover, matching every other strategy in this repo).
    """
    eq_w = 0.6
    gross = np.empty(len(rets))
    costs = np.zeros(len(rets))
    w_eq = eq_w  # current drifted equity weight
    spy = rets["SPY"].fillna(0.0).values
    ief = rets["IEF"].fillna(0.0).values
    months = rets.index.to_period("M")
    prev_month = months[0]
    for i in range(len(rets)):
        if months[i] != prev_month:  # month-end -> rebalance to target
            trade = abs(eq_w - w_eq)
            costs[i] = (cost_bps / 1e4) * trade
            w_eq = eq_w
            prev_month = months[i]
        gross[i] = w_eq * spy[i] + (1.0 - w_eq) * ief[i]
        w_eq = w_eq * (1.0 + spy[i]) / (1.0 + w_eq * spy[i] + (1.0 - w_eq) * ief[i])
    return pd.Series(gross - costs, index=rets.index)


# ------------------------------------------------- strategy zoo additions

def prob_weighted_weights(p_bull):
    """Probability-weighted allocation: w_SPY(t) = p_bull(t-1), continuous.

    No thresholding; exposure scales linearly with regime confidence.
    """
    w = p_bull.shift(1)
    return w.fillna(0.0).clip(0.0, 1.0)


def hysteresis_positions(p_bull, enter_bull=0.6, exit_bull=0.4):
    """Switch with hysteresis: enter bull at p > enter_bull, exit at p < exit_bull.

    Reduces whipsaw in choppy posterior regimes (state persists in the
    dead-band). Still fully lagged (position for day t from p at t-1).
    """
    sig = p_bull.shift(1)
    pos = np.zeros(len(sig))
    cur = 0.0
    vals = sig.values
    for i, p in enumerate(vals):
        if np.isnan(p):
            pos[i] = 0.0
            cur = 0.0
            continue
        if cur == 0.0 and p > enter_bull:
            cur = 1.0
        elif cur == 1.0 and p < exit_bull:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=p_bull.index)


def vol_target_switch_weights(p_bull, realized_vol, target_vol=0.10):
    """Vol-targeted exposure: w_SPY(t) = p_bull(t-1) * clip(target / rv(t-1), 0, 1).

    Scales equity exposure down when realized vol exceeds the target. rv must be
    lagged with the signal (both known at t-1).
    """
    lag_rv = realized_vol.shift(1)
    scale = (target_vol / lag_rv).clip(0.0, 1.0)
    w = (p_bull.shift(1) * scale)
    return w.fillna(0.0).clip(0.0, 1.0)
