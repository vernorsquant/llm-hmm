"""Price-only strategies with NO HMM and NO sentiment: SMA crossovers,
12-1 momentum, vol-targeted buy-and-hold. All signals lagged by one day."""

import numpy as np
import pandas as pd


def sma_cross_positions(close, short=50, long=200):
    """1 when SMA(short) > SMA(long), else 0. Signal from t-1 close positions day t."""
    sma_s = close.rolling(short).mean()
    sma_l = close.rolling(long).mean()
    sig = (sma_s > sma_l).astype(float)
    sig[~np.isfinite(sma_s) | ~np.isfinite(sma_l)] = np.nan
    return sig.shift(1).fillna(0.0)


def momentum_positions(close, lookback=252, skip=21):
    """1 when trailing `lookback-skip` day return is positive (12-1 momentum style)."""
    mom = close.pct_change(lookback - skip)
    sig = (mom > 0).astype(float)
    sig[~np.isfinite(mom)] = np.nan
    return sig.shift(1).fillna(0.0)


def vol_target_positions(realized_vol, target_vol=0.10):
    """Equity exposure clipped at target/rv, lagged. No regime model at all."""
    scale = (target_vol / realized_vol.shift(1)).clip(0.0, 1.0)
    return scale.fillna(0.0)
