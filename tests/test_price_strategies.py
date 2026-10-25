"""Price-only zoo tests: lag correctness and bounds."""

import numpy as np
import pandas as pd


def _close(n=260, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(400 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)
    return idx, close


class TestPriceStrategies:
    def test_sma_cross_lagged_and_bounded(self):
        from hmm_sentiment.price_strategies import sma_cross_positions

        idx, close = _close()
        pos = sma_cross_positions(close, short=10, long=40)
        assert (pos >= 0).all() and (pos <= 1).all()
        assert pos.iloc[:40].eq(0.0).all()  # NaN warmup -> flat
        # no lookahead: recompute signal at t and compare position at t+1
        sma_s, sma_l = close.rolling(10).mean(), close.rolling(40).mean()
        raw = (sma_s > sma_l).astype(float)
        assert (pos.iloc[41:] == raw.shift(1).iloc[41:]).all()

    def test_momentum_lagged(self):
        from hmm_sentiment.price_strategies import momentum_positions

        idx, close = _close()
        pos = momentum_positions(close, lookback=60, skip=10)
        assert (pos >= 0).all() and (pos <= 1).all()
        assert pos.iloc[:10].eq(0.0).all()

    def test_vol_target_flat_when_rv_missing(self):
        from hmm_sentiment.price_strategies import vol_target_positions

        idx, close = _close()
        rv = pd.Series(np.nan, index=idx)
        assert vol_target_positions(rv).eq(0.0).all()
