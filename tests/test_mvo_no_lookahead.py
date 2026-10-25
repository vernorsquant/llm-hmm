"""MVO no-lookahead regression tests: weights at day t must not depend on any
information from day t or later (posteriors, returns, or regime moments)."""

import numpy as np
import pandas as pd


def _setup(n=80, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = pd.DataFrame({"SPY": rng.normal(0.0004, 0.01, n), "IEF": rng.normal(0.0001, 0.002, n)}, index=idx)
    states = np.where(rng.random(n) < 0.5, 0, 1)
    p_bull = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    return idx, p_bull, states, rets


class TestMVONoLookahead:
    def test_future_return_perturbation_does_not_change_past_weights(self):
        from hmm_sentiment.backtest import run_mvo

        idx, p_bull, states, rets = _setup()
        r1, w1 = run_mvo(p_bull, rets, states)
        rets2 = rets.copy()
        rets2.iloc[70:] += 0.05  # shock the FUTURE
        r2, w2 = run_mvo(p_bull, rets2, states)
        assert np.allclose(w1.iloc[:60], w2.iloc[:60], atol=1e-9), \
            "MVO weights before the shock must not depend on future returns"

    def test_posterior_perturbation_affects_only_next_day_onward(self):
        from hmm_sentiment.backtest import mvo_weight_series

        idx = pd.bdate_range("2023-01-02", periods=80)
        # deterministic low posterior: blend sits firmly in the IEF-favored corner
        p_bull = pd.Series(0.2, index=idx)
        moments = {
            0: (np.array([0.20, -0.02]), np.array([[0.04, 0.0005], [0.0005, 0.004]])),
            1: (np.array([-0.02, 0.06]), np.array([[0.04, 0.0005], [0.0005, 0.004]])),
        }
        w1 = mvo_weight_series(p_bull, moments=moments)
        p2 = p_bull.copy()
        p2.iloc[50] = 0.99  # cross the corner on day 50
        w2 = mvo_weight_series(p2, moments=moments)
        assert np.allclose(w1.iloc[:50], w2.iloc[:50], atol=1e-9)
        assert not np.allclose(w1.iloc[50:], w2.iloc[50:], atol=1e-6)
        assert w1.iloc[:50]["IEF"].gt(0.9).all()  # IEF corner before
        assert w2.iloc[51]["SPY"] > 0.9  # SPY corner exactly on the day after the flip
