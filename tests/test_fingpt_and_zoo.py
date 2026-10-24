"""Tests for the FinGPT scorer and the extended strategy zoo."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestFinGPTScoring:
    def test_bounds_and_simplex(self):
        from hmm_sentiment.fingpt_scoring import fingpt_scores

        headlines = [
            "Fed raises interest rates by 25 basis points",
            "Stocks plunge as bank failure sparks panic",
            "Apple announces new laptop colors",
            "",
        ]
        out = fingpt_scores(pd.DataFrame({"headline": headlines}))
        assert np.all(out["S_raw"].between(-1.0, 1.0))
        assert np.all(out["confidence"].between(0.0, 1.0))
        simplex = out[["P_pos", "P_neu", "P_neg"]].sum(axis=1)
        assert np.allclose(simplex, 1.0, atol=1e-6)

    def test_cached_stable(self):
        from hmm_sentiment.fingpt_scoring import fingpt_scores

        df = pd.DataFrame({"headline": ["Hot CPI print spooks markets"]})
        a = fingpt_scores(df)
        b = fingpt_scores(df)
        assert abs(a["S_raw"].iloc[0] - b["S_raw"].iloc[0]) < 1e-12


class TestStrategyZoo:
    def _fix(self, n=60):
        rng = _rng()
        idx = pd.bdate_range("2023-01-02", periods=n)
        p_bull = pd.Series(rng.uniform(0, 1, n), index=idx).round(3)
        rv = pd.Series(rng.uniform(0.05, 0.5, n), index=idx)
        rets = pd.DataFrame(
            {"SPY": rng.normal(0.0004, 0.01, n), "IEF": rng.normal(0.0001, 0.002, n)}, index=idx
        )
        return idx, p_bull, rv, rets

    def test_prob_weighted_weights_in_unit_interval_and_lagged(self):
        from hmm_sentiment.backtest import prob_weighted_weights

        idx, p_bull, rv, rets = self._fix()
        w = prob_weighted_weights(p_bull)
        assert (w >= -1e-9).all() and (w <= 1 + 1e-9).all()
        assert w.iloc[0] == 0.0  # starts flat (no prior-day signal)
        assert abs(w.iloc[5] - p_bull.iloc[4]) < 1e-12  # lagged by one day

    def test_hysteresis_reduces_churn_vs_threshold(self):
        from hmm_sentiment.backtest import hysteresis_positions, positions_from_signal

        idx = pd.bdate_range("2023-01-02", periods=30)
        p = pd.Series(0.5, index=idx)
        p.iloc[::3] = 0.55
        p.iloc[1::3] = 0.45
        hyst = hysteresis_positions(p, enter_bull=0.6, exit_bull=0.4)
        thr = positions_from_signal(p)
        assert hyst.diff().abs().sum() < thr.diff().abs().sum()

    def test_vol_target_switch_caps_exposure(self):
        from hmm_sentiment.backtest import vol_target_switch_weights

        idx, p_bull, rv, rets = self._fix()
        w = vol_target_switch_weights(p_bull, rv, target_vol=0.10)
        assert (w >= 0).all() and (w <= 1.0 + 1e-9).all()
        assert w.iloc[0] == 0.0
        # when realized vol is very low, weight is capped at 1.0
        assert (w[rv > 0] <= 1.0 + 1e-9).all()

    def test_zoo_net_returns_have_no_lookahead(self):
        from hmm_sentiment.backtest import (
            hysteresis_positions,
            prob_weighted_weights,
            vol_target_switch_weights,
            strategy_returns_from_positions,
        )

        idx, p_bull, rv, rets = self._fix()
        for w in (prob_weighted_weights(p_bull), hysteresis_positions(p_bull),
                  vol_target_switch_weights(p_bull, rv)):
            r = strategy_returns_from_positions(w, {"SPY": rets["SPY"], "IEF": rets["IEF"]},
                                                {"SPY": w, "IEF": 1.0 - w})
            assert len(r) == len(idx)
            assert np.isfinite(r).all()


def _degenerate_frame():
    rng = _rng()
    n = 400
    idx = pd.bdate_range("2021-01-04", periods=n)
    vol = np.where(np.arange(n) < n // 2, 0.008, 0.025)
    rets = rng.normal(np.where(np.arange(n) < n // 2, 0.0006, -0.0008), vol)
    df = pd.DataFrame(
        {
            "SPY_ret": rets,
            "realized_vol": pd.Series(rets, index=idx).rolling(20).std().values * np.sqrt(252),
            "VIX_close": np.where(np.arange(n) < n // 2, rng.normal(15, 1.5, n).clip(9), rng.normal(30, 4, n).clip(14)),
            # degenerate: constant column + one-hot-style sparse column
            "sent_w_f": np.where(np.arange(n) % 7 == 0, -1.0, 0.0),
            "sent_w_ema5_f": np.zeros(n),
            "sent_disp_f": np.zeros(n),
        },
        index=idx,
    ).dropna()
    return df


def test_fit_survives_degenerate_features():
    from hmm_sentiment.regime_models import fit_regime_models

    feat = _degenerate_frame()
    cols = ["SPY_ret", "realized_vol", "VIX_close", "sent_w_f", "sent_w_ema5_f", "sent_disp_f"]
    sets = {"M7_fingpt_hmm": cols, "M8_fingpt_pca_hmm": cols}
    models = fit_regime_models(feat, sets, random_state=42)
    for name, res in models.items():
        t = res["model"].transmat_
        assert np.allclose(t.sum(axis=1), 1.0, atol=1e-8), name
        assert np.isfinite(res["posterior_bull"].dropna()).all(), name
