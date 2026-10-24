"""VADER baseline scorer + the two VADER regime models (M5/M6)."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestVaderScoring:
    def test_s_raw_and_confidence_bounds(self):
        from hmm_sentiment.vader_scoring import vader_scores

        rng = _rng()
        headlines = [
            "Fed raises interest rates sharply",
            "Stocks rally on cooling inflation",
            "Apple ships new laptop",
            "Bank failure sparks deposit flight",
            "",
        ]
        out = vader_scores(pd.DataFrame({"headline": headlines}))
        assert np.all(out["S_raw"].between(-1.0, 1.0))
        assert np.all(out["confidence"].between(0.0, 1.0))
        # prob columns must form a simplex (neg + neu + pos = 1)
        simplex = out[["P_pos", "P_neu", "P_neg"]].sum(axis=1)
        assert np.allclose(simplex, 1.0, atol=1e-6)

    def test_scores_are_cached_and_stable(self):
        from hmm_sentiment.vader_scoring import vader_scores

        df = pd.DataFrame({"headline": ["Hot CPI print spooks markets"] * 3})
        a = vader_scores(df)
        b = vader_scores(df)
        assert np.allclose(a["S_raw"], b["S_raw"])


class TestVaderRegimeModels:
    def _feat(self):
        rng = _rng()
        n = 600
        idx = pd.bdate_range("2021-01-04", periods=n)
        half = n // 2
        vol = np.where(np.arange(n) < half, 0.008, 0.025)
        drift = np.where(np.arange(n) < half, 0.0006, -0.0008)
        rets = rng.normal(drift, vol)
        vix = np.where(np.arange(n) < half, rng.normal(15, 1.5, n).clip(9), rng.normal(30, 4.0, n).clip(14))
        sent = rng.normal(0, 0.3, n)
        return pd.DataFrame(
            {
                "SPY_ret": rets,
                "realized_vol": pd.Series(rets, index=idx).rolling(20).std().values * np.sqrt(252),
                "VIX_close": vix,
                "sent_w_v": sent,
                "sent_w_ema5_v": pd.Series(sent, index=idx).ewm(span=5).mean().values,
                "sent_disp_v": np.abs(rng.normal(0.2, 0.05, n)),
            },
            index=idx,
        ).dropna()

    def test_m5_m6_fit_valid_and_canonicalized(self):
        from hmm_sentiment.regime_models import FEATURE_SETS, fit_regime_models

        feat = self._feat()
        sets = {k: v for k, v in FEATURE_SETS.items() if k.startswith("M5") or k.startswith("M6")}
        models = fit_regime_models(feat, sets, random_state=42)
        for name in ["M5_vader_hmm", "M6_vader_pca_hmm"]:
            res = models[name]
            t = res["model"].transmat_
            assert np.allclose(t.sum(axis=1), 1.0, atol=1e-8)
            assert res["model"].monitor_.converged
            post = res["posterior_bull"]
            assert np.all(post.dropna().between(0.0, 1.0))
            vix = feat["VIX_close"].values
            states = res["states"]
            assert vix[states == 1].mean() > vix[states == 0].mean(), name
