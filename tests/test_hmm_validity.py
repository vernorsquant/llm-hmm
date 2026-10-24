"""Validity tests: HMM convergence, transition matrices, state canonicalization."""

import numpy as np
import pandas as pd

from conftest import _rng


def _feature_frame(market, rng):
    """Build a synthetic feature frame mimicking the real pipeline output,
    including a sentiment block for models 3/4."""
    n = len(market)
    sentiment = rng.normal(0, 0.3, n) - 2.0 * (market["VIX_close"].values - market["VIX_close"].mean()) / market["VIX_close"].std()
    feat = pd.DataFrame(
        {
            "SPY_ret": market["SPY_ret"],
            "realized_vol": market["realized_vol"],
            "VIX_close": market["VIX_close"],
            "sent_w": sentiment,
            "sent_w_ema5": pd.Series(sentiment, index=market.index).ewm(span=5).mean(),
            "hi_imp_volume": rng.poisson(1.0, n).astype(float),
            "sent_disp": np.abs(rng.normal(0.2, 0.05, n)),
        },
        index=market.index,
    )
    return feat


FEATURE_SETS = {
    "M1": ["SPY_ret", "realized_vol", "VIX_close"],
    "M2": ["SPY_ret", "realized_vol", "VIX_close"],
    "M3": ["SPY_ret", "realized_vol", "VIX_close", "sent_w", "sent_w_ema5", "hi_imp_volume", "sent_disp"],
    "M4": ["SPY_ret", "realized_vol", "VIX_close", "sent_w", "sent_w_ema5", "hi_imp_volume", "sent_disp"],
}


class TestHMMValidity:
    def test_all_four_models_fit_valid(self, synthetic_market):
        rng = _rng()
        from hmm_sentiment.regime_models import fit_regime_models

        feat = _feature_frame(synthetic_market, rng)
        models = fit_regime_models(feat, FEATURE_SETS, random_state=42)
        assert set(models.keys()) == {"M1", "M2", "M3", "M4"}

        for name, res in models.items():
            transmat = res["model"].transmat_
            assert transmat.shape == (2, 2), name
            assert np.all(transmat >= 0.0) and np.all(transmat <= 1.0)
            assert np.allclose(transmat.sum(axis=1), 1.0, atol=1e-8), name
            assert res["model"].monitor_.converged, f"{name} did not converge"
            post = res["posterior_bull"]
            assert np.all(post >= 0.0) and np.all(post <= 1.0), name
            assert len(post) == len(feat), name

    def test_bear_state_is_high_vol_state(self, synthetic_market):
        rng = _rng()
        from hmm_sentiment.regime_models import fit_regime_models

        feat = _feature_frame(synthetic_market, rng)
        models = fit_regime_models(feat, FEATURE_SETS, random_state=42)
        for name, res in models.items():
            vix = feat["VIX_close"].values
            states = res["states"]
            vix_bear = vix[states == 1].mean()
            vix_bull = vix[states == 0].mean()
            assert vix_bear > vix_bull, f"{name}: bear state must have higher VIX"

    def test_pca_keeps_required_variance(self, synthetic_market):
        rng = _rng()
        from hmm_sentiment.regime_models import fit_regime_models

        feat = _feature_frame(synthetic_market, rng)
        models = fit_regime_models(feat, FEATURE_SETS, random_state=42)
        for name in ["M2", "M4"]:
            ratio = models[name]["pca_explained_variance"]
            assert ratio >= 0.94, f"{name}: PCA explained variance {ratio:.3f} < 0.95 - tol"

    def test_transition_diagonal_is_persistent(self, synthetic_market):
        """Literature baseline: 2-state HMMs on daily data show high persistence
        (diagonal typically > 0.9; cf. regime-switching literature)."""
        rng = _rng()
        from hmm_sentiment.regime_models import fit_regime_models

        feat = _feature_frame(synthetic_market, rng)
        models = fit_regime_models(feat, FEATURE_SETS, random_state=42)
        for name, res in models.items():
            diag = np.diag(res["model"].transmat_)
            assert np.all(diag > 0.75), f"{name}: weak persistence {diag}"
