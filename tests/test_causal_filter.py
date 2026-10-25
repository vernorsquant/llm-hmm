"""Regression test for the smoothing-lookahead leak: walk-forward posteriors
must be CAUSAL — shocking future observations must not change past posteriors."""

import numpy as np
import pandas as pd

from conftest import _rng


def _train_and_month():
    rng = _rng()
    n_train, n_month = 400, 22
    idx_tr = pd.bdate_range("2022-01-03", periods=n_train)
    idx_mo = pd.bdate_range("2023-05-01", periods=n_month)
    vol = np.where(np.arange(n_train) < n_train // 2, 0.008, 0.025)
    r_tr = rng.normal(np.where(np.arange(n_train) < n_train // 2, 0.0006, -0.0008), vol)
    cols = ["SPY_ret", "realized_vol", "VIX_close"]

    def frame(idx, rets, vix):
        df = pd.DataFrame({"SPY_ret": rets, "VIX_close": vix}, index=idx)
        df["realized_vol"] = df["SPY_ret"].rolling(20).std().values * np.sqrt(252)
        return df[cols].dropna()

    tr = frame(idx_tr, r_tr, rng.normal(17, 2, n_train).clip(9))
    vix_mo = np.concatenate([rng.normal(15, 1.5, n_month - 3), [45.0, 44.0, 43.0]])
    mo = frame(idx_mo, rng.normal(0.0004, 0.008, n_month), vix_mo)
    return tr, mo, cols


class TestCausalFilter:
    def test_filtered_posteriors_are_causal(self):
        from hmm_sentiment.regime_models import _fit_one, causal_filtered_probabilities

        tr, mo, cols = _train_and_month()
        res = _fit_one(tr, cols, False, 42)
        hmm, scaler = res["model"], res["scaler"]

        Xs = scaler.transform(mo.values)
        alpha = causal_filtered_probabilities(hmm, Xs)

        # shock the LAST day's features massively
        Xs2 = Xs.copy()
        Xs2[-1] += np.array([0.0, 0.0, 8.0])
        alpha2 = causal_filtered_probabilities(hmm, Xs2)

        # causal: days before the shock are bit-identical
        assert np.allclose(alpha[:-1], alpha2[:-1], atol=1e-12)
        # the shocked day itself changes
        assert not np.allclose(alpha[-1], alpha2[-1], atol=1e-9)

    def test_smoothed_predict_proba_was_leaking(self):
        """Documents the original bug: hmmlearn predict_proba (smoothing) lets a
        future shock change PAST posteriors. Kept as a canary so the regression
        stays honest about what was fixed."""
        from hmm_sentiment.regime_models import _fit_one

        tr, mo, cols = _train_and_month()
        res = _fit_one(tr, cols, False, 42)
        Xs = scaler = res["scaler"].transform(mo.values)
        smoothed = res["model"].predict_proba(Xs)

        Xs2 = Xs.copy()
        Xs2[-1] += np.array([0.0, 0.0, 8.0])
        smoothed2 = res["model"].predict_proba(Xs2)

        past_changed = not np.allclose(smoothed[:-1], smoothed2[:-1], atol=1e-12)
        assert past_changed, "smoothing leak appears fixed in hmmlearn — update README"

    def test_filtered_month_posterior_bounded_and_aligned(self):
        from hmm_sentiment.regime_models import filtered_month_posterior

        tr, mo, cols = _train_and_month()
        post = filtered_month_posterior(tr, mo, cols, False, 42)
        assert len(post) == len(mo)
        assert np.all(post.dropna().between(0.0, 1.0))
