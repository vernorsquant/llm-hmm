"""Statistical significance tests: DM/HAC, JK-Memmel Sharpe diff, block
bootstrap, McNemar, and F1 vs a stress proxy."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestNeweyWest:
    def test_iid_variance_matches_sample_variance(self):
        from hmm_sentiment.stats_tests import newey_west_var

        rng = _rng()
        x = rng.normal(0, 1, 500)
        assert abs(newey_west_var(x, lags=0) - x.var(ddof=0)) < 1e-9

    def test_positive_and_finite_on_autocorrelated(self):
        from hmm_sentiment.stats_tests import newey_west_var

        rng = _rng()
        x = np.empty(400)
        x[0] = 0
        for i in range(1, 400):
            x[i] = 0.8 * x[i - 1] + rng.normal(0, 0.1)
        v = newey_west_var(x, lags=5)
        assert np.isfinite(v) and v > 0


class TestSharpeDiff:
    def _two_series(self, n=1000, mu1=0.001, mu2=0.0005, sd=0.01, seed=3):
        rng = np.random.default_rng(seed)
        r1 = rng.normal(mu1, sd, n)
        r2 = rng.normal(mu2, sd, n)
        return r1, r2

    def test_identical_series_zero_statistic(self):
        from hmm_sentiment.stats_tests import jk_memmel_sharpe_diff

        rng = _rng()
        r = rng.normal(0.0005, 0.01, 500)
        out = jk_memmel_sharpe_diff(r, r)
        assert abs(out["z"]) < 1e-9
        assert out["p_value"] > 0.99

    def test_large_true_difference_rejected(self):
        from hmm_sentiment.stats_tests import jk_memmel_sharpe_diff

        r1, r2 = self._two_series(mu1=0.002, mu2=0.0)
        out = jk_memmel_sharpe_diff(r1, r2)
        assert out["p_value"] < 0.01
        assert out["z"] > 2

    def test_no_difference_not_rejected(self):
        from hmm_sentiment.stats_tests import jk_memmel_sharpe_diff

        r1, r2 = self._two_series(mu1=0.0005, mu2=0.0005, seed=7)
        out = jk_memmel_sharpe_diff(r1, r2)
        assert out["p_value"] > 0.05

    def test_bootstrap_pvalue_recovers_signal(self):
        from hmm_sentiment.stats_tests import bootstrap_sharpe_diff

        r1, r2 = self._two_series(mu1=0.002, mu2=0.0, seed=11)
        out = bootstrap_sharpe_diff(r1, r2, n_boot=2000, block=10, seed=5)
        assert out["p_value"] < 0.05
        assert out["ci_low"] > 0  # CI excludes zero

    def test_bootstrap_null_not_rejected(self):
        from hmm_sentiment.stats_tests import bootstrap_sharpe_diff

        r1, r2 = self._two_series(mu1=0.0005, mu2=0.0005, seed=13)
        out = bootstrap_sharpe_diff(r1, r2, n_boot=2000, block=10, seed=5)
        assert out["p_value"] > 0.05


class TestMcNemar:
    def test_hand_built_table(self):
        from hmm_sentiment.stats_tests import mcnemar_test

        # b = model A only correct = 20, c = model B only correct = 5
        out = mcnemar_test(b=20, c=5)
        # binomial two-sided p for 20/25 successes under p=0.5
        assert out["p_value"] < 0.01
        assert out["statistic"] > 0

    def test_symmetric_discordant_pairs_not_significant(self):
        from hmm_sentiment.stats_tests import mcnemar_test

        out = mcnemar_test(b=10, c=10)
        assert out["p_value"] > 0.5


class TestStressF1:
    def test_f1_perfect_prediction(self):
        from hmm_sentiment.stats_tests import stress_f1

        y = np.array([1, 1, 0, 0])
        pred = np.array([1, 1, 0, 0])
        assert stress_f1(y, pred) == 1.0

    def test_f1_zero_when_no_overlap(self):
        from hmm_sentiment.stats_tests import stress_f1

        y = np.array([1, 1, 0, 0])
        pred = np.array([0, 0, 1, 1])
        assert stress_f1(y, pred) == 0.0
