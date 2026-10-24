"""MVO constraint tests: long-only, fully-invested, correct blending, costs."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestMVOSolver:
    def test_weights_sum_to_one_and_nonneg(self):
        rng = _rng()
        from hmm_sentiment.backtest import solve_max_sharpe_weights

        for _ in range(50):
            mu = rng.normal(0.02, 0.10, size=2)
            a = rng.uniform(0.01, 0.25)
            b = rng.uniform(0.01, 0.25)
            rho = rng.uniform(-0.5, 0.8)
            cov = np.array([[a * a, rho * a * b], [rho * a * b, b * b]])
            w = solve_max_sharpe_weights(mu, cov)
            assert abs(w.sum() - 1.0) < 1e-6
            assert np.all(w >= -1e-9)

    def test_blended_inputs_match_formula(self):
        from hmm_sentiment.backtest import blend_moments

        mu0, mu1 = np.array([0.10, 0.03]), np.array([-0.05, 0.04])
        s0 = np.array([[0.04, 0.001], [0.001, 0.009]])
        s1 = np.array([[0.09, -0.002], [-0.002, 0.004]])
        mu_b, s_b = blend_moments(mu0, s0, mu1, s1, p=0.7)
        assert np.allclose(mu_b, 0.7 * mu0 + 0.3 * mu1)
        assert np.allclose(s_b, 0.7 * s0 + 0.3 * s1)

    def test_corner_pure_regimes(self):
        from hmm_sentiment.backtest import solve_max_sharpe_weights

        mu0 = np.array([0.10, 0.02])
        cov = np.array([[0.04, 0.001], [0.001, 0.009]])
        w = solve_max_sharpe_weights(mu0, cov)
        assert w.shape == (2,)
        assert abs(w.sum() - 1.0) < 1e-6 and np.all(w >= -1e-9)

    def test_singular_cov_does_not_crash(self):
        from hmm_sentiment.backtest import solve_max_sharpe_weights

        mu = np.array([0.1, 0.05])
        cov = np.zeros((2, 2))
        w = solve_max_sharpe_weights(mu, cov)
        assert np.isfinite(w).all() and abs(w.sum() - 1.0) < 1e-6 and np.all(w >= -1e-9)


class TestCosts:
    def test_turnover_cost_is_bps_times_turnover(self):
        from hmm_sentiment.backtest import transaction_cost

        w_prev = np.array([1.0, 0.0])
        w_new = np.array([0.0, 1.0])
        # turnover = |dw| summed = 2.0 ; 5 bps per unit one-sided trade
        assert abs(transaction_cost(w_prev, w_new, cost_bps=5.0) - 2.0 * 0.0005) < 1e-12

    def test_no_trade_no_cost(self):
        from hmm_sentiment.backtest import transaction_cost

        w = np.array([0.6, 0.4])
        assert transaction_cost(w, w, cost_bps=5.0) == 0.0


class TestRegimeConditionedMoments:
    def test_moments_from_mapped_samples(self):
        rng = _rng()
        from hmm_sentiment.backtest import regime_conditioned_moments

        rets = pd.DataFrame(
            {"SPY": rng.normal(0.0005, 0.01, 200), "IEF": rng.normal(0.0001, 0.002, 200)}
        )
        states = np.zeros(200, dtype=int)
        states[100:] = 1
        moments = regime_conditioned_moments(rets, states)
        assert set(moments.keys()) == {0, 1}
        for k in (0, 1):
            mu, cov = moments[k]
            assert mu.shape == (2,)
            assert cov.shape == (2, 2)
            assert np.allclose(cov, cov.T)
            assert np.all(np.linalg.eigvalsh(cov) > 0)
