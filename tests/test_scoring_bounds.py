"""Bounds tests: LLM importance and FinBERT-derived confidence in [0, 1]."""

import numpy as np

from conftest import _rng


class TestConfidence:
    def test_confidence_bounds_random_simplex(self):
        rng = _rng()
        from hmm_sentiment.scoring import confidence_from_probs

        p = rng.dirichlet([2, 2, 2], size=500)
        c = confidence_from_probs(p)
        assert np.all(c >= 0.0) and np.all(c <= 1.0)

    def test_confidence_uniform_is_zero_and_degenerate_is_one(self):
        from hmm_sentiment.scoring import confidence_from_probs

        uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
        # the spec formula includes a 1e-9 epsilon inside the log, so exact
        # zero is not expected; tolerance reflects that epsilon.
        assert abs(confidence_from_probs(uniform)[0]) < 5e-8

        degenerate = np.array([[1.0, 0.0, 0.0]])
        assert abs(confidence_from_probs(degenerate)[0] - 1.0) < 1e-9

    def test_entropy_matches_manual_computation(self):
        from hmm_sentiment.scoring import entropy

        p = np.array([[0.5, 0.25, 0.25]])
        expected = -(0.5 * np.log(0.5 + 1e-9) + 0.25 * np.log(0.25 + 1e-9) * 2)
        assert abs(entropy(p)[0] - expected) < 1e-9


class TestImportance:
    def test_clamped_to_unit_interval(self):
        from hmm_sentiment.scoring import clamp_importance

        for raw in [-5.0, -0.01, 0.0, 0.42, 1.0, 1.01, 3.7]:
            v = clamp_importance(raw)
            assert 0.0 <= v <= 1.0
        assert clamp_importance(-5.0) == 0.0
        assert clamp_importance(3.7) == 1.0
        assert abs(clamp_importance(0.42) - 0.42) < 1e-9

    def test_llm_parse_tolerates_garbage(self):
        from hmm_sentiment.scoring import parse_importance

        assert parse_importance("0.73") == 0.73
        assert parse_importance("importance: 0.15") == 0.15
        assert parse_importance("the market is big") is None


class TestSentimentSynthesis:
    def test_s_raw_bounds(self):
        rng = _rng()
        from hmm_sentiment.scoring import sentiment_raw

        p = rng.dirichlet([2, 2, 2], size=300)
        s = sentiment_raw(p)
        assert np.all(s >= -1.0) and np.all(s <= 1.0)

    def test_daily_weighted_sentiment_bounds(self):
        rng = _rng()
        from hmm_sentiment.daily_features import daily_weighted_sentiment

        p = rng.dirichlet([2, 2, 2], size=40)
        s_raw = 2 * p[:, 0] - 1  # pos - neg
        w = rng.uniform(0, 1, size=40)
        s_t = daily_weighted_sentiment(s_raw, w)
        assert -1.0 - 1e-9 <= s_t <= 1.0 + 1e-9

    def test_daily_weighted_sentiment_zero_weight_is_zero(self):
        from hmm_sentiment.daily_features import daily_weighted_sentiment

        assert daily_weighted_sentiment(np.array([0.5, -0.7]), np.array([0.0, 0.0])) == 0.0
