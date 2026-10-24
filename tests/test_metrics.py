"""Exact-value tests for performance metrics and transition lead time."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestPerfStats:
    def test_constant_daily_return_compounds_correctly(self):
        from hmm_sentiment.metrics import perf_stats

        idx = pd.bdate_range("2023-01-02", periods=252)
        r = pd.Series(0.0004, index=idx)
        s = perf_stats(r, "const")
        expected_ann = (1.0004 ** 252) - 1.0
        assert abs(s["ann_return"] - expected_ann) < 1e-9
        assert s["ann_vol"] < 1e-12 or abs(s["sharpe"]) > 1e6  # zero vol edge
        assert s["max_drawdown"] == 0.0

    def test_max_drawdown_known_series(self):
        from hmm_sentiment.metrics import max_drawdown

        idx = pd.bdate_range("2023-01-02", periods=4)
        r = pd.Series([0.10, -0.50, 0.10, 0.0], index=idx)
        # equity: 1.1, 0.55, 0.605, 0.605 -> trough dd = 0.55/1.1 - 1 = -0.50
        assert abs(max_drawdown(r) - (-0.50)) < 1e-12


class TestLeadTime:
    def _series(self, values, idx):
        return pd.Series(values, index=idx)

    def test_positive_lead_when_flip_before_reference(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = pd.bdate_range("2023-03-01", periods=10)
        p_bear = self._series([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], idx)
        out = bear_entry_lead(p_bear, ref_date=idx[4], window_start=idx[0])
        # entry on idx[1]; ref = idx[4] -> lead = 3 trading days
        assert out["lead_days"] == 3
        assert out["entry_date"] == idx[1]
        assert out["carried_in"] is False

    def test_flip_after_reference_is_not_a_lead(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = pd.bdate_range("2023-03-01", periods=10)
        p_bear = self._series([0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8], idx)
        out = bear_entry_lead(p_bear, ref_date=idx[2], window_start=idx[0])
        # entry happens after the reference -> no lead before the crisis
        assert out["lead_days"] is None
        assert out["entry_date"] is None

    def test_none_when_never_flips(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = pd.bdate_range("2023-03-01", periods=10)
        p_bear = self._series([0.2] * 10, idx)
        out = bear_entry_lead(p_bear, ref_date=idx[4], window_start=idx[0])
        assert out["lead_days"] is None
