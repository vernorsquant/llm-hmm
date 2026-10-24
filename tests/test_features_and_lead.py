"""Tests for coverage-gap handling and crisis lead-time semantics."""

import numpy as np
import pandas as pd

from conftest import _rng


class TestCoverageGaps:
    def _articles(self, day, n=2, s_raw=-0.6):
        rng = _rng()
        return pd.DataFrame(
            {
                "trading_date": [day] * n,
                "S_raw": [s_raw] * n,
                "confidence": [0.8] * n,
                "importance": [0.9] * n,
            }
        )

    def test_no_coverage_day_ffills_prior_sentiment(self):
        from hmm_sentiment.daily_features import build_daily_news_features

        td = pd.bdate_range("2023-01-02", periods=6)
        arts = self._articles(td[0])  # coverage only on day 0
        feat = build_daily_news_features(arts, td)
        # day 1 within ffill limit: inherits day-0 sentiment level
        assert feat["n_articles"].iloc[1] == 0
        assert feat["sent_w"].iloc[1] == feat["sent_w"].iloc[0]
        # day 5 beyond the 10-day ffill limit would decay only after limit;
        # with limit=10 all 5 gap days inherit here
        assert abs(feat["sent_w"].iloc[5] - feat["sent_w"].iloc[0]) < 1e-12
        # counts remain 0 on gap days (true observation)
        assert (feat["hi_imp_volume"].iloc[1:] == 0).all()

    def test_gap_days_cannot_distinguish_states(self):
        """The regression test for the coverage-artifact bug: sentiment features
        on gap days must equal the last covered day's, not zeros."""
        from hmm_sentiment.daily_features import build_daily_news_features

        td = pd.bdate_range("2023-01-02", periods=4)
        arts = self._articles(td[0], s_raw=-0.8)
        feat = build_daily_news_features(arts, td)
        assert feat["sent_disp"].iloc[2] == feat["sent_disp"].iloc[0]


class TestBearEntryLead:
    def _idx(self):
        return pd.bdate_range("2023-02-15", periods=20)  # Feb 15 .. Mar 14

    def test_genuine_flip_inside_window(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = self._idx()
        p_bear = pd.Series([0.2] * 5 + [0.8] * 15, index=idx)
        out = bear_entry_lead(p_bear, ref_date=idx[-2], window_start=idx[0])
        # entry at idx[5], ref idx[-2]=idx[18] -> 13 trading days early
        assert out["lead_days"] == 13
        assert out["entry_date"] == idx[5]
        assert out["carried_in"] is False

    def test_carried_in_bear_flagged(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = self._idx()
        p_bear = pd.Series([0.8] * 20, index=idx)  # bear from window start
        out = bear_entry_lead(p_bear, ref_date=idx[-2], window_start=idx[0])
        assert out["carried_in"] is True
        assert out["entry_date"] is None
        assert out["lead_days"] == len(idx) - 2  # bear for the whole pre-ref window

    def test_never_bear_returns_none(self):
        from hmm_sentiment.metrics import bear_entry_lead

        idx = self._idx()
        p_bear = pd.Series([0.2] * 20, index=idx)
        out = bear_entry_lead(p_bear, ref_date=idx[-2], window_start=idx[0])
        assert out["lead_days"] is None and out["carried_in"] is False and out["entry_date"] is None
