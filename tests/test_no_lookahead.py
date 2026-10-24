"""Zero-lookahead tests: close-safeguard mapping and signal lagging."""

import pandas as pd


class TestCloseSafeguard:
    def test_after_close_maps_to_next_trading_day(self, synthetic_market, synthetic_articles):
        from hmm_sentiment.daily_features import map_articles_to_trading_days

        trading_days = synthetic_market.index
        mapped = map_articles_to_trading_days(synthetic_articles, trading_days)

        days = list(trading_days)
        d0, d1 = days[50], days[51]

        # 09:30 and 15:00 ET on d0 -> d0
        assert mapped.iloc[0]["trading_date"] == d0
        assert mapped.iloc[1]["trading_date"] == d0
        # 16:30 and 23:59 ET on d0 -> d1
        assert mapped.iloc[2]["trading_date"] == d1
        assert mapped.iloc[3]["trading_date"] == d1

    def test_non_trading_day_maps_to_next_session(self, synthetic_market):
        from hmm_sentiment.daily_features import map_articles_to_trading_days

        trading_days = synthetic_market.index
        # Saturday after days[0]
        sat = trading_days[0] + pd.Timedelta(days=2)
        art = pd.DataFrame(
            {
                "timestamp_utc": [sat.tz_localize("UTC")],
                "source": ["Reuters"],
                "headline": ["Weekend macro worries build"],
                "summary": ["s"],
            }
        )
        mapped = map_articles_to_trading_days(art, trading_days)
        assert mapped.iloc[0]["trading_date"] == trading_days[1]

    def test_articles_after_eval_end_are_dropped(self, synthetic_market):
        from hmm_sentiment.daily_features import map_articles_to_trading_days

        trading_days = synthetic_market.index
        late = trading_days[-1] + pd.Timedelta(days=30)
        art = pd.DataFrame(
            {
                "timestamp_utc": [late.tz_localize("UTC")],
                "source": ["X"],
                "headline": ["Future news"],
                "summary": ["s"],
            }
        )
        mapped = map_articles_to_trading_days(art, trading_days)
        assert len(mapped) == 0


class TestSignalLag:
    def test_binary_switch_position_uses_lagged_posterior(self):
        from hmm_sentiment.backtest import positions_from_signal

        idx = pd.bdate_range("2023-01-02", periods=10)
        posterior = pd.Series([1.0] * 5 + [0.0] * 5, index=idx)
        pos = positions_from_signal(posterior)
        # Position on day t must come from posterior at t-1 (first day -> NaN prior -> 0)
        assert pos.iloc[0] == 0.0
        assert (pos.iloc[1:5] == 1.0).all()   # lag of bull
        assert pos.iloc[5] == 1.0             # still yesterday's bull
        assert (pos.iloc[6:] == 0.0).all()    # lag of bear

    def test_strategy_return_applies_position_to_same_day_return(self, synthetic_market):
        from hmm_sentiment.backtest import strategy_returns_from_positions

        idx = synthetic_market.index[:50]
        spy = synthetic_market["SPY_ret"].iloc[:50]
        pos = pd.Series(1.0, index=idx)
        pos.iloc[-1] = 0.0
        rets = strategy_returns_from_positions(pos, {"SPY": spy}, {"SPY": pos})
        # Last day: position 0 => return 0 regardless of SPY move
        assert rets.iloc[-1] == 0.0
        pd.testing.assert_series_equal(rets.iloc[:-1], spy.iloc[:-1], check_names=False)
