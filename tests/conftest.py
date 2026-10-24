import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_market():
    """600 business days of synthetic SPY/IEF/VIX with a regime shift in the middle.

    Regime A (first half): low vol, positive drift, VIX ~ 15.
    Regime B (second half): high vol, negative drift, VIX ~ 30.
    """
    rng = _rng()
    n = 600
    idx = pd.bdate_range("2021-01-04", periods=n)
    half = n // 2

    vol = np.where(np.arange(n) < half, 0.008, 0.025)
    drift = np.where(np.arange(n) < half, 0.0006, -0.0008)
    spy_ret = rng.normal(drift, vol)
    ief_ret = rng.normal(0.0001, 0.0035) * np.ones(n)

    vix = np.where(
        np.arange(n) < half,
        rng.normal(15, 1.5, n).clip(9),
        rng.normal(30, 4.0, n).clip(14),
    )

    spy_close = 400.0 * np.exp(np.cumsum(spy_ret))
    ief_close = 100.0 * np.exp(np.cumsum(ief_ret))

    df = pd.DataFrame(
        {
            "SPY_close": spy_close,
            "IEF_close": ief_close,
            "VIX_close": vix,
            "SPY_ret": spy_ret,
            "IEF_ret": ief_ret,
        },
        index=idx,
    )
    df["realized_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    return df.dropna()


@pytest.fixture
def synthetic_articles(synthetic_market):
    """A handful of synthetic articles with ET timestamps spanning trading and
    non-trading days, including after-close publishes."""
    days = synthetic_market.index
    d0 = days[50]  # a trading day
    d1 = days[51]
    assert d1 > d0

    def ts(day, hour_et, minute=0):
        # EDT in October would be UTC-4; use UTC-5/EST-independent construction:
        # 21:30 UTC == 16:30 EST. Use fixed offset UTC-5 for determinism.
        utc = pd.Timestamp(f"{day.date()} {hour_et:02d}:{minute:02d}:00", tz="America/New_York").tz_convert("UTC")
        return utc

    return pd.DataFrame(
        {
            "timestamp_utc": [
                ts(d0, 9, 30),   # pre-open on trading day -> day d0
                ts(d0, 15, 0),   # before close -> day d0
                ts(d0, 16, 30),  # after close -> day d1
                ts(d0, 23, 59),  # just before midnight ET -> day d1
            ],
            "source": ["Reuters", "Bloomberg", "CNBC", "WSJ"],
            "headline": [
                "Fed signals rate path unchanged",
                "CPI comes in hotter than expected",
                "Regional bank deposits fall amid SVB turmoil",
                "Markets rally on cooling inflation print",
            ],
            "summary": [
                "Federal Reserve officials held rates steady.",
                "Consumer prices rose more than economists expected.",
                "Silicon Valley Bank failure sparks deposit flight.",
                "Equities rallied after inflation cooled.",
            ],
        }
    )
