"""Market data ingestion: SPY, IEF (yfinance, adjusted) and VIX (CBOE CSV).

Produces a trading-day-aligned frame with log returns and 20-day realized vol.
"""

import io

import numpy as np
import pandas as pd

from .config import CACHE_DIR, EVAL_END, FEATURE_START

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def _fetch_prices_yf(tickers, start, end):
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close[tickers].dropna(how="all")


def _fetch_vix_cboe():
    import requests

    r = requests.get(VIX_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    vix = pd.read_csv(io.StringIO(r.text))
    vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    vix = vix.set_index("DATE")["CLOSE"].astype(float)
    vix.index.name = "Date"
    return vix


def fetch_market_data(start=FEATURE_START, end=EVAL_END, use_cache=True):
    """Return DataFrame indexed by trading day with columns:
    SPY_close, IEF_close, VIX_close, SPY_ret, IEF_ret, realized_vol.

    `end` is inclusive. Caches to data/cache/market_<start>_<end>.parquet.
    """
    cache_file = f"{CACHE_DIR}/market_{start}_{end}.parquet"
    if use_cache:
        try:
            cached = pd.read_parquet(cache_file)
            if cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
                return cached
        except Exception:
            pass

    end_incl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = _fetch_prices_yf(["SPY", "IEF"], start, end_incl)
    vix = _fetch_vix_cboe().loc[lambda s: (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]

    idx = prices.index.intersection(vix.index)
    df = pd.DataFrame(index=idx)
    df["SPY_close"] = prices["SPY"].reindex(idx)
    df["IEF_close"] = prices["IEF"].reindex(idx)
    df["VIX_close"] = vix.reindex(idx).ffill()
    df = df.dropna()

    for a in ("SPY", "IEF"):
        df[f"{a}_ret"] = np.log(df[f"{a}_close"]).diff()
    df["realized_vol"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    df = df.dropna()

    df.index.name = "Date"
    df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    m = fetch_market_data(use_cache=False)
    print(m.head(3))
    print(m.tail(3))
    print("rows:", len(m), "range:", m.index.min(), "->", m.index.max())
