"""Daily news feature synthesis with the 16:00 ET close safeguard.

Articles published after 16:00 America/New_York on day t map strictly to the
NEXT trading session t+1 (and weekend/holiday publishes map to the next open
session), eliminating lookahead bias.
"""

import numpy as np
import pandas as pd


def _et(ts_utc):
    return ts_utc.tz_convert("America/New_York") if ts_utc.tzinfo else ts_utc.tz_localize("UTC").tz_convert("America/New_York")


def map_articles_to_trading_days(articles, trading_days):
    """Return a copy of `articles` with a `trading_date` column (tz-naive dates).

    Rule: effective day = publish date if it is a trading day and publish time
    (ET) <= 16:00; otherwise the first trading day STRICTLY after the publish
    date. Articles mapping beyond the last trading day are dropped.
    """
    td = pd.DatetimeIndex(trading_days).normalize()
    rows = []
    for _, art in articles.iterrows():
        et = _et(pd.Timestamp(art["timestamp_utc"]))
        d = pd.Timestamp(et.date())  # tz-naive calendar date in ET
        after_close = (et.hour, et.minute, et.second) >= (16, 0, 0)
        if after_close:
            pos = td.searchsorted(d, side="right")
        else:
            pos = td.searchsorted(d, side="left")
            if pos < len(td) and td[pos] == d:
                pass  # publish date is itself a trading day, pre-close
        if pos >= len(td):
            continue  # maps beyond evaluation horizon
        row = art.to_dict()
        row["trading_date"] = td[pos]
        rows.append(row)
    return pd.DataFrame(rows)


def daily_weighted_sentiment(s_raw, weights):
    """S_t = sum(S_raw_i * W_i) / (sum(W_i) + 1e-6). Scalar-safe."""
    s_raw = np.asarray(s_raw, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(s_raw * weights) / (np.sum(weights) + 1e-6))


def build_daily_news_features(articles_scored, trading_days, weighted=None):
    """Aggregate scored articles to daily features aligned on `trading_days`.

    Required article columns: trading_date, S_raw, confidence, importance.
    Output columns: sent_w, sent_w_ema5, hi_imp_volume, sent_disp, n_articles.

    `weighted` (default: config.WEIGHTED_SENTIMENT) selects the aggregation:
      - True: S_t = sum(S_raw_i * C_i * I_i) / (sum C_i * I_i + 1e-6)
      - False: plain unweighted mean of S_raw (current default — the article
        weighting mechanism is disabled pending importance calibration; all
        scores stay cached and the mechanism is one flag away).
    """
    from . import config as _config
    from .scoring import article_weights

    if weighted is None:
        weighted = _config.WEIGHTED_SENTIMENT

    td = pd.DatetimeIndex(trading_days).normalize()
    arts = articles_scored.copy()
    arts["trading_date"] = pd.DatetimeIndex(arts["trading_date"]).normalize()
    if weighted:
        arts["weight"] = article_weights(arts["confidence"].values, arts["importance"].values)
    else:
        arts["weight"] = 1.0

    # Dedupe identical headlines mapping to the SAME trading day: the corpus
    # dedupes on (headline, window), but overlapping GDELT windows refetch the
    # same article into multiple windows, double-counting weight on the event
    # days this pipeline cares about (up to 7x on some FOMC days).
    dedupe_cols = [c for c in ("headline", "trading_date") if c in arts.columns]
    if dedupe_cols:
        arts = arts.drop_duplicates(subset=dedupe_cols)

    out = pd.DataFrame(index=td)
    grouped = arts.groupby("trading_date")
    sent = {}
    disp = {}
    hiv = {}
    nart = {}
    for day, g in grouped:
        s_t = daily_weighted_sentiment(g["S_raw"].values, g["weight"].values)
        sent[day] = s_t
        disp[day] = float(np.std(g["S_raw"].values)) if len(g) > 1 else 0.0
        hiv[day] = int((g["importance"].values >= 0.7).sum())
        nart[day] = int(len(g))

    # No-coverage days are UNOBSERVED, not neutral: forward-fill the sentiment
    # level/first (NaN days inherit the last covered day, limit 10), then fill
    # leading NaNs with 0 so the HMM cannot split states on coverage gaps.
    # Counts stay 0 on gap days (true observation).
    out["sent_w"] = pd.Series(sent).reindex(td).ffill(limit=10).fillna(0.0)
    out["sent_w_ema5"] = out["sent_w"].ewm(span=5, adjust=False).mean()
    out["hi_imp_volume"] = pd.Series(hiv).reindex(td).fillna(0).astype(float)
    out["sent_disp"] = pd.Series(disp).reindex(td).ffill(limit=10).fillna(0.0)
    out["n_articles"] = pd.Series(nart).reindex(td).fillna(0).astype(float)
    return out
