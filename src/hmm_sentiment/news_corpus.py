"""News corpus for 2023: GDELT DOC 2.0 (primary, real headlines + timestamps)
with a deterministic synthesized-calendar fallback.

GDELT artlist provides [timestamp, source(domain), headline]; no summaries —
the LLM importance rater and FinBERT therefore operate on headlines, which is
disclosed in the README. Articles are fetched in event-keyed windows (FOMC,
CPI, jobs reports, SVB, debt ceiling) plus routine days for background noise.
"""

import hashlib
import json
import time

import pandas as pd
import requests

from .config import CACHE_DIR, RAW_DIR

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SLEEP_SEC = 6.0  # documented rate limit: 1 call / 5s

CORE_QUERY = '("federal reserve" OR inflation OR "interest rates" OR recession OR "stock market" OR "bond yields") sourcecountry:US'

# 2023 event calendar (UTC windows). FOMC statements 14:00 ET; CPI/jobs 08:30 ET.
EVENT_WINDOWS = [
    # (label, start_utc "YYYYMMDDHHMMSS", end_utc, query)
    ("fomc_jan", "20230131130000", "20230201040000", CORE_QUERY),
    ("cpi_jan", "20230112080000", "20230112230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_jan", "20230203080000", "20230203230000", '(("jobs report" OR "nonfarm payrolls" OR unemployment)) sourcecountry:US'),
    ("cpi_feb", "20230214080000", "20230214230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("fomc_mar", "20230322120000", "20230323040000", CORE_QUERY),
    ("svb_1", "20230309000000", "20230310000000", '("Silicon Valley Bank" OR SVB OR "SVB Financial") sourcecountry:US'),
    ("svb_2", "20230310000000", "20230311000000", '("Silicon Valley Bank" OR SVB OR "bank failure" OR "deposit insurance") sourcecountry:US'),
    ("svb_3", "20230311000000", "20230312000000", '("Silicon Valley Bank" OR "Signature Bank" OR "regional banks" OR "bank failure") sourcecountry:US'),
    ("svb_4", "20230312000000", "20230314000000", '("Credit Suisse" OR "regional banks" OR "bank rescue" OR "deposit guarantee") sourcecountry:US'),
    ("svb_5", "20230314000000", "20230316000000", '("regional banks" OR "First Republic" OR "bank stocks") sourcecountry:US'),
    ("cpi_mar", "20230314080000", "20230314230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_mar", "20230310080000", "20230310230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("cpi_apr", "20230412080000", "20230412230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_apr", "20230407080000", "20230407230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("fomc_may", "20230503120000", "20230504040000", CORE_QUERY),
    ("cpi_may", "20230510080000", "20230510230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("debt_1", "20230522000000", "20230524000000", '("debt ceiling" OR "debt limit" OR default) sourcecountry:US'),
    ("jobs_may", "20230505080000", "20230505230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("cpi_jun", "20230613080000", "20230613230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("debt_2", "20230601000000", "20230604000000", '("debt ceiling" OR "debt limit") sourcecountry:US'),
    ("fomc_jun", "20230614120000", "20230615040000", CORE_QUERY),
    ("jobs_jun", "20230602080000", "20230602230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("cpi_jul", "20230712080000", "20230712230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_jul", "20230707080000", "20230707230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("fomc_jul", "20230726120000", "20230727040000", CORE_QUERY),
    ("china_aug", "20230815000000", "20230818000000", '("Country Garden" OR "china property" OR "evergrande") sourcecountry:US'),
    ("cpi_aug", "20230810080000", "20230810230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_aug", "20230804080000", "20230804230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("cpi_sep", "20230913080000", "20230913230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_sep", "20230901080000", "20230901230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("fomc_sep", "20230920120000", "20230921040000", CORE_QUERY),
    ("cpi_oct", "20231012080000", "20231012230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("yields_oct", "20231019000000", "20231021000000", '("10-year treasury" OR "bond yields" OR "term premium") sourcecountry:US'),
    ("jobs_oct", "20231006080000", "20231006230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("fomc_nov", "20231101120000", "20231102040000", CORE_QUERY),
    ("cpi_nov", "20231114080000", "20231114230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_nov", "20231103080000", "20231103230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
    ("fomc_dec", "20231213120000", "20231214040000", CORE_QUERY),
    ("cpi_dec", "20231212080000", "20231212230000", '(CPI OR "consumer price" OR inflation) sourcecountry:US'),
    ("jobs_dec", "20231208080000", "20231208230000", '(("jobs report" OR "nonfarm payrolls")) sourcecountry:US'),
]

# ~30 routine (non-event) business days across 2023 for background noise
def _routine_days(n=30, seed=7):
    days = pd.bdate_range("2023-01-05", "2023-12-22")
    return list(pd.DatetimeIndex(days).to_series().sample(n, random_state=seed).sort_values())


# 60 additional routine days (different seed, December-weighted) to densify
# coverage; the first fetch left 587/631 trading days with zero articles.
BROAD_QUERY = '("stock market" OR "S&P 500" OR "dow jones" OR nasdaq OR "treasury yields") sourcecountry:US'


def _routine_days_extra(n=60, seed=11):
    days = pd.bdate_range("2023-01-03", "2023-12-29")
    dec = pd.DatetimeIndex(days).to_series()[lambda s: s.index.month == 12]
    rest = pd.DatetimeIndex(days).to_series()[lambda s: s.index.month != 12]
    take_dec = list(dec.sample(min(20, len(dec)), random_state=seed).sort_values())
    take_rest = list(rest.sample(n - len(take_dec), random_state=seed).sort_values())
    return take_dec + take_rest


def _gdelt_call(query, start, end, maxrecords=75, retries=3):
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(maxrecords),
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
        "sort": "hybridrel",
    }
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_API, params=params, timeout=60)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            time.sleep(GDELT_SLEEP_SEC)
        except Exception:
            time.sleep(GDELT_SLEEP_SEC)
    return {}


def _parse_artlist(js):
    rows = []
    for art in js.get("articles", []):
        seen = art.get("seendate", "")  # e.g. 20230310T134500Z
        try:
            ts = pd.to_datetime(seen, format="%Y%m%dT%H%M%SZ", utc=True)
        except Exception:
            continue
        title = (art.get("title") or "").strip()
        if not title:
            continue
        rows.append(
            {
                "timestamp_utc": ts,
                "source": art.get("domain", "unknown"),
                "headline": title,
                "summary": "",
                "url": art.get("url", ""),
            }
        )
    return rows


def fetch_gdelt_corpus(per_window=75):
    all_rows = []
    windows = [(lbl, s, e, q) for (lbl, s, e, q) in EVENT_WINDOWS]
    for d in _routine_days():
        stamp = d.strftime("%Y%m%d")
        windows.append((f"routine_{stamp}", f"{stamp}120000", f"{stamp}230000", CORE_QUERY))

    for i, (lbl, s, e, q) in enumerate(windows):
        js = _gdelt_call(q, s, e, maxrecords=per_window)
        rows = _parse_artlist(js)
        for r in rows:
            r["window"] = lbl
        all_rows.extend(rows)
        print(f"[{i+1}/{len(windows)}] {lbl}: {len(rows)} articles", flush=True)
        time.sleep(GDELT_SLEEP_SEC)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["headline", "window"]).reset_index(drop=True)
    return df


def augment_corpus(cache_file=None, per_window=60):
    """Fetch additional routine-day windows (broad query, December-weighted)
    and merge into the cached corpus. Returns the combined deduped frame."""
    import os

    cache_file = cache_file or f"{RAW_DIR}/news_gdelt_2023.csv"
    if os.path.exists(cache_file):
        base = pd.read_csv(cache_file)
        base["timestamp_utc"] = pd.to_datetime(base["timestamp_utc"], utc=True)
    else:
        base = fetch_gdelt_corpus()

    extra_rows = []
    days = _routine_days_extra()
    for i, d in enumerate(days):
        stamp = d.strftime("%Y%m%d")
        q = BROAD_QUERY if i % 2 == 0 else CORE_QUERY
        js = _gdelt_call(q, f"{stamp}120000", f"{stamp}230000", maxrecords=per_window)
        rows = _parse_artlist(js)
        for r in rows:
            r["window"] = f"routine2_{stamp}"
        extra_rows.extend(rows)
        print(f"[augment {i+1}/{len(days)}] {stamp}: {len(rows)}", flush=True)
        time.sleep(GDELT_SLEEP_SEC)

    extra = pd.DataFrame(extra_rows)
    combined = pd.concat([base, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline", "window"]).reset_index(drop=True)
    combined.to_csv(cache_file, index=False)
    print("AUGMENTED TOTAL:", len(combined))
    return combined


def load_corpus(cache_file=None, refresh=False):
    cache_file = cache_file or f"{RAW_DIR}/news_gdelt_2023.csv"
    import os

    if not refresh and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df
    df = fetch_gdelt_corpus()
    if df.empty:
        raise RuntimeError("GDELT returned no articles")
    df.to_csv(cache_file, index=False)
    return df


if __name__ == "__main__":
    c = load_corpus(refresh=True)
    print("TOTAL:", len(c))
    print(c["source"].value_counts().head(10))
    print(c.head(5).to_string())
