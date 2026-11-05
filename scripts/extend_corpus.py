"""Extend the GDELT news corpus beyond 2023 (2017-01-01 → today).

Event-keyed windows per year (approximate dates, disclosed — this is corpus
selection, not trading lookahead, and identical for every scorer):
  - 8 approximate FOMC windows/year
  - 12 CPI windows/year (~10th-13th)
  - 12 jobs-report windows/year (first Friday)
  - hardcoded crisis windows (2018 vol spike, Oct/Dec 2018 selloff, 2019 curve
    inversion, COVID crash Feb-Apr 2020, vaccine day, 2022 bear episodes, 2024
    yen-carry unwind, 2024 US election, 2025 tariff shock)
  - ~50 routine business days/year (CORE/BROAD queries alternating)

Rate-limited to GDELT's 1 call / 6s. Appends rows to
data/raw/news_gdelt_extended.csv incrementally (crash-resumable: already-fetched
windows are skipped via a window-log sidecar). 2023 is NOT refetched (the
original corpus covers it).
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment.config import RAW_DIR
from hmm_sentiment.news_corpus import (
    BROAD_QUERY,
    CORE_QUERY,
    GDELT_SLEEP_SEC,
    _gdelt_call,
    _parse_artlist,
)

OUT_CSV = os.path.join(RAW_DIR, "news_gdelt_extended.csv")
DONE_LOG = os.path.join(RAW_DIR, "news_gdelt_extended_windows_done.txt")
YEARS = list(range(2017, 2027))
SKIP_YEARS = {2023}
TODAY = pd.Timestamp.today().normalize()

FOMC_APPROX = [(1, 28), (3, 17), (5, 1), (6, 16), (7, 28), (9, 15), (11, 2), (12, 14)]

CRISIS = [
    ("crisis_20180205", "2018-02-02", "2018-02-07"),
    ("crisis_20181010", "2018-10-09", "2018-10-12"),
    ("crisis_20181220", "2018-12-19", "2018-12-25"),
    ("crisis_20190814", "2019-08-13", "2019-08-16"),
    ("crisis_20200224", "2020-02-24", "2020-03-01"),
    ("crisis_20200309", "2020-03-09", "2020-03-15"),
    ("crisis_20200316", "2020-03-16", "2020-03-22"),
    ("crisis_20200323", "2020-03-23", "2020-04-10"),
    ("crisis_20201109", "2020-11-09", "2020-11-12"),
    ("crisis_20220120", "2022-01-19", "2022-01-25"),
    ("crisis_20220509", "2022-05-08", "2022-05-13"),
    ("crisis_20220613", "2022-06-12", "2022-06-17"),
    ("crisis_20220913", "2022-09-12", "2022-09-15"),
    ("crisis_20220926", "2022-09-25", "2022-09-30"),
    ("crisis_20240805", "2024-08-04", "2024-08-08"),
    ("crisis_20241105", "2024-11-04", "2024-11-08"),
    ("crisis_20250403", "2025-04-02", "2025-04-10"),
]


def _first_friday(year, month):
    d = pd.Timestamp(year, month, 1)
    offset = (4 - d.dayofweek) % 7
    return d + pd.Timedelta(days=offset)


def windows_for_year(year):
    wins = []
    for mo, dy in FOMC_APPROX:
        s = pd.Timestamp(year, mo, dy) - pd.Timedelta(days=1)
        e = pd.Timestamp(year, mo, dy) + pd.Timedelta(days=1)
        wins.append((f"xfomc_{year}{mo:02d}{dy:02d}", s, e, CORE_QUERY))
    for mo in range(1, 13):
        wins.append((f"xcpi_{year}{mo:02d}", pd.Timestamp(year, mo, 10), pd.Timestamp(year, mo, 13),
                     "(CPI OR \"consumer price\" OR inflation) sourcecountry:US"))
        ff = _first_friday(year, mo)
        wins.append((f"xjobs_{year}{mo:02d}", ff, ff,
                     "((\"jobs report\" OR \"nonfarm payrolls\" OR unemployment)) sourcecountry:US"))
    for lbl, s, e in CRISIS:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if s_ts.year == year:
            wins.append((lbl, s_ts, e_ts, CORE_QUERY))
    return wins


def routine_days_for_year(year, n=50, seed=None):
    days = pd.bdate_range(f"{year}-01-03", f"{year}-12-29")
    seedval = seed if seed is not None else year
    take = days.to_series().sample(min(n, len(days)), random_state=seedval).sort_values()
    return list(take)


def main():
    done = set()
    if os.path.exists(DONE_LOG):
        with open(DONE_LOG) as f:
            done = {line.strip() for line in f if line.strip()}

    all_windows = []
    for year in YEARS:
        if year in SKIP_YEARS:
            continue
        for lbl, s, e, q in windows_for_year(year):
            e_clip = min(e, TODAY + pd.Timedelta(days=1))
            if s > TODAY:
                continue
            all_windows.append((lbl, s, e_clip, q))
        for d in routine_days_for_year(year):
            if d > TODAY:
                continue
            stamp = d.strftime("%Y%m%d")
            q = CORE_QUERY if int(stamp[-1]) % 2 == 0 else BROAD_QUERY
            all_windows.append((f"xroutine_{stamp}", d, d, q))

    todo = [w for w in all_windows if w[0] not in done]
    print(f"[extend] windows total={len(all_windows)} done={len(all_windows) - len(todo)} todo={len(todo)}",
          flush=True)

    rows = []
    for i, (lbl, s, e, q) in enumerate(todo):
        s_utc = s.strftime("%Y%m%d") + "120000"
        e_utc = (e + pd.Timedelta(days=1)).strftime("%Y%m%d") + "040000"
        js = _gdelt_call(q, s_utc, e_utc, maxrecords=75)
        parsed = _parse_artlist(js)
        for r in parsed:
            r["window"] = lbl
        rows.extend(parsed)
        with open(DONE_LOG, "a") as f:
            f.write(lbl + "\n")
        if i % 10 == 0 or i == len(todo) - 1:
            print(f"[extend] {i + 1}/{len(todo)} windows | {lbl}: {len(parsed)} articles | total rows {len(rows)}",
                  flush=True)
        # incremental flush every 40 windows
        if rows and (i % 40 == 39 or i == len(todo) - 1):
            _flush(rows)

    print("[extend] corpus extension fetch complete", flush=True)
    print("EXTEND_FETCH_DONE", flush=True)


def _flush(new_rows):
    df = pd.DataFrame(new_rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    combined = combined.drop_duplicates(subset=["headline", "window"]).reset_index(drop=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"[extend] flushed -> {len(combined)} unique rows", flush=True)


if __name__ == "__main__":
    main()
