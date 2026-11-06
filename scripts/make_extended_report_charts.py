"""Multi-year report charts (2017-2026) + news source quality assessment.

Generates:
  extended_news_coverage.png      articles/month + % days covered per year
  extended_yearly_sharpe.png      yearly switch Sharpe: M1/M3/M5 vs B&H
  extended_finbert_vader_corr.png full-window FinBERT~VADER scatter + rolling corr
  news_outlets_quality.png        top outlets colored by quality tier
  news_quality_summary.csv        tier shares + top-outlet table
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment.config import FIG_DIR, TABLE_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER_MAP = {
    # tier 1: wire services + major financial press
    "reuters.com": 1, "apnews.com": 1, "bloomberg.com": 1, "wsj.com": 1, "ft.com": 1,
    "cnbc.com": 1, "marketwatch.com": 1, "barrons.com": 1, "nytimes.com": 1,
    "foxbusiness.com": 1, "investors.com": 1, "economictimes.indiatimes.com": 1,
    # tier 2: specialist / retail finance platforms (original analysis, uneven QA)
    "seekingalpha.com": 2, "fool.com": 2, "benzinga.com": 2, "investing.com": 2,
    "insidermonkey.com": 2, "zacks.com": 2, "thestreet.com": 2, "gurufocus.com": 2,
    "tipranks.com": 2, "4-traders.com": 2, "markets.businessinsider.com": 2,
    "streetinsider.com": 2, "marketscreener.com": 2, "finance.yahoo.com": 2,
    "insurancenewsnet.com": 2, "moomoo.com": 2, "stocktwits.com": 2, "mid-day.com": 2,
    # tier 3: aggregators / press releases / opinion
    "msn.com": 3, "news.google.com": 3, "feedx.net": 3, "prnewswire.com": 3,
    "businesswire.com": 3, "globenewswire.com": 3, "zerohedge.com": 3,
    "finanznachrichten.de": 3, "markets.businessinsider.com": 2,
}
TIER_LABEL = {1: "Wire/major press", 2: "Specialist finance", 3: "Aggregator/press-release", 0: "Other"}


def load_combined():
    n23 = pd.read_csv(os.path.join(ROOT, "data", "raw", "news_gdelt_2023.csv"))
    ext = pd.read_csv(os.path.join(ROOT, "data", "raw", "news_gdelt_extended.csv"))
    for df in (n23, ext):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    comb = pd.concat([n23, ext], ignore_index=True).drop_duplicates(subset=["headline", "window"])
    comb["month"] = comb["timestamp_utc"].dt.to_period("M")
    comb["year"] = comb["timestamp_utc"].dt.year
    return comb


def chart_coverage(comb, daily, path):
    per_month = comb.groupby("month").size()
    idx = per_month.index.to_timestamp()
    cov_year = daily.groupby(daily.index.year).apply(
        lambda s: 100.0 * ((s["finbert_n"] > 0).mean()))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=False)
    ax.bar(idx, per_month.values, width=22, color="#1f77b4", alpha=0.85)
    ax.set_ylabel("articles / month")
    ax.set_title("News corpus coverage over time — 15,037 articles, 2017-01 → 2026-11")
    ax.grid(alpha=0.3, axis="y")

    years = cov_year.index.astype(int)
    ax2.bar(years, cov_year.values, color="#2ca02c", alpha=0.85)
    for y, v in zip(years, cov_year.values):
        ax2.text(y, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax2.set_ylabel("% trading days with ≥1 article")
    ax2.set_title("Per-year news coverage of trading days (the rest is forward-filled, disclosed)")
    ax2.set_ylim(0, 60)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def chart_yearly_sharpe(path):
    yearly = pd.read_csv(os.path.join(TABLE_DIR, "extended_yearly.csv"))
    piv = yearly.pivot_table(index="year", columns="model", values="sharpe")
    models = [
        ("buy_hold_spy", "Buy & Hold SPY", "black"),
        ("M1_price_hmm", "Price-only HMM", "#7f7f7f"),
        ("M3_finbert_hmm", "FinBERT HMM", "#2ca02c"),
        ("M5_vader_hmm", "VADER HMM", "#ff7f0e"),
    ]
    years = piv.index.values.astype(int)
    x = np.arange(len(years))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, label, color) in enumerate(models):
        vals = piv[m].reindex(years).values
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("OOS switch Sharpe (net 5 bps)")
    ax.set_title("Year-by-year walk-forward switch Sharpe — every model loses to B&H in 2019/2021/2024 bulls,\n"
                 "and VADER ≈ FinBERT in most years (differences inside noise)", fontsize=10.5)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def chart_finbert_vader_corr(daily, path):
    fb = daily["finbert_sent_w"].dropna()
    vd = daily["vader_sent_w"].reindex(fb.index).dropna()
    common = fb.index.intersection(vd.index)
    fb, vd = fb.loc[common], vd.loc[common]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [1.3, 1.0]})
    rolling = fb.rolling(120).corr(vd)
    ax.plot(rolling.index, rolling.values, lw=1.4, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-0.4, 1)
    ax.set_title("FinBERT ~ VADER rolling 120-day correlation (full window)")
    ax.set_ylabel("r(t)")
    ax.grid(alpha=0.3)

    fbz = (fb - fb.mean()) / fb.std()
    vdz = (vd - vd.mean()) / vd.std()
    r = float(fb.corr(vd))
    ax2.scatter(fbz, vdz, s=8, alpha=0.4, color="#1f77b4", edgecolor="none")
    lim = 3.2
    ax2.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=1)
    ax2.set_xlabel("FinBERT (z-score)")
    ax2.set_ylabel("VADER (z-score)")
    ax2.set_title(f"Full-window scatter, Pearson r = {r:.2f}")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def chart_outlets(comb, path, summary_csv):
    counts = comb["source"].value_counts()
    top = counts.head(18)
    tiers = [TIER_MAP.get(d, 0) for d in top.index]
    tier_names = {1: "Wire/major press", 2: "Specialist finance", 3: "Aggregator/press-release", 0: "Other"}
    tier_colors = {1: "#2ca02c", 2: "#1f77b4", 3: "#d62728", 0: "#999999"}

    tier_counts = {t: 0 for t in (1, 2, 3, 0)}
    tier_outlets = {t: set() for t in (1, 2, 3, 0)}
    for d, c in counts.items():
        t = TIER_MAP.get(d, 0)
        tier_counts[t] += c
        tier_outlets[t].add(d)
    total = int(counts.sum())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})
    bars = ax.barh(range(len(top))[::-1], top.values,
                   color=[tier_colors[t] for t in tiers], alpha=0.9)
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(top.index, fontsize=9)
    for b, v in zip(bars, top.values):
        ax.text(v + 8, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8.5)
    ax.set_xlabel("articles")
    ax.set_title(f"Top 18 outlets of {total:,} articles / {counts.shape[0]:,} unique outlets")
    ax.grid(alpha=0.3, axis="x")

    tiers_present = [t for t in (1, 2, 3, 0) if tier_counts[t] > 0]
    vals = [100.0 * tier_counts[t] / total for t in tiers_present]
    labels = [f"{TIER_LABEL[t]}\n({len(tier_outlets[t])} outlets)" for t in tiers_present]
    ax2.bar(range(len(tiers_present)), vals,
            color=[tier_colors[t] for t in tiers_present], alpha=0.9)
    ax2.set_xticks(range(len(tiers_present)))
    ax2.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("% of corpus")
    ax2.set_title("Corpus share by outlet tier")
    ax2.set_ylim(0, 80)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    rows = []
    for t in (1, 2, 3, 0):
        if tier_counts[t] == 0:
            continue
        rows.append({"tier": t, "tier_label": TIER_LABEL[t], "articles": tier_counts[t],
                     "share_pct": round(100.0 * tier_counts[t] / total, 1),
                     "unique_outlets": len(tier_outlets[t]),
                     "examples": "; ".join(sorted(tier_outlets[t])[:8])})
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return path


def main():
    comb = load_combined()
    daily = pd.read_csv(os.path.join(TABLE_DIR, "extended_daily_series.csv"),
                        index_col=0, parse_dates=True)
    p1 = chart_coverage(comb, daily, f"{FIG_DIR}/extended_news_coverage.png")
    p2 = chart_yearly_sharpe(f"{FIG_DIR}/extended_yearly_sharpe.png")
    p3 = chart_finbert_vader_corr(daily, f"{FIG_DIR}/extended_finbert_vader_corr.png")
    p4 = chart_outlets(comb, f"{FIG_DIR}/news_outlets_quality.png",
                       os.path.join(TABLE_DIR, "news_quality_summary.csv"))
    for p in (p1, p2, p3, p4):
        print("saved", p, flush=True)
    print("EXT_REPORT_CHARTS_DONE", flush=True)


if __name__ == "__main__":
    main()
