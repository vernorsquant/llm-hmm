"""Plot SPY price in the top panel and the sentiment/impact indicator in the bottom panel,
with the March 2023 SVB crash highlighted and labeled on both panels.
Generates:
  1. outputs/figures/spy_price_and_bsm_indicator.png
  2. outputs/figures/spy_price_and_finbert_indicator.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from hmm_sentiment.config import FIG_DIR, TABLE_DIR
from hmm_sentiment.market_data import fetch_market_data

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_price_and_indicator(df, col_name, ema_col, title_name, short_name, line_color, out_filename):
    dates = df["Date"]
    spy = df["SPY_close"]
    score = df[col_name]
    score_ema = df[ema_col]

    # SVB crisis key dates
    svb_start = pd.Timestamp("2023-03-08")
    svb_end = pd.Timestamp("2023-03-20")
    svb_peak_stress = pd.Timestamp("2023-03-13")

    # Find exact SPY price and indicator values at SVB stress trough
    svb_window_df = df[(df["Date"] >= svb_start) & (df["Date"] <= svb_end)]
    svb_price_min_idx = svb_window_df["SPY_close"].idxmin()
    svb_price_min_date = df.loc[svb_price_min_idx, "Date"]
    svb_price_min_val = df.loc[svb_price_min_idx, "SPY_close"]

    svb_score_min_idx = svb_window_df[col_name].idxmin()
    svb_score_min_date = df.loc[svb_score_min_idx, "Date"]
    svb_score_min_val = df.loc[svb_score_min_idx, col_name]

    fig, (ax_price, ax_ind) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.12}
    )

    # --- Top Panel: SPY Price ---
    ax_price.plot(dates, spy, color="#0F172A", lw=2.0, label="SPY Close Price ($)")
    ax_price.fill_between(dates, spy.min() * 0.98, spy, color="#3B82F6", alpha=0.08)

    # Highlight SVB Crash Window
    ax_price.axvspan(svb_start, svb_end, color="#EF4444", alpha=0.18, label="SVB Banking Crisis Window (Mar 8–20)")
    ax_price.axvline(svb_peak_stress, color="#DC2626", ls="--", lw=1.3, alpha=0.85)

    # Annotate SVB Crash on Price Panel
    ax_price.scatter([svb_price_min_date], [svb_price_min_val], color="#DC2626", s=55, zorder=5)
    ax_price.annotate(
        f"SVB Crash Trough\n({svb_price_min_date.strftime('%b %d')}: ${svb_price_min_val:.2f})",
        xy=(svb_price_min_date, svb_price_min_val),
        xytext=(svb_price_min_date + pd.Timedelta(days=18), svb_price_min_val - 6),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#DC2626", lw=1.2),
    )

    ax_price.set_title(f"SPY Price vs. {title_name} — 2023", fontsize=14, fontweight="bold", pad=12)
    ax_price.set_ylabel("SPY Price ($)", fontsize=11, fontweight="bold")
    ax_price.set_ylim(spy.min() * 0.97, spy.max() * 1.03)
    ax_price.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_price.grid(True, alpha=0.3)

    # --- Bottom Panel: Indicator ---
    bar_colors = np.where(score >= 0, "#10B981", "#F43F5E")
    ax_ind.bar(dates, score, width=1.0, color=bar_colors, alpha=0.35, label=f"Daily {short_name} Score (sent_w)")
    ax_ind.plot(dates, score_ema, color=line_color, lw=2.0, label=f"{short_name} 5-Day EMA")
    ax_ind.axhline(0, color="#334155", lw=1.0, ls="-", alpha=0.7)

    # Highlight SVB Crash Window
    ax_ind.axvspan(svb_start, svb_end, color="#EF4444", alpha=0.18)
    ax_ind.axvline(svb_peak_stress, color="#DC2626", ls="--", lw=1.3, alpha=0.85)

    # Annotate SVB Crash on Indicator Panel
    ax_ind.scatter([svb_score_min_date], [svb_score_min_val], color="#DC2626", s=55, zorder=5)
    ax_ind.annotate(
        f"SVB Sentiment Plunge\n({svb_score_min_date.strftime('%b %d')}: {svb_score_min_val:.2f})",
        xy=(svb_score_min_date, svb_score_min_val),
        xytext=(svb_score_min_date + pd.Timedelta(days=18), svb_score_min_val + 0.08),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#DC2626", lw=1.2),
    )

    ax_ind.set_ylabel(f"{short_name} Score [−1, +1]", fontsize=11, fontweight="bold")
    ax_ind.set_xlabel("Date (2023)", fontsize=11, fontweight="bold")
    ax_ind.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_ind.grid(True, alpha=0.3)

    locator = mdates.MonthLocator()
    ax_ind.xaxis.set_major_locator(locator)
    ax_ind.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    out_path = os.path.join(FIG_DIR, out_filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main():
    sent_df = pd.read_csv(os.path.join(TABLE_DIR, "labeler_daily_series.csv"))
    sent_df["Date"] = pd.to_datetime(sent_df["Date"])

    market_df = fetch_market_data(start="2022-11-01", end="2023-12-31", use_cache=True).reset_index()
    market_df["Date"] = pd.to_datetime(market_df["Date"])

    df = pd.merge(sent_df, market_df, on="Date").sort_values("Date").reset_index(drop=True)
    df["bsm_ema5"] = df["labeler_sent_w"].ewm(span=5, adjust=False).mean()
    df["finbert_ema5"] = df["finbert_sent_w"].ewm(span=5, adjust=False).mean()

    plot_price_and_indicator(
        df,
        col_name="labeler_sent_w",
        ema_col="bsm_ema5",
        title_name="BSM (LLM-Labeler) Impact Indicator",
        short_name="BSM",
        line_color="#0284C7",
        out_filename="spy_price_and_bsm_indicator.png",
    )

    plot_price_and_indicator(
        df,
        col_name="finbert_sent_w",
        ema_col="finbert_ema5",
        title_name="FinBERT Sentiment Indicator",
        short_name="FinBERT",
        line_color="#7C3AED",
        out_filename="spy_price_and_finbert_indicator.png",
    )


if __name__ == "__main__":
    main()
