"""Plot SPY price in the top panel and the BSM (LLM-Labeler) indicator in the bottom panel,
with the March 2023 SVB crash highlighted and labeled on both panels.
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


def main():
    # 1. Load daily BSM series and market prices
    sent_df = pd.read_csv(os.path.join(TABLE_DIR, "labeler_daily_series.csv"))
    sent_df["Date"] = pd.to_datetime(sent_df["Date"])

    market_df = fetch_market_data(start="2022-11-01", end="2023-12-31", use_cache=True).reset_index()
    market_df["Date"] = pd.to_datetime(market_df["Date"])

    df = pd.merge(sent_df, market_df, on="Date").sort_values("Date").reset_index(drop=True)
    df["bsm_ema5"] = df["labeler_sent_w"].ewm(span=5, adjust=False).mean()

    dates = df["Date"]
    spy = df["SPY_close"]
    bsm = df["labeler_sent_w"]
    bsm_ema = df["bsm_ema5"]

    # SVB crisis key dates
    svb_start = pd.Timestamp("2023-03-08")
    svb_end = pd.Timestamp("2023-03-20")
    svb_peak_stress = pd.Timestamp("2023-03-13")

    # Find exact SPY price and BSM values at SVB stress trough
    svb_window_df = df[(df["Date"] >= svb_start) & (df["Date"] <= svb_end)]
    svb_price_min_idx = svb_window_df["SPY_close"].idxmin()
    svb_price_min_date = df.loc[svb_price_min_idx, "Date"]
    svb_price_min_val = df.loc[svb_price_min_idx, "SPY_close"]

    svb_bsm_min_idx = svb_window_df["labeler_sent_w"].idxmin()
    svb_bsm_min_date = df.loc[svb_bsm_min_idx, "Date"]
    svb_bsm_min_val = df.loc[svb_bsm_min_idx, "labeler_sent_w"]

    # 2. Build 2-panel stacked figure sharing x-axis
    fig, (ax_price, ax_bsm) = plt.subplots(
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

    ax_price.set_title("SPY Price vs. BSM (LLM-Labeler) Impact Indicator — 2023", fontsize=14, fontweight="bold", pad=12)
    ax_price.set_ylabel("SPY Price ($)", fontsize=11, fontweight="bold")
    ax_price.set_ylim(spy.min() * 0.97, spy.max() * 1.03)
    ax_price.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_price.grid(True, alpha=0.3)

    # --- Bottom Panel: BSM Indicator ---
    # Bar colors based on positive vs negative score
    bar_colors = np.where(bsm >= 0, "#10B981", "#F43F5E")
    ax_bsm.bar(dates, bsm, width=1.0, color=bar_colors, alpha=0.35, label="Daily BSM Score (sent_w)")
    ax_bsm.plot(dates, bsm_ema, color="#0284C7", lw=2.0, label="BSM 5-Day EMA")
    ax_bsm.axhline(0, color="#334155", lw=1.0, ls="-", alpha=0.7)

    # Highlight SVB Crash Window
    ax_bsm.axvspan(svb_start, svb_end, color="#EF4444", alpha=0.18)
    ax_bsm.axvline(svb_peak_stress, color="#DC2626", ls="--", lw=1.3, alpha=0.85)

    # Annotate SVB Crash on BSM Panel
    ax_bsm.scatter([svb_bsm_min_date], [svb_bsm_min_val], color="#DC2626", s=55, zorder=5)
    ax_bsm.annotate(
        f"SVB Sentiment Plunge\n({svb_bsm_min_date.strftime('%b %d')}: {svb_bsm_min_val:.2f})",
        xy=(svb_bsm_min_date, svb_bsm_min_val),
        xytext=(svb_bsm_min_date + pd.Timedelta(days=18), svb_bsm_min_val + 0.05),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#DC2626", lw=1.2),
    )

    ax_bsm.set_ylabel("BSM Impact Score [−1, +1]", fontsize=11, fontweight="bold")
    ax_bsm.set_xlabel("Date (2023)", fontsize=11, fontweight="bold")
    ax_bsm.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_bsm.grid(True, alpha=0.3)

    # Date formatting
    locator = mdates.MonthLocator()
    ax_bsm.xaxis.set_major_locator(locator)
    ax_bsm.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    out_path = os.path.join(FIG_DIR, "spy_price_and_bsm_indicator.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
