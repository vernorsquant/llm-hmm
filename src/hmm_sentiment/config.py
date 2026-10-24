"""Central configuration: paths, evaluation window, costs, constants."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

# Evaluation window (calendar year 2023); features start earlier for warmup.
EVAL_START = "2023-01-01"
EVAL_END = "2023-12-31"
FEATURE_START = "2021-06-01"  # warmup: includes the 2022 bear market for regime variety

TRADING_DAYS_PER_YEAR = 252
COST_BPS = 5.0                    # 5 bps transaction cost per unit of one-sided turnover
MARKET_CLOSE_ET = "16:00"         # articles published after this map to t+1
HIGH_IMPORTANCE_THRESHOLD = 0.7   # article-level LLM importance threshold for news volume

# March 2023 banking stress reference dates (SVB failure -> regional bank stress peak)
SVB_REF_DATE = "2023-03-13"       # first trading day after SVB failure (stress realized in prices)
SVB_WINDOW_START = "2023-02-15"
SVB_WINDOW_END = "2023-03-31"

ASSETS = ["SPY", "IEF"]

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, OUT_DIR, FIG_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)
