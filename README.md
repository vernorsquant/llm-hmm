# LLM-HMM: Importance-Weighted News Sentiment for HMM Market Regime Detection

**Status: proof-of-concept benchmark, complete.** Implements and evaluates exactly
4 regime-detection models across a binary regime switch and regime-conditioned
Mean-Variance Optimization (MVO) on 2023 data (evaluation window
2023-01-01 → 2023-12-31), with all article-level text inference cached to disk.
The original research proposal lives at [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md).

## Results (2023, net of 5 bps transaction costs)

| Strategy | Ann. Return | Ann. Vol | Net Sharpe | Max DD | Calmar | SVB lead |
|---|---|---|---|---|---|---|
| Buy & Hold SPY | 25.3% | 13.1% | 1.79 | −10.1% | 2.50 | — |
| 60/40 Static | 16.3% | 8.8% | 1.75 | −8.4% | 1.94 | — |
| M1 Price HMM [Switch] | 23.3% | 11.5% | 1.89 | −10.1% | 2.30 | 17d (carried-in) |
| M1 Price HMM [MVO] | 25.3% | 13.1% | 1.79 | −10.1% | 2.50 | 17d |
| M2 Price PCA-HMM [Switch] | 23.3% | 11.5% | 1.89 | −10.1% | 2.30 | 17d (carried-in) |
| M2 Price PCA-HMM [MVO] | 25.3% | 13.1% | 1.79 | −10.1% | 2.50 | 17d |
| **M3 Sentiment HMM [Switch]** | **29.5%** | 12.7% | **2.11** | **−8.3%** | **3.57** | none |
| M3 Sentiment HMM [MVO] | 19.7% | 10.9% | 1.71 | −9.5% | 2.07 | none |
| **M4 Sentiment PCA-HMM [Switch]** | 29.2% | 12.6% | **2.10** | −8.5% | 3.43 | none |
| M4 Sentiment PCA-HMM [MVO] | 19.3% | 10.8% | 1.69 | −9.6% | 2.00 | none |

![Cumulative returns](outputs/figures/cumulative_returns_all_models.png)

![Regime classification](outputs/figures/regime_classification_comparison.png)

![MVO weights](outputs/figures/mvo_weights_over_time.png)

![SVB transition timing](outputs/figures/transition_lead_time_svb.png)

### Reading the table — no sugar coating

- **Sentiment Switch strategies deliver the best net Sharpe (2.11 / 2.10 vs 1.89
  price-only) and the lowest drawdown (−8.3% vs −10.1%).** Directionally consistent
  with Nystrup et al.'s regime-based allocation drawdown result.
- **But the proposal's headline mechanism is NOT validated.** M3/M4 have *no bear
  entry at all* in the Feb 15 – Mar 13 SVB window — they entirely missed the March
  banking stress (only 15/22 bear days in all of 2023, none in March). The profit
  uplift comes from sidestepping the autumn selloff, not from earlier crisis detection.
- **The M1/M2 "17-day lead" is carried-in bear** (bear since January), not
  anticipation — which is exactly why the lead-time metric flags it.
- **MVO blending compresses everything toward B&H** (all MVO ≤ 1.79): probability-
  weighted blending rarely reaches corners, and in a +25% SPY year holding IEF drags.
- **Mechanism caveat:** M3/M4's bear state has legitimate market features (VIX 23.1
  vs 16.8, realized vol 19.3% vs 13.1%) but its sentiment features are exactly 0 and
  12 of 15 bear days fall in September — the month with the worst corpus coverage.
  The performance edge is real net-of-cost, but it is not yet cleanly attributable to
  news sentiment rather than residual coverage structure. Fixing that attribution
  (covered-days-only fitting or a denser corpus) is the obvious next step.

## The 4 models

| Model | Inputs | Architecture |
|---|---|---|
| M1 (baseline) | SPY return, 20d realized vol, VIX | 2-state Gaussian HMM |
| M2 (baseline) | same, PCA(95%) first | 2-state Gaussian HMM |
| M3 (proposed) | market + sentiment features | 2-state Gaussian HMM |
| M4 (proposed) | market + sentiment features, PCA(95%) | 2-state Gaussian HMM |

Sentiment features: daily importance-weighted sentiment `S_t = Σ(S_raw,i·C_i·I_i)/(Σ C_i·I_i + 1e-6)`,
its 5-day EMA, and sentiment dispersion, where `S_raw = P(pos) − P(neg)` (FinBERT),
`C_i = 1 − H(p)/ln 3` (normalized class entropy), and `I_i ∈ [0,1]` is a local-LLM
macro-importance rating (Qwen2.5-1.5B-Instruct Q4_K_M via llama.cpp, CPU).

**Frameworks.** *Switch*: P(bull) > 0.5 → 100% SPY, else 100% IEF.
*MVO*: regime-conditioned μ_k, Σ_k from state-mapped samples (annualized), blended
by the lagged posterior `p_t`, long-only max-Sharpe (SLSQP), 5 bps × one-sided turnover.
Benchmarks: SPY buy-and-hold and monthly-rebalanced 60/40, same cost model.

**Timing (no lookahead).** Articles published after 16:00 America/New_York map
strictly to the NEXT trading session; the signal known at the close of t−1
positions day t. Both are unit-tested (`tests/test_no_lookahead.py`).

## Pipeline

```
data ingestion → article scoring → daily features → 4 HMMs → backtests → artifacts
market_data.py    finbert_scoring.py  daily_features.py  regime_models.py  backtest.py  plots.py
news_corpus.py    llm_importance.py                      (state canon.)    metrics.py
```

- **Market data**: yfinance (SPY, IEF, adjusted) + CBOE CSV (VIX); features 2021-06→2023-12
  (warmup includes the 2022 bear market), evaluation on 2023 only.
- **News**: GDELT DOC 2.0 artlist — 2,117 real headlines with real intraday timestamps,
  334 outlets, event-keyed windows (FOMC, CPI, jobs, SVB, debt ceiling) + routine days.
  GDELT provides no summaries → raters run on headlines (disclosed).
- **Caching**: FinBERT probabilities parquet + LLM ratings JSONL, both sha1(text)-keyed —
  every headline passes through each model exactly once across all runs.
- **In-sample disclosure**: HMMs are fit on the full feature window (standard for this
  literature, e.g. Ang–Timmermann); the backtest evaluates 2023. `random_state=42` fixed.

## Key engineering findings (negative results included)

1. **Corpus-coverage artifact (caught and fixed).** In the first build, M3/M4's
   "bear" state was the *zero-news-coverage* state: 587/631 trading days had no
   articles, and zero-count news features made the HMM split on coverage
   (bear-days-by-month was identical to zero-coverage-days-by-month; bear-state mean
   article count 0.0 vs 22.0). Fixes: forward-fill level features (unobserved ≠
   neutral), exclude the zero-inflated article count from HMM inputs (still computed
   and reported), densify the corpus (+1,149 articles, December-weighted). After the
   fix: persistence 0.98–0.995, bear-state VIX ≈ 23 vs bull ≈ 17 — literature-consistent.
2. **PCA-HMM ≡ HMM here (expected, disclosed).** `PCA(0.95)` on 3 standardized features
   keeps all 3 (each carries ⅓ of variance), and the Gaussian-HMM likelihood is invariant
   to invertible linear maps — so M1/M2 agree to ~1e-3 in posteriors. On M4's 6 features
   PCA keeps 5 (≈98% variance) → near-identical to M3. With this few features, PCA is
   a structural no-op, not a bug.
3. **Lead-time metric honesty.** "First bear flip inside a window" inflates lead time for
   models already bear at window start. The reported metric uses the LAST bull→bear entry
   before the reference date (2023-03-13), flags carried-in states, and annotates entry
   dates on the figure.

## Comparison vs published baselines

Full extraction in [`papers/README.md`](papers/README.md) (every PDF title-verified).

- **FinBERT** (Araci 2019): published 0.86 accuracy / 0.84 F1 on Financial PhraseBank.
  Our layer reproduces the expected behavior on macro headlines (FOMC previews →
  high-confidence neutral; rate-risk headlines → negative).
- **2-state HMM stylized facts** (Ang & Timmermann, NBER w17182): high persistence,
  low-vol/positive-mean vs high-vol/negative-mean states. Our M1/M2: diagonal 0.994/0.993,
  bear-state VIX 24.6 vs bull 16.5, SVB week classified bear 100%.
- **Regime-based allocation** (Nystrup et al., QF 2017): regime switching should
  reduce drawdowns vs static weights — checked in the results table.
- **Nearest neighbor** (Li 2025, PLOS ONE e0331658): PCA-HMM + aggregate sentiment on
  tech stocks; no article-level weighting, no trading evaluation — differentiated here.

## Repository layout

```
src/hmm_sentiment/    config, market_data, news_corpus, finbert_scoring,
                      llm_importance, daily_features, regime_models,
                      backtest, metrics, plots
scripts/run_pipeline.py   end-to-end orchestrator
tests/                34 unit tests (lookahead, bounds, HMM validity, MVO constraints)
outputs/figures/      4 publication-grade PNGs (300 DPI)
outputs/tables/       metrics_summary.csv, model_diagnostics.json, article_scores_sample.csv
docs/                 original research proposal
papers/               PDFs (gitignored) + papers/README.md (verified citations + baseline numbers)
```

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py
.venv/bin/python -m pytest tests/ -q
```

## Limitations

In-sample HMM fitting (disclosed above) · GDELT headline-only corpus with residual
coverage gaps (forward-filled, limit 10 days) · 2-state constraint · single year ·
importance ratings from a 1.5B model at temperature 0 · transaction costs modeled as
5 bps one-sided turnover · results are a PoC, not a tradable strategy.

---

## Appendix: original research proposal (summary)

Hidden Markov models detect market regimes (bull/bear) from price and volatility, but they only
react *after* a regime change shows up in returns. News leads prices, yet raw daily news sentiment
is dominated by high-volume, low-importance articles and miscalibrated model output. This project
feeds article-level news observations — weighted by an **LLM importance prior** and **FinBERT
confidence** (confidence = 1 − class entropy) — into an HMM/PCA-HMM emission process, and evaluates
whether the resulting regime detection is *faster and more profitable* than a price-only HMM, using
trading strategies and portfolio allocation. Full proposal text (motivation, planned ablations,
team split, literature review): [`docs/RESEARCH_PROPOSAL.md`](docs/RESEARCH_PROPOSAL.md).

Key cited works: Hamilton (1989); Rydén et al. (1998); Nystrup et al. (2017/2019/2020);
Ang & Timmermann (2012); Tetlock (2007); Garcia & Norli (2021); Araci (2019, FinBERT);
Li (2025, PLOS ONE e0331658 — nearest neighbor).

## License

TBD
