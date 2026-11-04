# LLM-Labeler Pipeline — Technical Reference

Adapts [zenosim/LLM-Labeler](https://github.com/zenosim/LLM-Labeler)
(`simple_bsm.py`, rubric-decomposition impact scoring) as an alternative
article-level scorer for the HMM regime pipeline, and evaluates it head-to-head
against the incumbent FinBERT sentiment features on identical 2023 windows.

## What the pipeline does (per article)

1. **Plan (`branch_impact_rubric`)**: the LLM proposes up to 4 stock-impact
   evaluation criteria for the article, each with its own rubric.
2. **Solve (`solve_criterion`)**: per criterion, the LLM assigns an integer
   impact score in [−5, +5] with a rationale, judging *solely* on that rubric.
3. **Aggregate**: average criterion score → `Bullish` (≥ +1.5) /
   `Bearish` (≤ −1.5) / `Neutral` (otherwise).

The prompts are kept faithful to the original; the differences are disclosed
below.

## Adaptations from the original repo

| Aspect | Original | This repo | Why |
|---|---|---|---|
| Backend | OpenRouter `llama-3.1-70b` | **OpenRouter `llama-3.1-8b-instruct`** for the full corpus (temperature 0, `max_tokens` 250–700). GLM `glm-5.3-flash` was used for an early partial pass (~305 articles); those labels live under a separate cache key space and are NOT used in the headline comparison | OpenRouter is the labeler's native backend; single-backend labels avoid a GLM→llama backend switch that would be confounded with article type and calendar (audit MAJOR-2) |
| Cache keys | none | `sha1(headline) + backend + model + version` — labels are never shared across backends/models/pipeline versions, and every record carries `backend`, `model`, `version` | Full provenance; a backend or prompt change can never silently reuse stale labels (audit MAJOR-2 / MINOR-7) |
| Failure handling | bare `json.loads` | balanced-bracket JSON extraction, retries, then **degraded results are not cached** — fallback rubrics / unparseable solves stay in-memory and the article retries on the next run | A permanently-cached score-0 failure is an asymmetric shrink-to-neutral channel FinBERT does not have (audit MAJOR-4) |
| Ticker | per-article ticker | fixed `SPY` | The corpus is macro/index news; SPY/IEF are the traded assets |
| Output scale | avg score (−5..5) + 3-class | `S_raw = avg_impact / 5` ∈ [−1, 1] | Same scale as FinBERT `P_pos − P_neg`, so the identical feature builder applies. Raw scales still differ in spread — the HMM's per-model StandardScaler absorbs this; the correlation figure z-scores the scatter panel and reports raw stds (audit MINOR-5) |
| `confidence` | — (none) | `1 − std(criterion scores)/5`, clipped to [0.25, 1] | Dispersion across the article's own rubric branches; inactive while `WEIGHTED_SENTIMENT = False`, reported for completeness |
| Aggregation | mean, ±1.5 cut | criteria capped at 4 (prompt contract enforced), classification on the **unrounded** mean | audit MINOR-6 |

## Bias controls (inherited + audit-driven)

- **Close safeguard**: articles after 16:00 ET map to the next trading session
  (`daily_features.map_articles_to_trading_days`) — reused unchanged.
- **Coverage hygiene**: sentiment levels are forward-filled (limit 10) *before*
  zero-fill; raw news-count features are excluded from HMM inputs; articles
  with failed labels are **excluded from daily aggregation** (a NaN can no
  longer poison a whole day) and `n_labeled` counts them honestly
  (audit MAJOR-3). `run_labeler_zoo.py` aborts unless pre-drop labeler
  coverage is at least 0.995; tiny deficits (currently one
  persistently-failing headline) are DROPPED from feature building with
  per-headline disclosure.
- **Causal walk-forward**: every OOS posterior comes from the causal forward
  (Hamilton) filter with monthly prior-data-only refits
  (`regime_models.filtered_month_posterior`) — never `predict_proba` smoothing.
- **Canonicalization fix (audit CRITICAL, repo-wide)**:
  `filtered_month_posterior` previously canonicalized PCA models in PCA-rotated
  space (indexing PC3 instead of VIX), which could flip bull/bear in some
  monthly refits. It now canonicalizes from the standardized pre-PCA space,
  mirroring `_fit_one`, with a regression test
  (`tests/test_pca_canonicalization.py`) that fails on the old code. This
  changes some OOS rows of ALL PCA models (M2/M4/M6/M8/M10) in walk-forward
  tables relative to pre-audit runs.
- **MVO discipline**: weights lagged t−1 → t; regime moments from an expanding
  prior-only window; `rets` sliced to the eval index before `run_mvo`.
- **Identical windows**: labeler, FinBERT, FinGPT, VADER and price-only models
  are evaluated on exactly the same 2023 evaluation window with the same 5 bps
  one-sided cost model.

## Known scope limits (disclosed)

- **Corpus selection**: GDELT windows are hindsight-selected 2023 event dates
  (FOMC/CPI/SVB/debt ceiling) plus routine days. Shared by every scorer, so
  the A/B comparison is fair, but absolute "beats B&H" claims inherit
  event-densified news timing.
- **Warmup**: the news corpus starts 2023-01, so monthly refits before Feb 2023
  train on zero sentiment — January OOS cells are effectively price-only for
  every sentiment model. Symmetric across models.
- **Significance family is pre-registered**: the 5 reported pairs
  (M9 vs M1/M3/M7, M3 vs M1, M9 vs B&H — switch strategy, OOS) were fixed
  before the run; the full 50-cell zoo table is published unfiltered alongside.

## Files

- `src/hmm_sentiment/labeler_scoring.py` — backend-selectable client (OpenRouter/GLM), prompts, JSON
  extraction, cache, `labeler_scores(articles)`.
- `scripts/label_corpus.py` — labels the full corpus, writes
  `outputs/tables/labeler_scores_sample.csv` (30 articles across the score
  spectrum with per-criterion breakdowns).
- `scripts/run_labeler_zoo.py` — feature build, walk-forward OOS zoo
  (10 models × 5 strategies), in-sample companion, significance battery,
  diagnostics, and the two figures.
- `tests/test_labeler_scoring.py` — pure-function tests (JSON extraction,
  parsers, aggregation thresholds, scale mapping); network stubbed.

## Reproduce

```bash
cd /root/workspace/llm-hmm
source .env                        # provides OPENROUTER_API_KEY (labeler backend)
export LLM_LABELER_BACKEND=openrouter LLM_LABELER_WORKERS=16
.venv/bin/python scripts/label_corpus.py     # instant if the 8b cache is complete
.venv/bin/python scripts/run_labeler_zoo.py  # zoo + figures + significance
.venv/bin/python -m pytest tests/test_labeler_scoring.py -q
```

Backend note: the published labels were produced with
`LLM_LABELER_MODEL=meta-llama/llama-3.1-8b-instruct` (the default). Composite
cache keys make every backend/model a separate label space — rerunning with a
different model labels from scratch rather than reusing other models' labels.

## Results

See `README.md` → "LLM-Labeler vs sentiment baselines" section and
`outputs/tables/labeler_*.csv` for the numbers produced by the final run.
