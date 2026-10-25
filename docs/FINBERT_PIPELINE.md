# The FinBERT Sentiment Pipeline

Technical reference for the FinBERT layer that feeds models M3/M4 in this
repository. Everything here describes the code as implemented
(`src/hmm_sentiment/finbert_scoring.py`, `src/hmm_sentiment/scoring.py`,
`src/hmm_sentiment/daily_features.py`) and as verified by the project's
methodology audit.

## Role in the pipeline

FinBERT is the *domain-transformer sentiment source* for models M3 (FinBERT+HMM)
and M4 (FinBERT+PCA-HMM). It converts each news headline into three numbers:

| Output | Definition | Range |
|---|---|---|
| `S_raw` | P(positive) − P(negative) | [−1, +1] |
| `C` (confidence) | 1 − H(p)/ln 3, entropy of the class triple normalized by ln 3 | [0, 1] |
| weight `W` | C × I, where I is the LLM macro-importance rating | [0, 1] |

These are combined into the daily sentiment score
`S_t = Σ(S_raw,i · W_i) / (Σ W_i + 1e-6)`, then an EMA-5 and a dispersion
statistic are derived, and the three level features (`sent_w`, `sent_w_ema5`,
`sent_disp`) join returns, realized vol and VIX as HMM inputs.

## Model

- **Checkpoint**: `ProsusAI/finbert` (Araci 2019 — see `papers/README.md`).
  BERT-base, 438 MB safetensors, loaded via `transformers`.
- **Task head**: 3-way sequence classification with `config.id2label`:
  **{0: positive, 1: negative, 2: neutral}**.

⚠️ **Label-order trap.** The naive order one would assume is
{positive, neutral, negative}, but the checkpoint's actual mapping swaps
negative and neutral. The pipeline never trusts positional order: after each
softmax it reorders columns **by name** through
`[int(np.where([id2label[j] == lab for j in range(3)])[0][0]) for lab in ("positive", "neutral", "negative")]`.
The methodology audit verified the runtime ordering empirically (positive probe
→ P_pos = 0.946; negative probe → P_neg = 0.958) and confirmed the project
docstring originally asserted the wrong order — it has been corrected.

## Inference details

- **Device**: CPU only in this project (no GPU on the target box);
  `torch.set_num_threads(n_cores/2)`, `model.eval()`, `torch.no_grad()`.
- **Tokenization**: `truncation=True, max_length=128, padding=True`,
  batch size 64.
- **Text**: headline only. The GDELT corpus provides no article bodies, so
  `text = headline` (disclosed limitation; FinBERT was trained on
  headline/sentence-length financial text, so this is a natural fit).
- **Runtime**: ~2,069 unique headlines score in well under a minute on 16 CPU
  threads.

## Entropy-based confidence

For a probability triple p = (p⁺, pⁿ, p⁻):

```
H(p) = −Σ pᵢ ln(pᵢ + 1e-9)        (Shannon entropy, ε = 1e-9 per spec)
C    = clip(1 − H(p)/ln 3, 0, 1)
```

- Maximum-uncertainty uniform triple → H = ln 3 → C ≈ 0 (the 1e-9 ε leaves a
  ~2.7e-9 residual, covered by test tolerance).
- Concentrated triple (e.g. 0.94/0.05/0.01) → C ≈ 0.8–0.95.

**Behavior on this corpus** (audit-verified): the FinBERT cache contains
2,069 rows (full corpus coverage), zero duplicate keys, every row a valid
simplex summing to exactly 1, and **zero degenerate one-hot rows** — unlike the
VADER baseline (33.8% neutral point masses) and the FinGPT baseline
(100% one-hot ⇒ confidence ≡ 1). FinBERT is the only sentiment source in this
repo whose confidence layer carries real information, which is why it anchors
the weighting study.

## Caching: run-once guarantee

Every headline is keyed by `sha1(text)` into
`data/cache/finbert_scores.parquet` (`key, P_pos, P_neu, P_neg`). On any run:

1. Load the parquet into `{key → probs}`.
2. Score only the missing keys (batched, single model load).
3. Append the new keys and rewrite the parquet.

Consequences: text inference runs **strictly once per headline across all
runs**; re-running the whole pipeline after a code change costs zero FinBERT
compute; the cache is portable with the repo's data directory.

## From articles to daily features

`daily_features.build_daily_news_features`:

1. **Close safeguard**: articles are mapped to trading days first — ET
   timestamps ≥ 16:00 map strictly to the *next* session, weekend/holiday
   publishes map to the next open session (`map_articles_to_trading_days`,
   unit-tested in `tests/test_no_lookahead.py`).
2. **Dedupe**: identical headlines mapping to the *same* trading date are
   collapsed (overlapping GDELT windows refetch up to 7×; per-date dedupe
   removed a ~28% relative sentiment shift on FOMC days).
3. **Weighted aggregation** per the formula above; days with no coverage are
   **forward-filled** (limit 10 sessions) rather than treated as neutral —
   "unobserved" ≠ "neutral" (this distinction was a caught-and-fixed artifact;
   see the README's engineering-findings section).
4. `hi_imp_volume` (count of articles with importance ≥ 0.7) is computed and
   reported but deliberately **excluded from HMM inputs** — it is zero-inflated
   by corpus coverage and made an early model split states on news coverage
   instead of market regime.

## Known limitations (audit findings)

- **No summaries**: GDELT artlist supplies headlines only; FinBERT runs on
  headlines alone.
- **Coverage gaps**: 2023 corpus covers ~44% of trading days after
  December-weighted augmentation; forward-fill (limit 10) bridges the rest.
- **Crawl-time semantics**: GDELT `seendate` is a first-seen crawl timestamp,
  not publication time; ~31% of articles are first seen after 16:00 ET and map
  to t+1. The bias direction is anti-lookahead (seendate ≥ publication), but it
  dilutes same-day sentiment.
- **Confidence calibration**: entropy-confidence is *not* a calibrated
  probability of correctness (a known FinBERT limitation; the proposal lists
  calibration as future work).

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -c "
from hmm_sentiment.news_corpus import load_corpus
from hmm_sentiment.finbert_scoring import score_articles
s = score_articles(load_corpus())
print(s[['headline','S_raw','confidence']].describe())
"
```

First call downloads the checkpoint (~438 MB) and scores the full corpus in
under a minute on CPU; subsequent calls are pure cache reads.
