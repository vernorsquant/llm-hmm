# Baseline Papers — downloaded and title-verified

Every PDF below was opened with pymupdf and its **first-page title verified**
(the standard failure mode of arXiv/NBER ID guessing is a well-formed PDF with
the wrong content — caught exactly that with NBER w17168 before replacing it
with the correct w17182).

| File | Verified title | Role in this project |
|---|---|---|
| `finbert_araci_2019.pdf` | *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* (Araci, 2019) | Sentiment layer; published accuracy baseline |
| `ang_timmermann_2012_regimes.pdf` | *Regime Changes and Financial Markets* (Ang & Timmermann, NBER w17182) | Regime-detection stylized facts |
| `nystrup_2017_dynamic_portfolio_hmm.pdf` | *Dynamic Portfolio Optimization Across Hidden Market Regimes* (Nystrup, Madsen & Lindström, Quantitative Finance) | Regime-based allocation baseline |
| `plos_one_2025_pca_hmm_sentiment.pdf` | *Momentum, volume and investor sentiment study for U.S. technology sector stocks — a hidden Markov model based principal component analysis* (Li, PLOS ONE 2025, e0331658) | Nearest-neighbor work (PCA-HMM + sentiment), differentiated in README |

## Extracted baseline numbers our results are checked against

### 1. FinBERT sentiment quality (Araci 2019, Table: Financial PhraseBank)
- FinBERT accuracy **0.86 / F1 0.84** (all data); **0.97 / 0.95** (100%-agreement subset)
- Beats LSTM 0.71, ULMFit 0.83, HSC 0.76 baselines.
- Our check: probability distribution sanity on macro headlines (FOMC previews
  score neutral-high-confidence; rate-risk headlines negative) — confirmed in
  the run log.

### 2. 2-state HMM regime stylized facts (Ang & Timmermann w17182)
- Regimes in equities "correspond to periods of high and low volatility, and
  long bull and bear market periods"; regime behavior "persists for many
  periods" (high transition-matrix persistence).
- Our baseline reproduction (2021-06→2023-12 in-sample fit, SPY/IEF/VIX):
  - Transition matrix diagonal **0.994 / 0.993** (high persistence ✓)
  - Bull state mean VIX **16.5** vs bear state **24.6** ✓
  - Bull state mean daily return **+0.08%** vs bear **−0.03%** ✓
  - March 2023 (SVB) window classified bear **100%** of days ✓
  - Jan–Feb 2023 (post-2022 hangover, VIX>20) bear; Nov–Dec rally bull ✓

### 3. Regime-based allocation value (Nystrup et al., QF 2017)
- RBAA with HMM regimes "adds value over rebalancing to static weights and,
  in particular, reduces potential drawdowns". Our backtest table reports
  net-of-cost Sharpe and max drawdown for regime-conditioned MVO vs static
  60/40 to check the same direction of effect.

### 4. Nearest neighbor (Li 2025)
- PCA-HMM with an investor-sentiment series on tech stocks; finds sentiment
  informative for regime structure. Differs from this PoC: aggregate sentiment
  index (no article-level FinBERT, no importance/confidence weighting), no
  trading/portfolio evaluation, different market/period.
