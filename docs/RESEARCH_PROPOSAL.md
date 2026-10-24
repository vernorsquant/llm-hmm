# LLM-HMM: Importance-Weighted News Sentiment for HMM Market Regime Detection

**Status:** research proposal (draft v0.1)

## One-paragraph pitch

Hidden Markov models detect market regimes (bull/bear) from price and volatility, but they only
react *after* a regime change shows up in returns. News leads prices, yet raw daily news sentiment
is dominated by high-volume, low-importance articles and miscalibrated model output. This project
feeds article-level news observations — weighted by a **news importance prior** (outlet credibility
× news type × novelty) and **LLM confidence** (FinBERT sentiment, confidence = 1 − class entropy) —
into a **PCA-HMM** emission process, and evaluates whether the resulting regime detection is
*faster and more profitable* than a price-only HMM, using simple trading strategies and portfolio
allocation models.

## Motivation

- HMM regime detection on price/volume is mature (Hamilton 1989 → Rydén et al. 1998 → Nystrup et
  al. 2019) but backward-looking by construction.
- News sentiment demonstrably moves markets (Tetlock 2007; Garcia & Norli 2021), but aggregate
  sentiment series drown important articles in noise.
- We attack the measured noise component directly: down-weight unimportant outlets/topics and
  low-confidence predictions, then test whether the regime model times transitions earlier.

## Key contributions

1. First pipeline feeding **importance- and confidence-weighted** article-level news observations
   into a PCA-HMM's emission process for macro regime detection (prior work uses sentiment as an
   exogenous regressor, or unweighted sentiment in direct predictors).
2. We test the regime model the way it's actually used in practice — plugged into **portfolio
   allocation and actual trading strategies**, not just checked on classification accuracy.
   That means transition lead time, net-of-cost Sharpe, and drawdown across a strategy zoo
   (equity/bond switch, probability-weighted allocation, mean-variance, risk parity, vol
   targeting).
3. A **calibration study** of FinBERT confidence in the regime-detection context (does
   temperature-scaled confidence improve timing?).
4. A portable, ablatable **news importance scoring framework** (outlet prior × topic type × novelty).

## Pipeline

1. Daily news corpus (headline + body, outlet, timestamp) → **FinBERT** sentiment
   (positive / neutral / negative).
2. Score each article: **weighted sentiment = Σ P(class) · confidence · importance**, with
   confidence = 1 − class entropy and importance = w_outlet × w_topic × w_novelty.
3. Aggregate to daily features (weighted sentiment, EMA-5, news volume, article dispersion) and
   merge with market features (returns, realized vol, VIX, credit spread).
4. Standardize → **PCA** (~95% variance) → **Gaussian HMM** (2–4 states) → regime posteriors.
5. Trade the regimes: equity/bond switch, probability-weighted allocation, portfolio models.

## Experimental setup

**Baselines (control group):** buy & hold SPY · 60/40 · price-only HMM (returns + vol) ·
raw unweighted FinBERT-sentiment HMM (the critical ablation) · PCA-HMM without news ·
Markov-switching GARCH · always-bull.

**Ablations:** drop confidence weighting · drop importance weighting · drop both ·
placebo (shift news +5 days forward — signal must vanish) · state-count sweep (2/3/4) ·
PCA component sweep.

**Metrics:** net Sharpe (after transaction costs), max drawdown, Calmar, turnover,
regime-transition F1 vs ground truth (NBER recessions, VIX>25 episodes), and the headline
metric — **transition lead time in days** vs the price-only HMM.

**Statistics:** Diebold–Mariano on Sharpe differences, McNemar on regime classification
agreement, walk-forward stability of transition matrices.

## Data

| Source | Role | Cost |
|---|---|---|
| GDELT DOC 2.0 API | Primary news corpus (2017+, outlet domains, timestamps) | Free |
| Alpha Vantage NEWS_SENTIMENT | Vendor sentiment + topics + source, to validate the FinBERT layer | Free tier / ~$50 mo |
| Stooq / Sharadar | SPY, IEF/AGG, VIX daily prices | Free |

## Expected results & comparison

We compare directly against **price-only Hidden Markov models** — both a standard Gaussian HMM and
the PCA-HMM variant (identical setup, no news inputs). Our prediction: the importance- and
confidence-weighted news pipeline detects regime transitions earlier, and we therefore expect it to
deliver a **higher net Sharpe ratio** than the price-only baselines.

In addition, we plan to test **portfolio allocation algorithms conditioned on each detected regime**
(e.g., regime-switched equity/bond allocation, probability-weighted sizing). This part is
potentially optional and awaits further literature review.

## Limitations

Lookahead bias (article timestamps vs market close) is the main validity threat · FinBERT
confidence is miscalibrated out-of-domain · HMM state count invites overfitting · outlet
credibility priors drift · multi-day timing edges may not survive transaction costs ·
index-level results may not generalize to single names · news data coverage starts 2017
(regime sample includes ~2 recessions).

## Team & task split (4 people)

- **Literature lead (1 person):** owns the paper's introduction; reads and refines papers, cites
  our findings, and drives further research into large language models — especially confidence
  calibration.
- **Baseline engineer (1 person):** builds the price-only HMM baseline (Gaussian HMM / PCA-HMM)
  and produces the baseline results.
- **Sentiment-model engineer (1 person):** builds the sentiment model that feeds the HMM
  (FinBERT layer, confidence = 1 − class entropy).
- **Data engineer (1 person):** collects the sentiment data (together with the sentiment-model
  engineer) and the confidence levels.

## Relevant past papers

### HMM / regime detection
- **Hamilton (1989)**, *Econometrica 57(2)* — introduced Markov regime-switching for business
  cycles. *Gap:* two-state parametric model, no news, no ML estimation.
- **Nystrup et al. (2019)**, *JRFM 13(12):311* — regime-switching factor investing with HMMs,
  with live retraining. *Gap:* price/vol observations only; news absent.

### News sentiment & markets
- **Tetlock (2007)**, *Journal of Finance 62(3)* — media content predicts short-horizon market
  moves. *Gap:* bag-of-words on a single column; no regime model.
- **Garcia & Norli (2021)**, *JFE 141(2)* — most news tone is noise for prediction. *Gap:*
  motivates our importance weighting but does not build it.
- **Araci (2019)**, arXiv:1908.10063 — FinBERT: financial-domain sentiment transformer. *Gap:*
  classifier only; confidence miscalibration unexamined downstream.
- **JRFM 18(8):412 (2025)** — FinBERT sentiment alone stalls at ~55–60% directional accuracy on
  index direction. *Gap:* the accuracy ceiling our weighting scheme attacks.
- **J. Behavioral Finance (2016)** — media sentiment in a regime-switching return/vol model.
  *Gap:* sentiment as exogenous regressor, not HMM emissions; no article-level weighting.

### Closest work
- **PLOS ONE (2025), pone.0331658** — PCA-HMM on tech stocks including an investor-sentiment
  series; finds sentiment dominates momentum/volume. *Gap:* generic aggregate sentiment index —
  no article-level FinBERT, no confidence or importance weighting, no trading/portfolio
  evaluation. **This is the paper to cite as nearest-neighbor and differentiate against.**
- **TechRxiv: source credibility in LSTM stock prediction** — weights sentiment by source
  credibility. *Gap:* LSTM direct prediction; no regime model, no economic evaluation.
- **FusionLSTM-CNF (2025)** — confidence-calibrated weighting of FinBERT in a multi-modal LSTM.
  *Gap:* fusion for direct prediction; no HMM, no regime timing.
- **Mathematics 13(17):2747 (2025)** — FinBERT sentiment features with SHAP for stock
  prediction. *Gap:* supervised direct prediction, no regime detection.

## License

TBD
