"""The 4 regime models: Gaussian HMM and PCA-Gaussian HMM, with and without news.

State canonicalization: hmmlearn's state labels are arbitrary. We relabel so
State 0 = Bull/Low-Vol (lower mean VIX) and State 1 = Bear/High-Vol (higher
mean VIX), matching the spec and the regime-detection literature convention.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MARKET_FEATURES = ["SPY_ret", "realized_vol", "VIX_close"]
# Level-based sentiment signals only. The raw high-importance article COUNT is
# deliberately excluded from HMM inputs: it is zero-inflated by corpus coverage
# gaps, which made the HMM split states on news-coverage presence instead of
# market regime (documented in README). It is still computed and reported.
SENTIMENT_FEATURES = ["sent_w", "sent_w_ema5", "sent_disp"]
# VADER baseline: identical architecture, lexicon sentiment instead of FinBERT
# (`_v` suffix columns built by the pipeline from vader_scoring output).
SENTIMENT_FEATURES_VADER = ["sent_w_v", "sent_w_ema5_v", "sent_disp_v"]
# FinGPT-MT (Llama-3-8B) baseline: same architecture, instruction-tuned LLM sentiment.
SENTIMENT_FEATURES_FINGPT = ["sent_w_f", "sent_w_ema5_f", "sent_disp_f"]

FEATURE_SETS = {
    "M1_price_hmm": (MARKET_FEATURES, False),
    "M2_price_pca_hmm": (MARKET_FEATURES, True),
    "M3_sent_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES, False),
    "M4_sent_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES, True),
    "M5_vader_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_VADER, False),
    "M6_vader_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_VADER, True),
    "M7_fingpt_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_FINGPT, False),
    "M8_fingpt_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_FINGPT, True),
}

def _fit_one(feat, cols, use_pca, random_state):
    X = feat[cols].astype(float)
    valid = X.notna().all(axis=1).values
    Xv = X.values[valid]

    scaler = StandardScaler().fit(Xv)
    Xs = scaler.transform(Xv)

    pca = None
    explained = 1.0
    if use_pca:
        pca = PCA(n_components=0.95, svd_solver="full").fit(Xs)
        Xs = pca.transform(Xs)
        explained = float(pca.explained_variance_ratio_.sum())

    hmm = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=1000,
        tol=1e-4,
        random_state=random_state,
        min_covar=1e-3,
    )
    try:
        hmm.fit(Xs)
    except ValueError:
        # degenerate feature blocks (e.g., one-hot sentiment paths) can produce
        # singular covariances during EM — escalate regularization and retry
        hmm = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=1000,
            tol=1e-4,
            random_state=random_state + 1,
            min_covar=1e-2,
        )
        hmm.fit(Xs)

    # Canonicalize: state with higher mean VIX (original units) = Bear = 1
    vix_z = Xs if not use_pca else None
    states_raw = hmm.predict(Xs)
    # Mean VIX per state computed from the standardized VIX column (cols index 2)
    vix_col = scaler.transform(feat[cols].astype(float).values[valid])[:, cols.index("VIX_close")]
    mean_vix = {k: vix_col[states_raw == k].mean() for k in (0, 1)}
    bear_raw = max(mean_vix, key=mean_vix.get)
    bull_raw = 1 - bear_raw

    proba = hmm.predict_proba(Xs)
    posterior_bull = pd.Series(proba[:, bull_raw], index=feat.index[valid])
    posterior_bull = posterior_bull.reindex(feat.index)
    states = pd.Series(
        np.where(posterior_bull >= 0.5, 0, 1), index=feat.index
    )
    # NaN posteriors (warmup rows) -> unknown state
    states[posterior_bull.isna()] = -1

    return {
        "model": hmm,
        "scaler": scaler,
        "pca": pca,
        "pca_explained_variance": explained,
        "posterior_bull": posterior_bull,
        "states": states.values,
        "feature_cols": cols,
        "valid_mask": valid,
    }


def fit_regime_models(feat, feature_sets=None, random_state=42):
    """Fit all 4 models on `feat` (DataFrame with market + sentiment columns).

    `feature_sets` maps name -> (cols, use_pca) or -> cols (use_pca inferred
    from "pca" in the name). Returns {name: result_dict}.
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS
    results = {}
    for name, spec in feature_sets.items():
        if isinstance(spec, tuple):
            cols, use_pca = spec
        else:
            cols, use_pca = list(spec), ("pca" in name.lower())
        results[name] = _fit_one(feat, cols, use_pca, random_state)
    return results


def causal_filtered_probabilities(hmm, Xs):
    """CAUSAL forward (Hamilton) filter: P(z_t | X_{1:t}) per row.

    hmmlearn's predict_proba runs forward-BACKWARD smoothing, so its day-t
    posterior depends on FUTURE observations within the frame — a lookahead
    channel in walk-forward paths. This uses only past + current observations.
    """
    from scipy.special import logsumexp

    ll = hmm._compute_log_likelihood(np.asarray(Xs, dtype=float))
    n, K = ll.shape
    logstart = np.log(np.asarray(hmm.startprob_) + 1e-16)
    logtrans = np.log(np.asarray(hmm.transmat_) + 1e-16)
    logalpha = np.empty((n, K))
    logalpha[0] = logstart + ll[0]
    logalpha[0] -= logsumexp(logalpha[0])
    for t in range(1, n):
        # alpha_t(j) ∝ ll_t(j) * sum_i alpha_{t-1}(i) * A[i, j]
        logalpha[t] = ll[t] + logsumexp(logalpha[t - 1][:, None] + logtrans, axis=0)
    return np.exp(logalpha - logsumexp(logalpha, axis=1, keepdims=True))


def filtered_month_posterior(train_feat, month_feat, cols, use_pca, random_state=42, warmup_rows=60):
    """Fit on train only, then run a CAUSAL filtered forward pass over
    [train-tail warmup + month], carrying the filtered state distribution from
    the train window into the month. Returns posterior_bull for the month days.

    Replaces predict_proba-based smoothing (which leaked day t+1..month-end
    information into day-t posteriors — see README audit section).
    """
    res = _fit_one(train_feat, cols, use_pca, random_state)
    hmm, scaler, pca = res["model"], res["scaler"], res["pca"]

    def to_model_space(frame):
        X = frame[cols].astype(float)
        valid = X.notna().all(axis=1).values
        Xs = scaler.transform(X.values[valid])
        if pca is not None:
            Xs = pca.transform(Xs)
        return Xs, valid

    Xtr, _ = to_model_space(train_feat)
    Xmo, valid_mo = to_model_space(month_feat)
    if len(Xtr) == 0 or len(Xmo) == 0:
        return pd.Series(dtype=float).reindex(month_feat.index)

    tail = Xtr[-warmup_rows:]
    Xall = np.vstack([tail, Xmo])

    # canonicalization from TRAIN only (mirror of _fit_one)
    vix_col_train = Xtr[:, cols.index("VIX_close")]
    states_raw = hmm.predict(Xtr)
    mean_vix = {k: vix_col_train[states_raw == k].mean() for k in (0, 1)}
    bear_raw = max(mean_vix, key=mean_vix.get)
    bull_raw = 1 - bear_raw

    alpha = causal_filtered_probabilities(hmm, Xall)
    post = pd.Series(alpha[len(tail):, bull_raw], index=month_feat.index[valid_mo])
    return post.reindex(month_feat.index)
