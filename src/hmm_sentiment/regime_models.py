"""The 4 regime models: Gaussian HMM and PCA-Gaussian HMM, with and without news.

State canonicalization: hmmlearn's state labels are arbitrary. We relabel so
State 0 = Bull/Low-Vol (lower mean VIX) and State 1 = Bear/High-Vol (higher
mean VIX), matching the spec and the regime-detection literature convention.
"""

import numpy as np
import pandas as pd
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

FEATURE_SETS = {
    "M1_price_hmm": (MARKET_FEATURES, False),
    "M2_price_pca_hmm": (MARKET_FEATURES, True),
    "M3_sent_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES, False),
    "M4_sent_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES, True),
    "M5_vader_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_VADER, False),
    "M6_vader_pca_hmm": (MARKET_FEATURES + SENTIMENT_FEATURES_VADER, True),
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
