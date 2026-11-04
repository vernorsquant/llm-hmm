"""Regression test for the audit CRITICAL fix: walk-forward canonicalization
must read TRUE standardized VIX, never a PCA-rotated component.

filtered_month_posterior previously indexed the POST-PCA matrix with
cols.index("VIX_close") — for PCA feature sets that column is a rotated
component (empirically corr(PC3, VIX) ~ -0.56), so the bear state could be
picked from the wrong signal and bull/bear flip in some monthly refits.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment.regime_models import filtered_month_posterior

COLS = ["SPY_ret", "realized_vol", "VIX_close", "sent_w", "sent_w_ema5", "sent_disp"]


def _synthetic_features(n_train=700, n_month=80, seed=3):
    """Two persistent regimes driven by VIX (the canonicalization anchor):
    high-VIX regime co-moves with negative returns and high realized vol;
    the three sentiment columns are independent noise."""
    rng = np.random.default_rng(seed)

    def _block(n):
        states = np.zeros(n, dtype=int)
        for t in range(1, n):
            states[t] = states[t - 1] if rng.random() < 0.95 else 1 - states[t - 1]
        vix = np.where(states == 1, rng.normal(28, 2.5, n), rng.normal(15, 1.8, n))
        ret = np.where(states == 1, rng.normal(-0.0012, 0.009, n), rng.normal(0.0006, 0.006, n))
        rv = np.where(states == 1, rng.normal(0.028, 0.004, n), rng.normal(0.012, 0.003, n))
        sent = rng.normal(0, 0.25, n)
        return pd.DataFrame(
            {"SPY_ret": ret, "realized_vol": rv, "VIX_close": vix,
             "sent_w": sent, "sent_w_ema5": pd.Series(sent).ewm(span=5, adjust=False).mean().values,
             "sent_disp": rng.normal(0.2, 0.05, n)},
            index=pd.bdate_range("2021-01-04", periods=n),
        )

    return _block(n_train), _block(n_month)


def test_pca_walkforward_canonicalizes_on_true_vix():
    pytest.importorskip("hmmlearn")
    train, month = _synthetic_features()
    post = filtered_month_posterior(train, month, COLS, use_pca=True, random_state=42)

    assert post.notna().all(), "PCA walk-forward must return posteriors for all month rows"

    # true standardized VIX (train scaler, mirroring the implementation)
    vix_z = (month["VIX_close"] - train["VIX_close"].mean()) / train["VIX_close"].std()
    bear = post < 0.5
    bull = post >= 0.5
    assert bear.any() and bull.any(), "both regimes must appear in the month"
    assert vix_z[bear].mean() > vix_z[bull].mean(), (
        "bear state (posterior<0.5) must have HIGHER true standardized VIX — "
        "canonicalization must use real VIX, not a PCA-rotated component"
    )

    # the PCA path and the plain-HMM path must recover the SAME latent regime
    post_plain = filtered_month_posterior(train, month, COLS, use_pca=False, random_state=42)
    corr = float(post.corr(post_plain))
    assert corr > 0.8, (
        f"PCA and plain-HMM posteriors disagree (corr={corr:.2f}) — "
        "canonicalization orientation is inconsistent between the two paths"
    )
