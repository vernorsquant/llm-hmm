"""Statistical significance tests for strategy comparisons.

- Diebold-Mariano style test on daily return differentials with Newey-West HAC.
- Jobson-Korkie / Memmel (2003) test for the difference of two Sharpe ratios.
- Stationary block bootstrap for the Sharpe difference (distribution-free).
- McNemar test (exact binomial, two-sided) on discordant classification pairs.
- F1 of binary stress flags vs a proxy.
"""

import numpy as np
from scipy import stats


def newey_west_var(x, lags=5):
    """Newey-West HAC variance of the sample mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    xbar = x.mean()
    e = x - xbar
    v = e @ e / T  # gamma_0
    for l in range(1, min(lags, T - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        gamma = e[l:] @ e[:-l] / T
        v += 2.0 * w * gamma
    return v


def dm_test(r1, r2, lags=5):
    """Diebold-Mariano on the loss differential d_t = r1_t - r2_t (loss = -return).

    Tests H0: E[r1] == E[r2] with HAC variance. Two-sided normal p.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    d = r1 - r2
    d = d[~np.isnan(d)]
    T = len(d)
    dm = d.mean() / np.sqrt(newey_west_var(d, lags))
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return {"dm": float(dm), "p_value": float(p), "n": T}


def jk_memmel_sharpe_diff(r1, r2):
    """Jobson-Korkie (1981) / Memmel (2003) test for Sharpe(r1) - Sharpe(r2).

    Normal-theory asymptotic z statistic; rf = 0.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    assert len(r1) == len(r2)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    s12 = np.cov(r1, r2, ddof=1)[0, 1]
    sr1, sr2 = mu1 / s1, mu2 / s2

    t_var = (
        2.0
        + mu1**2 / (2.0 * s1**2)
        + mu2**2 / (2.0 * s2**2)
        - 2.0 * s12 / (s1 * s2)
        - (mu1 * mu2 * s12**2) / (s1**3 * s2**3)
    )
    var_d = max(t_var / T, 1e-16)
    z = (sr1 - sr2) / np.sqrt(var_d)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return {"sr_diff": float(sr1 - sr2), "z": float(z), "p_value": float(p), "n": T}


def _block_indices(T, block, rng):
    """Stationary-block-bootstrap style indices: overlapping blocks of ~`block`."""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([ (s + np.arange(block)) % T for s in starts ])[:T]
    return idx


def bootstrap_sharpe_diff(r1, r2, n_boot=10000, block=10, seed=0):
    """Two-sided bootstrap p-value and 95% CI for Sharpe(r1) - Sharpe(r2),
    paired stationary block bootstrap (preserves short-range dependence)."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    T = len(r1)

    def sharpe(a):
        return a.mean() / a.std(ddof=1) if a.std(ddof=1) > 0 else 0.0

    d_obs = sharpe(r1) - sharpe(r2)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(T, block, rng)
        diffs[i] = sharpe(r1[idx]) - sharpe(r2[idx])
    # two-sided p relative to the null of zero, centered by the bootstrap dist
    centered = diffs - diffs.mean()
    p = float(np.mean(np.abs(centered) >= abs(d_obs)))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "sr_diff": float(d_obs),
        "p_value": p,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": n_boot,
        "block": block,
    }


def mcnemar_test(b, c):
    """Exact two-sided binomial McNemar test on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    k = min(b, c)
    p = float(stats.binomtest(k, n, 0.5).pvalue)
    stat = (abs(b - c) - 1.0) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return {"statistic": float(stat), "p_value": p, "b": int(b), "c": int(c)}


def stress_f1(y_true, y_pred):
    """F1 of binary stress flags vs a proxy (positive class = 1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def holm_correction(p_values):
    """Holm-Bonferroni adjusted p-values (same family order preserved)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted
