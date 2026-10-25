"""Article-level scoring: FinBERT sentiment, entropy-based confidence, LLM importance.

Column convention for probability arrays: the pipeline resolves FinBERT's
columns BY NAME via model.config.id2label (actual mapping:
{0: 'positive', 1: 'negative', 2: 'neutral'} — note neg/neu are swapped vs
the naive order). All public helpers here expect the resolved order
[P(positive), P(neutral), P(negative)].
"""

import re

import numpy as np

ENTROPY_EPS = 1e-9
_LOG3 = float(np.log(3.0))


def entropy(p):
    """Row-wise Shannon entropy H(p) = -sum p_i * ln(p_i + eps). p: (n,3)."""
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p + ENTROPY_EPS), axis=1)


def confidence_from_probs(p):
    """Normalized confidence C = 1 - H(p)/ln(3), clipped to [0, 1].

    Point-mass-at-neutral rows (P_neu = 1 exactly — VADER outputs with no
    lexicon words, or unparseable LLM fallbacks) carry NO sentiment signal,
    so their confidence is capped at 0.25 rather than rewarded with the
    maximal 1.0 a zero-entropy point mass would otherwise receive.
    """
    c = 1.0 - entropy(p) / _LOG3
    c = np.clip(c, 0.0, 1.0)
    neutral_point_mass = (np.asarray(p)[:, 1] >= 0.999) if np.asarray(p).ndim == 2 else None
    if neutral_point_mass is not None:
        c = np.where(neutral_point_mass, np.minimum(c, 0.25), c)
    return c


def sentiment_raw(p):
    """S_raw = P(positive) - P(negative) in [-1, 1]. p: (n,3) as [pos, neu, neg]."""
    p = np.asarray(p, dtype=float)
    return p[:, 0] - p[:, 2]


def clamp_importance(x):
    """Clamp LLM importance rating to [0, 1]."""
    return float(np.clip(float(x), 0.0, 1.0))


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_importance(text):
    """Parse the first float in an LLM response; None if no number found."""
    if text is None:
        return None
    m = _FLOAT_RE.search(str(text))
    if m is None:
        return None
    return float(m.group(0))


def article_weights(confidence, importance):
    """W_i = C_i * I_i (elementwise product of two [0,1] vectors)."""
    return np.asarray(confidence, dtype=float) * np.asarray(importance, dtype=float)
