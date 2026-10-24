"""Article-level scoring: FinBERT sentiment, entropy-based confidence, LLM importance.

Column convention for probability arrays: [P(positive), P(neutral), P(negative)]
matching ProsusAI/finbert's id2label mapping {0: positive, 1: neutral, 2: negative}.
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
    """Normalized confidence C = 1 - H(p)/ln(3), clipped to [0, 1]."""
    c = 1.0 - entropy(p) / _LOG3
    return np.clip(c, 0.0, 1.0)


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
