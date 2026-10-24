"""VADER sentiment baseline scoring (Hutto & Gilbert 2014, ICWSM).

Same weighting architecture as the FinBERT layer: VADER's {neg, neu, pos}
probabilities feed the identical entropy-based confidence and P(pos) - P(neg)
sentiment, so M5/M6 differ from M3/M4 ONLY in the sentiment model — a clean
domain-transformer vs general-lexicon ablation. Cached by sha1(text).
"""

import hashlib

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .scoring import confidence_from_probs, sentiment_raw

_CACHE_FILE = f"{CACHE_DIR}/vader_scores.parquet"


def _text_key(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _get_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_scores(articles, text_col="headline"):
    """Return copy of `articles` with P_pos, P_neu, P_neg, S_raw, confidence
    from VADER. Same column contract as finbert_scoring.score_articles."""
    texts = articles[text_col].astype(str).tolist()
    keys = [_text_key(t) for t in texts]

    cache = {}
    try:
        cached = pd.read_parquet(_CACHE_FILE)
        for k, p in zip(cached["key"], cached[["P_pos", "P_neu", "P_neg"]].values):
            vec = np.asarray(p, dtype=float)
            if vec.sum() <= 0:  # sanitize legacy degenerate entries
                vec = np.array([0.0, 1.0, 0.0])
            cache[k] = vec
    except Exception:
        pass

    missing = sorted({k for k in keys if k not in cache})
    if missing:
        sia = _get_analyzer()
        key_to_text = {}
        for k, t in zip(keys, texts):
            if k in set(missing):
                key_to_text[k] = t
        for k in missing:
            p = sia.polarity_scores(key_to_text[k])
            vec = np.array([p["pos"], p["neu"], p["neg"]], dtype=float)
            if vec.sum() <= 0:  # empty/degenerate text -> neutral
                vec = np.array([0.0, 1.0, 0.0])
            # VADER's neg/neu/pos already sum to 1 — a valid probability simplex
            cache[k] = vec

        pd.DataFrame(
            [{"key": k, "P_pos": cache[k][0], "P_neu": cache[k][1], "P_neg": cache[k][2]} for k in cache]
        ).to_parquet(_CACHE_FILE)

    P = np.array([cache[k] for k in keys])
    out = articles.copy().reset_index(drop=True)
    out["P_pos"], out["P_neu"], out["P_neg"] = P[:, 0], P[:, 1], P[:, 2]
    out["S_raw"] = sentiment_raw(P)
    out["confidence"] = confidence_from_probs(P)
    return out
