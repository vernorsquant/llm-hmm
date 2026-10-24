"""FinBERT sentiment scoring (ProsusAI/finbert) with disk cache.

Probability column convention: [P(positive), P(neutral), P(negative)] — mapped
explicitly via the model's id2label so any checkpoint reordering is safe.
Text inference runs strictly once per article (sha1-keyed parquet cache).
"""

import hashlib

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .scoring import confidence_from_probs, sentiment_raw

MODEL_ID = "ProsusAI/finbert"
_CACHE_FILE = f"{CACHE_DIR}/finbert_scores.parquet"


def _text_key(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, (open("/proc/cpuinfo").read().count("processor") // 2)))
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return torch, tok, model


def score_articles(articles, text_col="headline", batch_size=64):
    """Return copy of `articles` with P_pos, P_neu, P_neg, S_raw, confidence.

    Cached by sha1(text): each unique text is passed through the model once
    across all runs.
    """
    texts = articles[text_col].astype(str).tolist()
    keys = [_text_key(t) for t in texts]

    cache = {}
    try:
        cached = pd.read_parquet(_CACHE_FILE)
        cache = dict(zip(cached["key"], cached[["P_pos", "P_neu", "P_neg"]].values))
    except Exception:
        pass

    missing = sorted({k for k in keys if k not in cache})
    if missing:
        torch, tok, model = _load_model()
        key_to_text = {}
        for k, t in zip(keys, texts):
            if k in set(missing):
                key_to_text[k] = t
        uniq_texts = [key_to_text[k] for k in missing]

        probs = []
        with torch.no_grad():
            for i in range(0, len(uniq_texts), batch_size):
                batch = uniq_texts[i : i + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                logits = model(**enc).logits
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        probs = np.vstack(probs)

        id2label = model.config.id2label
        order = [
            int(np.where([id2label[j] == lab for j in range(len(id2label))])[0][0])
            for lab in ("positive", "neutral", "negative")
        ]
        p_ordered = probs[:, order]
        for k, p in zip(missing, p_ordered):
            cache[k] = p

        pd.DataFrame(
            [{"key": k, "P_pos": cache[k][0], "P_neu": cache[k][1], "P_neg": cache[k][2]} for k in cache]
        ).to_parquet(_CACHE_FILE)

    P = np.array([cache[k] for k in keys])
    out = articles.copy().reset_index(drop=True)
    out["P_pos"], out["P_neu"], out["P_neg"] = P[:, 0], P[:, 1], P[:, 2]
    out["S_raw"] = sentiment_raw(P)
    out["confidence"] = confidence_from_probs(P)
    return out
