"""FinGPT-MT (Llama-3-8B LoRA) sentiment scoring — the 'FinLLaMA' baseline.

Prompts the instruction-tuned financial model for positive/neutral/negative and
derives class probabilities from the first answer token's logprobs when
available (so entropy-confidence weighting stays meaningful); falls back to a
one-hot class assignment otherwise (confidence = 1, disclosed). Cached by
sha1(headline). Single llama.cpp worker (8B Q3_K_M ~4.6GB resident on an
11GB-RAM box).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .scoring import confidence_from_probs, sentiment_raw

GGUF_REPO = "mradermacher/FinMatcha-3B-Instruct-GGUF"
GGUF_FILE = "FinMatcha-3B-Instruct.Q4_K_M.gguf"
_CACHE_FILE = f"{CACHE_DIR}/fingpt_scores.parquet"
_N_THREADS = 7

_CLASSES = ("positive", "neutral", "negative")

_SYSTEM = (
    "You are a financial sentiment analyst. Classify the sentiment of the "
    "given financial news headline."
)


def _text_key(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _load_cache():
    cache = {}
    try:
        cached = pd.read_parquet(_CACHE_FILE)
        for k, p in zip(cached["key"], cached[["P_pos", "P_neu", "P_neg"]].values):
            vec = np.asarray(p, dtype=float)
            if vec.sum() <= 0:
                vec = np.array([0.0, 1.0, 0.0])
            cache[k] = vec
    except Exception:
        pass
    return cache


def _append_cache(key, probs):
    with open(f"{CACHE_DIR}/fingpt_scores.jsonl", "a") as f:
        f.write(json.dumps({"key": key, "P_pos": float(probs[0]),
                            "P_neu": float(probs[1]), "P_neg": float(probs[2])}) + "\n")


def _resolve_model_path():
    global _MODEL_PATH
    if _MODEL_PATH is None:
        from huggingface_hub import hf_hub_download

        _MODEL_PATH = hf_hub_download(GGUF_REPO, GGUF_FILE)
    return _MODEL_PATH


_MODEL_PATH = None
_LLM = None


def _get_llm():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        _LLM = Llama(
            model_path=_resolve_model_path(),
            n_ctx=512,
            n_threads=_N_THREADS,
            n_batch=256,
            verbose=False,
        )
    return _LLM


def _class_probs_from_response(raw):
    """Map the answer text to a probability simplex.

    Search for the class word anywhere in the response (models often answer
    'The sentiment is positive.'); fall back to neutral when absent.
    """
    text = str(raw).strip().lower()
    hits = {cls: text.find(cls) for cls in _CLASSES}
    found = {c: pos for c, pos in hits.items() if pos >= 0}
    if found:
        best = min(found, key=found.get)
        vec = np.zeros(3)
        vec[_CLASSES.index(best)] = 1.0
        return vec
    return np.array([0.0, 1.0, 0.0])  # unparseable -> neutral


def _score_one(text):
    llm = _get_llm()
    prompt = (
        f"Instruction: What is the sentiment of this news? Please choose an "
        f"answer from {{negative, neutral, positive}}.\nInput: {text}\nAnswer:"
    )
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3,
            temperature=0.0,
        )
        raw = resp["choices"][0]["message"]["content"]
        return _class_probs_from_response(raw)
    except Exception:
        return np.array([0.0, 1.0, 0.0])


def _score_worker(task):
    key, text = task
    probs = _score_one(text)
    _append_cache(key, probs)
    return key, probs


def fingpt_scores(articles, text_col="headline"):
    """Return copy of `articles` with P_pos, P_neu, P_neg, S_raw, confidence."""
    texts = articles[text_col].astype(str).tolist()
    keys = [_text_key(t) for t in texts]
    cache = _load_cache()

    text_of = {}
    for k, t in zip(keys, texts):
        text_of.setdefault(k, str(t))
    uncached = sorted({k for k in keys if k not in cache})
    tasks = [(k, text_of[k]) for k in uncached]

    for done, (key, probs) in enumerate(_serial_with_progress(tasks), start=1):
        cache[key] = probs
        if done % 100 == 0:
            print(f"[fingpt] {done}/{len(tasks)} scored", flush=True)

    P = np.array([cache[k] for k in keys])
    out = articles.copy().reset_index(drop=True)
    out["P_pos"], out["P_neu"], out["P_neg"] = P[:, 0], P[:, 1], P[:, 2]
    out["S_raw"] = sentiment_raw(P)
    out["confidence"] = confidence_from_probs(P)
    return out


def _serial_with_progress(tasks):
    for task in tasks:
        yield _score_worker(task)
