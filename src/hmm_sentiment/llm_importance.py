"""LLM macro-importance rating via a local quantized model (llama.cpp CPU).

Each article headline is rated on [0, 1]:
  0.0 = noise / single-stock trivia / promotional
  1.0 = systemic shocks: FOMC decisions, credit/liquidity stress, major
        inflation prints, systemic banking stress.

Ratings are cached to JSONL keyed by sha1(headline); every rating is appended
immediately after inference, so runs are resumable and observable. Work is
dispatched as per-article tasks to a small process pool (spawn context — fork
plus llama.cpp is unreliable); if pool execution fails, a serial in-process
fallback finishes the remainder.
"""

import hashlib
import json
import os

import numpy as np

from .config import CACHE_DIR
from .scoring import clamp_importance, parse_importance

GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_CACHE_FILE = f"{CACHE_DIR}/llm_importance.jsonl"
N_PARALLEL = 4
N_THREADS = 4

_SYSTEM = (
    "You are a meticulous financial news analyst. You rate the systemic "
    "macroeconomic importance of news headlines on a scale from 0.0 to 1.0."
)
_RUBRIC = (
    "Rating rubric:\n"
    "1.0 = systemic shocks: FOMC rate decisions, credit or liquidity stress, "
    "major inflation prints (CPI), systemic banking failures, debt-ceiling default risk.\n"
    "0.5 = moderately macro-relevant: jobs reports, earnings of systemically "
    "important firms, commodity shocks, notable geopolitical risk.\n"
    "0.0 = noise: single-stock trivia, routine promotional content, celebrity "
    "or sports news, minor product launches.\n"
    "Reply with ONLY one number between 0.0 and 1.0."
)

_MODEL_PATH = None


def _resolve_model_path():
    global _MODEL_PATH
    if _MODEL_PATH is None:
        from huggingface_hub import hf_hub_download

        _MODEL_PATH = hf_hub_download(GGUF_REPO, GGUF_FILE)
    return _MODEL_PATH


def _key(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _load_cache():
    cache = {}
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = float(rec["importance"])
                except Exception:
                    continue
    return cache


def _append(key, importance):
    with open(_CACHE_FILE, "a") as f:
        f.write(json.dumps({"key": key, "importance": importance}) + "\n")


def _ask_llm(llm, text):
    prompt = f"{_RUBRIC}\n\nHeadline: {text}\nRating (0.0-1.0):"
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        return parse_importance(resp["choices"][0]["message"]["content"])
    except Exception:
        return None


_LLM = None


def _get_llm():
    """Per-process lazy Llama singleton: each pool worker loads the model once."""
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        _LLM = Llama(
            model_path=_resolve_model_path(),
            n_ctx=768,
            n_threads=N_THREADS,
            n_batch=256,
            verbose=False,
        )
    return _LLM


def _rate_one(text):
    """Rate a single headline (reuses this process's loaded model)."""
    val = _ask_llm(_get_llm(), text)
    return clamp_importance(val) if val is not None else float("nan")


def _worker(task):
    """Pool worker: rate one text, append to cache, return (key, importance|nan)."""
    key, text = task
    importance = _rate_one(text)
    if np.isfinite(importance):
        _append(key, importance)
    return key, importance


def rate_importance(headlines, progress_every=50):
    """Rate each headline; returns np.array of importance in [0, 1].

    Unparseable LLM outputs are backfilled with the median of parsed ratings
    (0.5 if none). Ratings persist immediately; the LLM sees each unique
    headline exactly once across all runs.
    """
    keys = [_key(h) for h in headlines]
    cache = _load_cache()

    text_of = {}
    for k, h in zip(keys, headlines):
        text_of.setdefault(k, str(h))
    uncached = sorted({k for k in keys if k not in cache})
    tasks = [(k, text_of[k]) for k in uncached]

    if tasks:
        done = 0
        try:
            from multiprocessing import get_context

            ctx = get_context("spawn")
            with ctx.Pool(processes=N_PARALLEL) as pool:
                for key, importance in pool.imap_unordered(_worker, tasks, chunksize=2):
                    if np.isfinite(importance):
                        cache[key] = importance
                    done += 1
                    if done % progress_every == 0:
                        print(f"[llm_importance] {done}/{len(tasks)} rated", flush=True)
        except Exception as e:
            print(f"[llm_importance] pool failed ({e}); serial fallback", flush=True)
            for key, text in tasks:
                if key in cache:
                    continue
                importance = _rate_one(text)
                if np.isfinite(importance):
                    cache[key] = importance
                    _append(key, importance)

    vals = np.array([cache.get(k, np.nan) for k in keys], dtype=float)
    fallback = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.5
    vals = np.where(np.isfinite(vals), vals, fallback)
    return vals
