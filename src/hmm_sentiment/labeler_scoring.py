"""LLM-Labeler rubric-decomposition impact scoring (adapted from
zenosim/LLM-Labeler `simple_bsm.py`). Backend-selectable: OpenRouter
(`meta-llama/llama-3.1-8b-instruct` for the published 8b labels; GLM
`glm-5.3-flash` also supported via LLM_LABELER_BACKEND=glm — the very first
partial labeling pass used GLM, those labels live in a separate key space
and are not part of the published comparison).

Faithful to the original pipeline per article:
  1. branch_impact_rubric: the LLM proposes up to 4 stock-impact criteria,
     each with its own rubric.
  2. solve_criterion: per criterion, the LLM assigns an integer impact score
     in [-5, +5] with a rationale.
  3. Aggregate: average criterion score -> Bullish (>= +1.5) /
     Bearish (<= -1.5) / Neutral otherwise.

Adaptations (disclosed):
  - Backend: GLM `glm-5.3-flash` (z.ai OpenAI-compatible endpoint), temperature 0.
  - Ticker is "SPY": this corpus is macro/index news, and SPY/IEF are the
    traded assets in the HMM zoo.
  - S_raw = avg_impact / 5 so the series lives on the same [-1, 1] scale as
    FinBERT's P_pos - P_neg.
  - confidence = 1 - std(criterion scores)/5, clipped to [0.25, 1]
    (dispersion across the article's own rubric branches). Inactive while
    config.WEIGHTED_SENTIMENT is False; reported for completeness.
  - Cache: data/cache/llm_labeler_scores.jsonl, one JSON record per LLM call,
    appended per call (crash-resumable), loaded last-wins. Keyed by
    sha1(headline) + stage + criterion.
"""

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR

GLM_BASE_URL = os.getenv(
    "GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4"
)
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5.3-flash")

# Backend selection: 'glm' (default) or 'openrouter'. When OPENROUTER_API_KEY
# is set AND LLM_LABELER_BACKEND=openrouter, calls go to OpenRouter instead
# (the original zenosim/LLM-Labeler target) — lets a small fast model finish
# the corpus in parallel while GLM stays free for other work.
BACKEND = os.getenv("LLM_LABELER_BACKEND", "glm")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.getenv(
    "LLM_LABELER_MODEL", "meta-llama/llama-3.1-8b-instruct"
)

MAX_WORKERS = int(os.getenv("LLM_LABELER_WORKERS") or "8")
MAX_ATTEMPTS = 6
BULL_THRESHOLD = 1.5
BEAR_THRESHOLD = -1.5
MAX_CRITERIA = 4

# Token budgets. Reasoning models burn thinking tokens before the answer, so
# variants running a reasoning model need headroom or the JSON gets truncated.
PLAN_MAX_TOKENS = int(os.getenv("LLM_LABELER_PLAN_TOKENS", "700"))
SOLVE_MAX_TOKENS = int(os.getenv("LLM_LABELER_SOLVE_TOKENS", "250"))

# Bump when prompts, scale mapping, ticker substitution, or aggregation rules
# change — it is folded into the cache key so stale labels can never be
# silently reused across pipeline versions (audit MINOR-7).
LABELER_VERSION = 2

_LABELER_CACHE = os.path.join(CACHE_DIR, "llm_labeler_scores.jsonl")

_local = threading.local()


def _api_key_for(backend):
    if backend == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY", "")
    return os.environ.get("GLM_API_KEY", "")


def _api_key():
    return _api_key_for(BACKEND)


def call_llm(prompt: str, max_tokens: int = 700, max_retries: int = MAX_ATTEMPTS,
             backend: str = None, model: str = None) -> str:
    """One chat completion (backend/model overridable per call), temperature 0,
    thinking disabled on GLM (rubric/scoring prompts need no chain-of-thought;
    think-mode tripled latency), with 429/5xx backoff. Reasoning models keep
    the default thinking behavior — max_tokens must budget for reasoning."""
    backend = backend or BACKEND
    model = model or _model_tag(backend)
    key = _api_key_for(backend)
    if not key:
        raise RuntimeError(
            f"No API key for backend '{backend}' — set "
            f"{'OPENROUTER_API_KEY' if backend == 'openrouter' else 'GLM_API_KEY'}"
        )
    if backend == "openrouter":
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        extra = {}
    else:
        url = f"{GLM_BASE_URL}/chat/completions"
        extra = {"thinking": {"type": "disabled"}}
    last_err = None
    for attempt in range(max_retries):
        try:
            res = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    **extra,
                },
                timeout=120,
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            last_err = f"HTTP {res.status_code}: {res.text[:150]}"
            if res.status_code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # network hiccup
            last_err = str(e)
        time.sleep(min(2 ** attempt + 0.5 * attempt, 30))
    raise RuntimeError(f"GLM call failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------- JSON utils

def extract_json_blob(text):
    """First balanced JSON array or object in `text`, tolerating code fences
    and prose wrappers. Returns Python object or None."""
    if text is None:
        return None
    cleaned = str(text).replace("```json", "```")
    pairs = {"[": "]", "{": "}"}
    closers = set(pairs.values())
    for i, ch in enumerate(cleaned):
        if ch not in pairs:
            continue
        stack = [pairs[ch]]
        in_str = False
        esc = False
        for j in range(i + 1, len(cleaned)):
            c = cleaned[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in pairs:
                stack.append(pairs[c])
            elif c in closers:
                if not stack or stack[-1] != c:
                    break  # mismatched closer — abandon this opener
                stack.pop()
                if not stack:
                    blob = cleaned[i : j + 1]
                    try:
                        return json.loads(blob)
                    except Exception:
                        break  # balanced but invalid JSON — try next opener
    return None


def parse_rubric_list(raw):
    """Parse the branch_impact_rubric response into [{name, rubric}, ...],
    capped at MAX_CRITERIA (the prompt says 'up to 4' — enforce it)."""
    obj = extract_json_blob(raw)
    if isinstance(obj, list) and obj:
        out = []
        for item in obj:
            if isinstance(item, dict) and item.get("name") and item.get("rubric"):
                out.append({"name": str(item["name"])[:80], "rubric": str(item["rubric"])[:400]})
        if out:
            return out[:MAX_CRITERIA]
    return None


def parse_score(raw):
    """Parse the solve_criterion response into (score:int, rationale:str)."""
    obj = extract_json_blob(raw)
    if isinstance(obj, dict) and "score" in obj:
        try:
            score = int(round(float(obj["score"])))
        except Exception:
            return None
        score = max(-5, min(5, score))
        return score, str(obj.get("rationale", ""))[:400]
    return None


# ------------------------------------------------------- prompts (faithful)

def branch_impact_rubric_prompt(ticker: str, article: str) -> str:
    return f"""
You are a financial analyst.
Ticker: {ticker}
News: {article}

Propose an evaluation plan of up to 4 distinct criteria/dimensions to assess the stock price impact.
For each criterion, provide:
1. "name": The short dimension title (e.g. "Revenue Risk", "Public Relations")
2. "rubric": A concrete description specifying what factors to examine for this article.

Return ONLY a valid JSON list of objects in this format:
[
  {{"name": "Dimension Name", "rubric": "Evaluate whether..."}}
]
"""


def solve_criterion_prompt(ticker: str, article: str, criterion: dict) -> str:
    return f"""
Ticker: {ticker}
News: {article}

[Evaluation Dimension]
Name: {criterion["name"]}
Rubric / Instructions: {criterion["rubric"]}
[End of Evaluation Dimension]

Evaluate the stock impact SOLELY according to the rubric above.
Assign an impact score on a scale from -5 to +5:
 -5 = Catastrophic negative impact for the future
  0 = Neutral / negligible / already priced in / currently has no effect
 +5 = Transformative positive impact for the future

Provide an integer score and a brief rationale.
Return ONLY valid JSON in this exact format:
{{"score": -3, "rationale": "Short explanation..."}}
"""


# fallback rubric used when the plan stage fails twice (deterministic, disclosed)
FALLBACK_RUBRIC = [
    {"name": "Earnings & Fundamentals", "rubric": "Evaluate whether the news materially changes expected revenues, earnings, or costs for the company."},
    {"name": "Macro & Policy", "rubric": "Evaluate whether the news changes monetary policy, regulation, or macro conditions relevant to the company."},
    {"name": "Sentiment & Risk", "rubric": "Evaluate whether the news shifts investor sentiment or risk perception around the company."},
]


# ------------------------------------------------------------- cache

_cache_lock = threading.Lock()


def _model_tag(backend=None):
    backend = backend or BACKEND
    return OPENROUTER_MODEL if backend == "openrouter" else GLM_MODEL


def _text_key(text, backend=None, model=None):
    """Composite cache key: content hash + backend + model + pipeline version.

    Distinct backends/models/versions never share label space, so a backend
    or model switch can never silently mix labels (audit MAJOR-2 / MINOR-7)."""
    backend = backend or BACKEND
    model = model or _model_tag(backend)
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return f"{digest}|{backend}|{model}|v{LABELER_VERSION}"


def load_labeler_cache(path=_LABELER_CACHE):
    """Last-wins load of every cached LLM call record."""
    cache = {"plan": {}, "solve": {}, "final": {}}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                stage = rec.get("stage")
                key = rec.get("key")
                if stage == "plan":
                    cache["plan"][key] = rec["criteria"]
                elif stage == "solve":
                    cache["solve"][(key, rec["criterion"])] = (rec["score"], rec.get("rationale", ""))
                elif stage == "final":
                    cache["final"][key] = rec
    return cache


def _append(rec, path=_LABELER_CACHE):
    with _cache_lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------- per-article flow

def label_article(ticker, article, key, cache, path: "str | None" = _LABELER_CACHE,
                  backend=None, model=None):
    """Run the full rubric-decomposition pipeline for one article (cached).

    `path` is the JSONL cache file records are appended to; None disables
    persistence (tests). backend/model override the env-selected backend for
    multi-variant runs (keys and provenance reflect the effective pair).
    Returns dict: avg_impact (float, -5..5), sentiment
    (Bullish/Neutral/Bearish), n_criteria, breakdown scores, plan_fallback,
    solve_fallbacks.
    """
    def _persist(rec):
        """Append with full provenance. Failures (fallback plans, unparseable
        solves, any degraded final) are NEVER persisted — the article stays
        uncached and is retried on the next run (audit MAJOR-4)."""
        if path is not None:
            rec = dict(rec, backend=backend or BACKEND,
                       model=model or _model_tag(backend or BACKEND),
                       version=LABELER_VERSION)
            _append(rec, path)

    plan = cache["plan"].get(key)
    plan_fallback = False
    if plan is None:
        criteria = None
        for _ in range(2):
            try:
                criteria = parse_rubric_list(
                    call_llm(branch_impact_rubric_prompt(ticker, article),
                             max_tokens=PLAN_MAX_TOKENS, backend=backend, model=model))
            except Exception:
                criteria = None
            if criteria:
                break
        if not criteria:
            criteria = [c for c in FALLBACK_RUBRIC]
            plan_fallback = True
        plan = criteria
        if not plan_fallback:
            _persist({"key": key, "stage": "plan", "criteria": plan})

    scores, rationales, solve_fallbacks = [], {}, 0
    for crit in plan:
        name = crit["name"]
        hit = cache["solve"].get((key, name))
        if hit is None:
            parsed = None
            for _ in range(2):
                try:
                    parsed = parse_score(
                        call_llm(solve_criterion_prompt(ticker, article, crit),
                                 max_tokens=SOLVE_MAX_TOKENS, backend=backend, model=model))
                except Exception:
                    parsed = None
                if parsed:
                    break
            if parsed is None:
                # in-memory neutral fallback ONLY; nothing persisted, so the
                # next run retries this article from scratch
                parsed = (0, "unparseable/failed solve -> neutral fallback (not cached)")
                solve_fallbacks += 1
            score, rationale = parsed
            if solve_fallbacks == 0:
                _persist({"key": key, "stage": "solve", "criterion": name,
                          "score": score, "rationale": rationale})
        else:
            score, rationale = hit
        scores.append(score)
        rationales[name] = rationale

    # classify on the UNROUNDED mean (audit MINOR-6); rounded value is for reporting
    avg_raw = float(np.mean(scores)) if scores else 0.0
    avg = round(avg_raw, 3)
    if avg_raw >= BULL_THRESHOLD:
        sentiment = "Bullish"
    elif avg_raw <= BEAR_THRESHOLD:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    result = {
        "key": key,
        "stage": "final",
        "avg_impact": avg,
        "sentiment": sentiment,
        "n_criteria": len(scores),
        "scores": scores,
        "plan_fallback": plan_fallback,
        "solve_fallbacks": solve_fallbacks,
    }
    if not plan_fallback and solve_fallbacks == 0:
        _persist(result)
    return result


def labeler_scores(articles, text_col="headline", ticker="SPY", max_workers=MAX_WORKERS,
                   cache=None, path: "str | None" = _LABELER_CACHE, backend=None, model=None):
    """Label every article through the rubric-decomposition pipeline.

    backend/model select the label space (composite cache keys) and the LLM
    used — enables multi-variant runs in one process. Returns a copy of
    `articles` with columns: avg_impact, sentiment_lbl, S_raw (= avg_impact/5),
    confidence (= 1 - std(scores)/5 clipped to [0.25, 1]), n_criteria.
    """
    texts = articles[text_col].astype(str).tolist()
    keys = [_text_key(t, backend, model) for t in texts]
    if cache is None:
        cache = load_labeler_cache()

    todo = {}
    text_of = {}
    for k, t in zip(keys, texts):
        text_of.setdefault(k, str(t))
        if k not in cache["final"]:
            todo[k] = text_of[k]
    print(f"[labeler] {len(cache['final'])} cached, {len(todo)} to label", flush=True)

    fails = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(label_article, ticker, todo[k], k, cache, path, backend, model): k for k in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            k = futs[fut]
            try:
                res = fut.result()
                cache["final"][k] = res
            except Exception as e:
                fails += 1
                print(f"[labeler] FAIL {k}: {e}", flush=True)
            if i % 100 == 0:
                print(f"[labeler] {i}/{len(todo)} labeled", flush=True)
    if fails:
        print(f"[labeler] {fails} articles failed (left uncached, rerun resumes)", flush=True)

    finals = [cache["final"].get(k) for k in keys]
    out = articles.copy().reset_index(drop=True)
    # Degraded in-memory results (fallback rubric/solves, missing cache entry)
    # MUST read as NaN so downstream coverage gates bite — a fallback score-0
    # would otherwise masquerade as a real label (audit MAJOR-3/MAJOR-4).
    degraded = [
        (f is None) or f.get("plan_fallback") or bool(f.get("solve_fallbacks"))
        for f in finals
    ]
    out["avg_impact"] = [f["avg_impact"] if (f and not d) else np.nan
                         for f, d in zip(finals, degraded)]
    out["sentiment_lbl"] = [f["sentiment"] if (f and not d) else None
                            for f, d in zip(finals, degraded)]
    out["n_criteria"] = [f["n_criteria"] if (f and not d) else 0
                         for f, d in zip(finals, degraded)]
    out["S_raw"] = out["avg_impact"] / 5.0
    def _conf(scores):
        if not scores:
            return np.nan
        return float(np.clip(1.0 - np.std(scores) / 5.0, 0.25, 1.0))
    out["confidence"] = [_conf(f["scores"]) if f else np.nan for f in finals]
    return out
