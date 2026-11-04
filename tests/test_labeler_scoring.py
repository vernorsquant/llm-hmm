"""Tests for the LLM-Labeler scoring adapter (pure functions + cache)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from hmm_sentiment.labeler_scoring import (
    extract_json_blob,
    label_article,
    load_labeler_cache,
    parse_rubric_list,
    parse_score,
    labeler_scores,
)


# ------------------------------------------------------- JSON extraction

def test_extract_json_list_plain():
    obj = extract_json_blob('[{"name": "A", "rubric": "r1"}]')
    assert obj == [{"name": "A", "rubric": "r1"}]


def test_extract_json_list_in_fences_and_prose():
    raw = 'Here you go:\n```json\n[{"name": "A", "rubric": "has \\"quotes\\" inside"}]\n```\nDone.'
    obj = extract_json_blob(raw)
    assert isinstance(obj, list) and obj[0]["name"] == "A"
    assert obj[0]["rubric"] == 'has "quotes" inside'


def test_extract_json_object_with_nested_strings():
    raw = '{"score": -3, "rationale": "bad {stuff} and \\"escapes\\""}'
    obj = extract_json_blob(raw)
    assert obj["score"] == -3


def test_extract_json_returns_none_on_garbage():
    assert extract_json_blob("no json here at all") is None
    assert extract_json_blob(None) is None


# ------------------------------------------------------------- parsers

def test_parse_rubric_list_filters_bad_items():
    raw = '[{"name": "A", "rubric": "r"}, {"nope": 1}, "junk", {"name": "B", "rubric": "r2"}]'
    out = parse_rubric_list(raw)
    assert [o["name"] for o in out] == ["A", "B"]


def test_parse_rubric_list_none_on_empty():
    assert parse_rubric_list("[]") is None
    assert parse_rubric_list("garbage") is None


def test_parse_score_clamps_and_rounds():
    assert parse_score('{"score": -3, "rationale": "x"}') == (-3, "x")
    assert parse_score('{"score": -9.4, "rationale": "x"}')[0] == -5
    assert parse_score('{"score": 7, "rationale": "x"}')[0] == 5
    assert parse_score('{"score": 2.6, "rationale": "x"}')[0] == 3
    assert parse_score("not json") is None


# -------------------------------------------------- aggregation + scaling

def test_label_article_aggregation_thresholds():
    pytest.importorskip("requests")
    cache = {"plan": {}, "solve": {}, "final": {}}
    key = "k1"
    cache["plan"][key] = [{"name": "A", "rubric": "r"}, {"name": "B", "rubric": "r"}]
    cache["solve"][(key, "A")] = (4, "up")
    cache["solve"][(key, "B")] = (3, "up")
    res = label_article("SPY", "headline", key, cache, path=None)
    assert res["avg_impact"] == 3.5
    assert res["sentiment"] == "Bullish"
    assert res["plan_fallback"] is False
    assert res["solve_fallbacks"] == 0

    # neutral band: avg in (-1.5, 1.5)
    cache2 = {"plan": {key: [{"name": "A", "rubric": "r"}]},
              "solve": {(key, "A"): (1, "meh")}, "final": {}}
    res2 = label_article("SPY", "headline", key, cache2, path=None)
    assert res2["sentiment"] == "Neutral"

    # bearish
    cache3 = {"plan": {key: [{"name": "A", "rubric": "r"}]},
              "solve": {(key, "A"): (-4, "down")}, "final": {}}
    assert label_article("SPY", "headline", key, cache3, path=None)["sentiment"] == "Bearish"


def test_label_article_fallback_rubric_when_plan_missing_and_solve_cached(monkeypatch):
    """Plan stage fails twice -> deterministic FALLBACK_RUBRIC; cached solves
    mean no solve-stage LLM calls. call_llm is stubbed so the test never
    touches the network."""
    from hmm_sentiment import labeler_scoring as ls

    def _boom(prompt, max_retries=1):
        raise RuntimeError("network disabled in test")

    monkeypatch.setattr(ls, "call_llm", _boom)
    cache = {"plan": {}, "solve": {}, "final": {}}
    # pre-seed solve hits for all three fallback criteria so no LLM call happens
    key = "kf"
    for c in ls.FALLBACK_RUBRIC:
        cache["solve"][(key, c["name"])] = (0, "pre")
    res = ls.label_article("SPY", "headline", key, cache, path=None)
    assert res["plan_fallback"] is True
    assert res["avg_impact"] == 0.0
    assert res["sentiment"] == "Neutral"


# ------------------------------------------------------------ S_raw scale

def test_labeler_scores_scaling_and_confidence():
    pytest.importorskip("requests")
    from hmm_sentiment.labeler_scoring import _text_key

    arts = pd.DataFrame({"headline": ["h1"], "summary": [""]})
    key = _text_key("h1")
    cache = {"plan": {key: [{"name": "A", "rubric": "r"}, {"name": "B", "rubric": "r"}]},
             "solve": {(key, "A"): (-5, "a"), (key, "B"): (5, "b")},
             "final": {}}
    out = labeler_scores(arts, cache=cache, path=None)
    assert out.loc[0, "avg_impact"] == 0.0
    assert out.loc[0, "S_raw"] == 0.0
    # std of [-5, 5] is 5 -> confidence clipped to the 0.25 floor
    assert out.loc[0, "confidence"] == 0.25
    assert out.loc[0, "n_criteria"] == 2


# ------------------------------------------- audit-response regressions

def test_degraded_articles_read_as_nan(monkeypatch):
    """Fallback-labeled articles must surface as NaN so coverage gates bite
    (MAJOR-3/MAJOR-4 follow-up: fallback score-0 must not masquerade as a label)."""
    pytest.importorskip("requests")
    from hmm_sentiment.labeler_scoring import _text_key

    arts = pd.DataFrame({"headline": ["good", "bad"], "summary": ["", ""]})
    cache = {"plan": {}, "solve": {}, "final": {}}
    kg = _text_key("good")
    kb = _text_key("bad")
    cache["final"][kg] = {"avg_impact": 2.0, "sentiment": "Bullish",
                          "n_criteria": 2, "scores": [2, 2],
                          "plan_fallback": False, "solve_fallbacks": 0}
    cache["final"][kb] = {"avg_impact": 0.0, "sentiment": "Neutral",
                          "n_criteria": 3, "scores": [0, 0, 0],
                          "plan_fallback": False, "solve_fallbacks": 3}
    out = labeler_scores(arts, cache=cache, path=None)
    assert out.loc[0, "avg_impact"] == 2.0          # clean label kept
    assert np.isnan(out.loc[1, "avg_impact"])        # degraded -> NaN (gate bites)
    assert pd.isna(out.loc[1, "sentiment_lbl"])


def test_text_key_is_backend_and_version_scoped(monkeypatch):
    """Distinct backends/models/versions never share label space (MAJOR-2/7)."""
    from hmm_sentiment import labeler_scoring as ls

    monkeypatch.setattr(ls, "BACKEND", "glm")
    monkeypatch.setattr(ls, "GLM_MODEL", "glm-x")
    k_glm = ls._text_key("headline")
    monkeypatch.setattr(ls, "BACKEND", "openrouter")
    monkeypatch.setattr(ls, "OPENROUTER_MODEL", "llama-y")
    k_or = ls._text_key("headline")
    assert k_glm != k_or
    assert "glm-x" in k_glm and "llama-y" in k_or


def test_parse_rubric_list_caps_at_max_criteria():
    from hmm_sentiment.labeler_scoring import MAX_CRITERIA
    raw = json.dumps([{"name": f"c{i}", "rubric": "r"} for i in range(7)])
    out = parse_rubric_list(raw)
    assert len(out) == MAX_CRITERIA == 4


def test_failed_solves_and_plans_are_never_persisted(monkeypatch):
    """Degenerate results stay out of the cache so reruns retry (MAJOR-4)."""
    import tempfile

    from hmm_sentiment import labeler_scoring as ls

    def _boom(prompt, max_tokens=700, max_retries=1):
        raise RuntimeError("network disabled in test")

    monkeypatch.setattr(ls, "call_llm", _boom)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.close()
    cache = {"plan": {}, "solve": {}, "final": {}}
    res = ls.label_article("SPY", "headline", "kx", cache, path=tmp.name)
    assert res["plan_fallback"] is True
    assert res["solve_fallbacks"] == len(ls.FALLBACK_RUBRIC)
    with open(tmp.name) as f:
        assert f.read().strip() == "", "degraded article must not be cached"
    os.unlink(tmp.name)


def test_unrounded_mean_classification(monkeypatch):
    """Classification uses the unrounded mean (MINOR-6): scores [2, 1] -> mean
    1.5 exactly -> Bullish even though round(...,3) would still be 1.5; use a
    case where rounding could flip: [1, 1, 2] -> 1.333 Neutral; [-2, -1, 1, 1]
    -> -0.25 Neutral; [2, 2, -1, 1] -> 1.0 Neutral."""
    cache = {"plan": {}, "solve": {}, "final": {}}
    key = "kr"
    crits = [{"name": f"c{i}", "rubric": "r"} for i in range(4)]
    cache["plan"][key] = crits
    for c, s in zip(crits, [2, 2, -1, 1]):
        cache["solve"][(key, c["name"])] = (s, "r")
    res = label_article("SPY", "headline", key, cache, path=None)
    assert res["avg_impact"] == 1.0
    assert res["sentiment"] == "Neutral"


def test_provenance_in_persisted_records(monkeypatch):
    """Every persisted record carries backend/model/version (MAJOR-2)."""
    import tempfile

    from hmm_sentiment import labeler_scoring as ls

    monkeypatch.setattr(ls, "BACKEND", "openrouter")
    monkeypatch.setattr(ls, "OPENROUTER_MODEL", "test-model")
    cache = {"plan": {}, "solve": {}, "final": {}}
    key = "kp"
    cache["plan"][key] = [{"name": "A", "rubric": "r"}]
    cache["solve"][(key, "A")] = (3, "ok")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.close()
    ls.label_article("SPY", "headline", key, cache, path=tmp.name)
    recs = [json.loads(l) for l in open(tmp.name) if l.strip()]
    assert recs, "clean article must be persisted"
    for rec in recs:
        assert rec["backend"] == "openrouter"
        assert rec["model"] == "test-model"
        assert rec["version"] == ls.LABELER_VERSION
    os.unlink(tmp.name)
