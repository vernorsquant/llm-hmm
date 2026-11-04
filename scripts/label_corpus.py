"""Label the full 2023 GDELT corpus through the LLM-Labeler rubric-decomposition
pipeline (GLM backend, content-hash JSONL cache, crash-resumable).

Writes:
  data/cache/llm_labeler_scores.jsonl   (per-call records, last-wins)
  outputs/tables/labeler_scores_sample.csv  (30 labeled articles w/ breakdowns)
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hmm_sentiment.labeler_scoring import (
    _text_key,
    load_labeler_cache,
    labeler_scores,
)
from hmm_sentiment.news_corpus import load_corpus


def main():
    news = load_corpus()
    print(f"[label-corpus] corpus rows={len(news)}", flush=True)
    out = labeler_scores(news)
    done = out["avg_impact"].notna().sum()
    print(f"[label-corpus] labeled={done}/{len(out)}", flush=True)

    # human-readable sample: 30 articles across the sentiment spectrum
    labeled = out.dropna(subset=["avg_impact"]).copy()
    sample = pd.concat([
        labeled.nlargest(10, "avg_impact"),
        labeled.nsmallest(10, "avg_impact"),
        labeled.sample(10, random_state=7),
    ]).drop_duplicates(subset=["headline"])
    cache = load_labeler_cache()
    rows = []
    for _, a in sample.iterrows():
        k = _text_key(str(a["headline"]))
        f = cache["final"].get(k, {})
        breakdown = "; ".join(
            f"{name}={s}" for name, s in zip(
                [c["name"] for c in cache["plan"].get(k, [])], f.get("scores", [])))
        rows.append({
            "timestamp_utc": a["timestamp_utc"], "source": a["source"],
            "headline": a["headline"], "avg_impact": f.get("avg_impact"),
            "sentiment": f.get("sentiment"), "criteria_scores": breakdown,
            "backend": f.get("backend"), "model": f.get("model"),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "tables", "labeler_scores_sample.csv"), index=False)
    print("[label-corpus] sample CSV written", flush=True)
    print("LABEL_CORPUS_DONE", flush=True)


if __name__ == "__main__":
    main()
