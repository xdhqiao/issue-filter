"""Run labeled descriptions through keyword recall and NLI; emit JSON metrics."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from issue_filter.config import Settings
from issue_filter.history import HistoryStore, validate_description
from issue_filter.matcher import IssueMatcher
from issue_filter.nli import TransformersNLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="JSONL with description and expected_id (string or null)")
    parser.add_argument("--thresholds", default="0.75,0.80,0.85,0.90,0.95")
    args = parser.parse_args()
    settings = Settings.from_env()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    # Validate the full dataset/configuration before allocating model memory.
    configs = [replace(settings, threshold=value) for value in thresholds]
    store = HistoryStore.load(settings)
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        parser.error("dataset is empty")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"description", "expected_id"}:
            parser.error("each row must contain description and expected_id")
        validate_description(row["description"], settings.max_description_chars)
        if row["expected_id"] is not None and (
            not isinstance(row["expected_id"], str) or row["expected_id"] not in store.by_id
        ):
            parser.error("expected_id must be a known string ID or null")
    matcher = IssueMatcher(settings, TransformersNLI(settings))
    results = []
    for row in rows:
        candidates = store.candidates(row["description"])
        results.append((row, matcher.score(row["description"], candidates), {item.id for item in candidates}))
    positive_count = sum(row["expected_id"] is not None for row in rows)
    recall_hits = sum(row["expected_id"] in candidates for row, _, candidates in results if row["expected_id"] is not None)
    metrics = []
    for config in configs:
        selector = IssueMatcher(config, matcher.backend)
        correct = predicted = false_filters = wrong_ids = exact = 0
        for row, scores, _ in results:
            prediction = selector.select(scores)
            expected = row["expected_id"]
            predicted += prediction is not None
            correct += prediction is not None and prediction == expected
            false_filters += prediction is not None and expected is None
            wrong_ids += prediction is not None and expected is not None and prediction != expected
            exact += prediction == expected
        metrics.append({
            "threshold": config.threshold,
            "id_precision": correct / predicted if predicted else None,
            "id_recall": correct / positive_count if positive_count else None,
            "exact_accuracy": exact / len(rows),
            "false_filters": false_filters,
            "wrong_ids": wrong_ids,
            "predicted_matches": predicted,
        })
    print(json.dumps({
        "samples": len(rows),
        "keyword_recall": recall_hits / positive_count if positive_count else None,
        "match_mode": settings.match_mode,
        "quantization": settings.quantization,
        "max_contradiction": settings.max_contradiction,
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
