import json
from pathlib import Path

from issue_filter.history import HistoryStore
from scripts.benchmark_api import summarize


def test_fixed_benchmark_labels_and_expected_keyword_candidates(settings):
    path = Path(__file__).resolve().parent.parent / "data/benchmark.v1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    store = HistoryStore.load(settings)
    assert len(rows) == 120
    assert sum(row["expected_id"] is not None for row in rows) == 60
    assert len({row["case_id"] for row in rows}) == 120
    assert len({row["description"] for row in rows}) == 120
    for row in rows:
        assert row["expected_id"] is None or row["expected_id"] in store.by_id
        assert sorted(row["expected_candidate_ids"]) == sorted(issue.id for issue in store.candidates(row["description"]))


def test_metrics_separate_wrong_id_and_api_failure_from_correct_rejections():
    def row(expected, predicted, success=True):
        return {"expected_id": expected, "predicted_id": predicted, "success": success,
                "recall_status": 200, "candidate_ids": ["a", "b"], "expected_candidate_ids": ["a", "b"],
                "recall_seconds": .1, "match_seconds": .9}

    metrics = summarize([
        row("a", "a"), row("a", "b"), row("a", None),
        row(None, "a"), row(None, None), row(None, None, success=False),
    ])
    assert metrics["exact_correct"] == 2
    assert metrics["exact_accuracy"] == 2 / 6
    assert metrics["binary_correct"] == 3
    assert metrics["api_errors"] == 1
    assert metrics["wrong_ids"] == 1
    assert metrics["false_filter_rate"] == 1 / 3
    assert metrics["confusion_matrix"] == {"TP": 2, "FP": 1, "TN": 1, "FN": 1}
