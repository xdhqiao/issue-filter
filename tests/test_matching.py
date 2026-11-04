from dataclasses import replace
import json

import pytest

from conftest import FakeBackend
from issue_filter.config import Settings
from issue_filter.errors import ServiceError
from issue_filter.history import HistoryStore
from issue_filter.matcher import IssueMatcher
from issue_filter.nli import NLIProbabilities as P


def test_keyword_any_all_case_and_literal_substrings(settings):
    values = [{"id": "x", "key_words": ["NOT_INIT", "编译", "a+b"], "description": "风险"}]
    store = HistoryStore(values, settings)
    assert len(store.candidates("涉及编译")) == 1
    assert not store.candidates("not_init")
    assert len(store.candidates("x a+b x")) == 1
    assert not store.candidates("aab")
    assert HistoryStore(values, replace(settings, keyword_case_sensitive=False)).candidates("not_init")
    strict = HistoryStore(values, replace(settings, keyword_mode="all"))
    assert not strict.candidates("NOT_INIT")
    assert len(strict.candidates("编译 NOT_INIT a+b")) == 1


@pytest.mark.parametrize("mutation", ["empty_keyword", "empty_keywords", "wrong_keyword_type", "duplicate_id",
                                      "empty_description", "empty_id", "extra_field", "not_array"])
def test_bad_history_fails_at_startup(settings, history, mutation):
    if mutation == "empty_keyword":
        history[0]["key_words"] = [""]
    elif mutation == "empty_keywords":
        history[0]["key_words"] = []
    elif mutation == "wrong_keyword_type":
        history[0]["key_words"] = [123]
    elif mutation == "duplicate_id":
        history[1]["id"] = history[0]["id"]
    elif mutation == "empty_description":
        history[0]["description"] = " "
    elif mutation == "empty_id":
        history[0]["id"] = ""
    elif mutation == "extra_field":
        history[0]["other"] = "value"
    else:
        history = {}
    with pytest.raises(ServiceError):
        HistoryStore(history, settings)


def test_bom_file_and_more_than_one_hundred_items(settings, tmp_path):
    values = [{"id": f"i-{i}", "key_words": ["编译"], "description": "编译风险"} for i in range(500)]
    path = tmp_path / "history.json"
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8-sig")
    store = HistoryStore.load(replace(settings, history_path=path))
    assert len(store.candidates("编译失败")) == 500


def test_best_match_not_first_and_threshold(settings, history):
    candidates = list(HistoryStore(history, settings).issues)
    backend = FakeBackend([P(.02, .88, .10), P(.01, .97, .02)])
    matcher = IssueMatcher(settings, backend)
    assert matcher.match("NOT_INIT 编译失败", candidates) == "no_init_issue"
    backend.values = [P(.15, .80, .05), P(.80, .10, .10)]
    assert matcher.match("NOT_INIT 编译正常", candidates) is None
    backend.values = [P(.11, .89, .0), P(.07, .85, .08)]
    assert matcher.match("NOT_INIT 编译失败", candidates) == "no_init_issue"


def test_tie_is_stable_independent_of_candidate_order(settings, history):
    candidates = list(HistoryStore(history, settings).issues)
    matcher = IssueMatcher(settings, FakeBackend())
    assert matcher.match("编译 NOT_INIT", candidates) == "compile_issue"
    assert matcher.match("编译 NOT_INIT", candidates[::-1]) == "compile_issue"


def test_bidirectional_minimum_and_pair_alignment(settings, history):
    settings = replace(settings, match_mode="bidirectional")
    candidates = list(HistoryStore(history, settings).issues)
    backend = FakeBackend([P(.01, .98, .01), P(.02, .95, .03), P(.10, .40, .50), P(.02, .91, .07)])
    matcher = IssueMatcher(settings, backend)
    assert matcher.match("编译 NOT_INIT", candidates) == "no_init_issue"
    assert backend.calls[0] == [("编译 NOT_INIT", item.description) for item in candidates] + [
        (item.description, "编译 NOT_INIT") for item in candidates]


def test_busy_and_failure_release_lock(settings, history):
    candidates = list(HistoryStore(history, settings).issues)
    backend = FakeBackend(error=RuntimeError("OOM"))
    matcher = IssueMatcher(settings, backend)
    matcher._lock.acquire()
    with pytest.raises(ServiceError, match="其他请求"):
        matcher.match("编译 NOT_INIT", candidates)
    matcher._lock.release()
    with pytest.raises(ServiceError) as error:
        matcher.match("编译 NOT_INIT", candidates)
    assert error.value.status == 503
    backend.error = None
    assert matcher.match("编译 NOT_INIT", candidates) == "compile_issue"


def test_model_result_count_must_match_input(settings, history):
    matcher = IssueMatcher(settings, FakeBackend([]))
    with pytest.raises(ServiceError):
        matcher.match("编译", list(HistoryStore(history, settings).issues))


@pytest.mark.parametrize("values", [(float("nan"), 0, 1), (-1, 1, 1), (.2, .2, .2)])
def test_invalid_probabilities_fail(values):
    with pytest.raises(ValueError):
        P(*values)


@pytest.mark.parametrize("overrides", [
    {"threshold": float("nan")}, {"threshold": 1.1}, {"max_contradiction": -1},
    {"batch_size": 0}, {"keyword_mode": "unknown"}, {"device": "cpu", "quantization": "4bit"},
    {"inference_lock_timeout": float("inf")},
])
def test_invalid_settings_fail(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_environment_config(monkeypatch, tmp_path):
    monkeypatch.setenv("THRESHOLD", "0.90")
    monkeypatch.setenv("HISTORY_PATH", str(tmp_path / "data.json"))
    monkeypatch.setenv("KEYWORD_CASE_SENSITIVE", "false")
    settings = Settings.from_env()
    assert settings.threshold == .90
    assert settings.history_path == tmp_path / "data.json"
    assert not settings.keyword_case_sensitive
    monkeypatch.setenv("PRELOAD_MODEL", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()
