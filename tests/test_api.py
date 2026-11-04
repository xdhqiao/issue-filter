from dataclasses import replace

import pytest

from issue_filter import create_app
from issue_filter.errors import InputTooLong


@pytest.fixture
def app(settings, history, backend):
    return create_app(settings, history=history, backend=backend)


def test_two_stage_contract(app, history, backend):
    client = app.test_client()
    text = "NOT_INIT 未定义，可能导致编译不通过。"
    candidates = client.post("/api/v1/potential-issues", json={"description": text})
    assert candidates.status_code == 200
    assert candidates.json == history
    assert backend.calls == []
    result = client.post("/api/v1/match", json={
        "description": text, "potential_historical_issue_list": candidates.json,
    })
    assert result.status_code == 200
    assert result.json == {"matched_issue_id": "compile_issue"}
    assert len(backend.calls[0]) == 2


def test_empty_recall_and_candidates_do_not_load_model(app, backend):
    client = app.test_client()
    assert client.post("/api/v1/potential-issues", json={"description": "内存泄漏"}).json == []
    result = client.post("/api/v1/match", json={"description": "内存泄漏", "potential_historical_issue_list": []})
    assert result.json == {"matched_issue_id": None}
    assert backend.calls == []


@pytest.mark.parametrize("body", [None, [], "x", {}, {"description": 1}, {"description": " "},
                                      {"description": "编译", "unexpected": True}])
def test_invalid_bodies(app, body):
    response = app.test_client().post("/api/v1/potential-issues", json=body, content_type="application/json")
    assert response.status_code == 400
    assert "error" in response.json


def test_content_type_and_malformed_json(app):
    client = app.test_client()
    assert client.post("/api/v1/potential-issues", data="text").status_code == 415
    response = client.post("/api/v1/potential-issues", data="{", content_type="application/json")
    assert response.status_code == 400
    assert response.is_json
    assert client.get("/missing").status_code == 404
    assert client.get("/api/v1/match").status_code == 405


@pytest.mark.parametrize("kind", ["unknown", "changed", "duplicate", "keyword_miss", "not_list", "invalid_item"])
def test_reject_invalid_candidates(app, history, backend, kind):
    candidates = [history[0].copy()]
    text = "编译不通过"
    if kind == "unknown":
        candidates[0]["id"] = "unknown"
    elif kind == "changed":
        candidates[0]["description"] = "任意新描述"
    elif kind == "duplicate":
        candidates *= 2
    elif kind == "keyword_miss":
        text = "内存泄漏"
    elif kind == "not_list":
        candidates = "oops"
    else:
        candidates = [None]
    result = app.test_client().post("/api/v1/match", json={
        "description": text, "potential_historical_issue_list": candidates,
    })
    assert result.status_code == 400
    assert backend.calls == []


def test_model_errors_are_not_null_matches(app, history, backend):
    backend.error = RuntimeError("private input or device details")
    result = app.test_client().post("/api/v1/match", json={
        "description": "编译不通过", "potential_historical_issue_list": [history[0]],
    })
    assert result.status_code == 503
    assert result.json["error"]["code"] == "model_unavailable"
    assert "matched_issue_id" not in result.json
    assert "private" not in result.text
    assert result.headers["Retry-After"] == "1"
    backend.error = InputTooLong(2048)
    assert app.test_client().post("/api/v1/match", json={
        "description": "编译不通过", "potential_historical_issue_list": [history[0]],
    }).status_code == 422


def test_request_and_description_limits(settings, backend):
    app = create_app(replace(settings, max_content_length=150, max_description_chars=10), backend=backend, history=[])
    client = app.test_client()
    assert client.post("/api/v1/potential-issues", json={"description": "x" * 11}).status_code == 400
    assert client.post("/api/v1/potential-issues", json={"description": "x" * 500}).status_code == 413


def test_authentication_and_health(settings, history, backend):
    app = create_app(replace(settings, api_key="test-secret"), history=history, backend=backend)
    client = app.test_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    assert client.post("/api/v1/potential-issues", json={"description": "编译"}).status_code == 401
    assert client.post("/api/v1/potential-issues", json={"description": "编译"},
                       headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_preload_performs_a_real_backend_call(settings, history, backend):
    app = create_app(replace(settings, preload_model=True), history=history, backend=backend)
    assert len(backend.calls) == 1
    assert app.test_client().get("/readyz").status_code == 200


def test_readiness_tracks_inference_failures_and_recovery(app, history, backend):
    client = app.test_client()
    payload = {"description": "编译不通过", "potential_historical_issue_list": [history[0]]}
    assert client.get("/readyz").status_code == 503
    assert client.post("/api/v1/match", json=payload).status_code == 200
    assert client.get("/readyz").status_code == 200
    backend.error = RuntimeError("OOM")
    assert client.post("/api/v1/match", json=payload).status_code == 503
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200
    backend.error = None
    assert client.post("/api/v1/match", json=payload).status_code == 200
    assert client.get("/readyz").status_code == 200
