"""API contract tests for the STA inference service (serving/serve.py)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rca_pipeline.ranking import Ranker
from serving.tests.conftest import NODE_NAMES, F, N, W


@pytest.fixture()
def client(serve_module):
    # Context manager triggers the startup event (bundle load).
    with TestClient(serve_module.app) as c:
        yield c


def _window(n=N, f=F, w=W):
    return np.random.default_rng(0).normal(size=(n, f, w)).tolist()


def test_health_ready_info(client):
    assert client.get("/health").status_code == 200

    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}

    info = client.get("/info").json()
    assert info["node_names"] == NODE_NAMES
    assert info["config"]["num_nodes"] == N
    assert "model_card" in info


def test_predict_happy_path(client):
    r = client.post("/predict", json={"window": _window(), "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 3
    assert body["latency_ms"] >= 0
    ranks = [c["rank"] for c in body["candidates"]]
    scores = [c["score"] for c in body["candidates"]]
    assert ranks == [1, 2, 3]
    assert scores == sorted(scores, reverse=True)
    assert all(c["service"] in NODE_NAMES for c in body["candidates"])


def test_predict_shape_mismatches(client):
    r = client.post("/predict", json={"window": _window(n=N + 1)})
    assert r.status_code == 400
    assert "N mismatch" in r.json()["detail"]

    r = client.post("/predict", json={"window": _window(f=F + 1)})
    assert r.status_code == 400
    assert "F mismatch" in r.json()["detail"]

    # Wrong nesting depth is caught by pydantic before the handler runs.
    r = client.post("/predict", json={"window": [[1.0, 2.0]]})
    assert r.status_code == 422


def test_predict_ragged_window_rejected(client):
    ragged = _window()
    ragged[0][0] = ragged[0][0][:-1]
    r = client.post("/predict", json={"window": ragged})
    assert r.status_code == 400


def test_predict_top_k_bounds(client):
    r = client.post("/predict", json={"window": _window(), "top_k": 0})
    assert r.status_code == 422


def test_endpoints_503_when_bundle_missing(serve_module):
    loaded = serve_module._BUNDLE
    serve_module._BUNDLE = None
    try:
        plain = TestClient(serve_module.app)  # no context manager: no startup
        assert plain.get("/health").status_code == 503
        assert plain.get("/ready").status_code == 503
        assert plain.get("/info").status_code == 503
        r = plain.post("/predict", json={"window": _window()})
        assert r.status_code == 503
    finally:
        serve_module._BUNDLE = loaded


def test_predict_realigns_named_services(client):
    window = np.random.RandomState(0).randn(len(NODE_NAMES), F, W)
    ref = client.post("/predict", json={"window": window.tolist(), "services": NODE_NAMES, "top_k": 4})
    assert ref.status_code == 200
    perm = [2, 0, 3, 1]
    shuffled = client.post("/predict", json={
        "window": window[perm].tolist(), "services": [NODE_NAMES[i] for i in perm], "top_k": 4})
    assert shuffled.status_code == 200
    assert [c["service"] for c in shuffled.json()["candidates"]] == [c["service"] for c in ref.json()["candidates"]]


def test_predict_rejects_unknown_services(client):
    window = np.zeros((len(NODE_NAMES), F, W)).tolist()
    r = client.post("/predict", json={"window": window, "services": ["x", "y", "z", "w"]})
    assert r.status_code == 400
    assert "services do not match" in r.json()["detail"]


def test_default_model_ranks_metrics(serve_module, monkeypatch):
    monkeypatch.setattr(serve_module, "BUNDLE_PATH", None)
    monkeypatch.setattr(serve_module, "_BUNDLE", None)
    monkeypatch.setattr(serve_module, "_RANKER", None)
    payload = {"metrics": {"time": [0, 1, 2, 3], "cart_cpu": [0, 0, 3, 5],
                           "orders_cpu": [1, 1, 1, 1]}, "inject_time": 2}
    with TestClient(serve_module.app) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/info").json()["request"] == "metrics"
        result = client.post("/predict", json=payload)
        assert result.status_code == 200
        expected = Ranker().version
        assert result.json()["model_version"] == expected
        assert {c["service"] for c in result.json()["candidates"]} == {"cart", "orders"}
        result = client.post("/predict", json={**payload, "inject_time": 4})
        assert result.status_code == 400
        assert "Supply samples before and at or after" in result.json()["detail"]
