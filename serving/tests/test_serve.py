"""API contract tests for the STA inference service (serving/serve.py)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

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
