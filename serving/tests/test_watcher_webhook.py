"""Webhook-driven alert history tests for the watcher (serving/rca_watcher.py).

The TestClient is used without a context manager so the startup event (and
its Prometheus polling thread) never runs; ``_emit_ntfy`` is monkeypatched so
no network call leaves the process.
"""

import pytest
from fastapi.testclient import TestClient

from serving import rca_watcher as watcher


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(watcher, "_emit_ntfy", lambda *a, **k: True)
    with watcher._STATE_LOCK:
        watcher._WEBHOOK_ALERTS.clear()
    return TestClient(watcher.app)


def _am_payload(status="firing", root_cause="orders-db", severity="critical"):
    return {
        "status": status,
        "commonLabels": {"namespace": "sock-shop"},
        "alerts": [
            {
                "labels": {
                    "alertname": "RCAStaServeRootCause",
                    "severity": severity,
                    "root_cause": root_cause,
                },
                "annotations": {"summary": f"root cause {root_cause}"},
                "startsAt": "2026-06-12T00:00:00Z",
                "fingerprint": "abcdef123456",
            }
        ],
    }


def test_ntfy_relay_feeds_webhook_alerts(client):
    r = client.post("/ntfy-relay", json=_am_payload())
    assert r.status_code == 200
    assert r.json() == {"received": 1, "delivered": 1, "status": "firing"}

    alerts = client.get("/api/alerts").json()
    assert alerts["source"] == "webhook"
    assert alerts["count"] == 1
    assert alerts["alerts"][0]["labels"]["root_cause"] == "orders-db"
    assert alerts["alerts"][0]["status"] == "firing"
    assert "receivedAt" in alerts["alerts"][0]


def test_resolved_alerts_excluded_from_active(client):
    client.post("/ntfy-relay", json=_am_payload(status="firing"))
    client.post("/ntfy-relay", json=_am_payload(status="resolved"))

    # /api/alerts must only surface firing entries...
    active = [a for a in client.get("/api/alerts").json()["alerts"]
              if a.get("status") != "resolved"]
    assert all(a["status"] == "firing" for a in active)

    # ...while the full history keeps both firing and resolved.
    items = client.get("/api/webhook-alerts").json()["items"]
    assert [i["status"] for i in items] == ["firing", "resolved"]


def test_alerts_fallback_when_no_webhook(client, monkeypatch):
    def _no_am(*a, **k):
        raise watcher.requests.RequestException("AM down")

    monkeypatch.setattr(watcher.requests, "get", _no_am)
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "alerts": [], "source": "webhook"}


def test_ntfy_relay_rejects_invalid_json(client):
    r = client.post("/ntfy-relay", content=b"not-json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_ntfy_relay_counts_failed_deliveries(client, monkeypatch):
    monkeypatch.setattr(watcher, "_emit_ntfy", lambda *a, **k: False)
    r = client.post("/ntfy-relay", json=_am_payload())
    assert r.json() == {"received": 1, "delivered": 0, "status": "firing"}
