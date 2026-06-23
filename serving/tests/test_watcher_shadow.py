"""Tests for the shadow-deployment comparison (serving/rca_watcher.py M2.6)."""

import numpy as np

from serving import rca_watcher as watcher


def _window():
    return np.zeros((watcher.N, watcher.F, watcher.W), dtype=np.float32)


def test_compare_shadow_noop_without_url(monkeypatch):
    monkeypatch.setattr(watcher, "SHADOW_INFERENCE_URL", "")
    called = []
    monkeypatch.setattr(watcher, "call_predict", lambda *a, **k: called.append(1))
    champion = [("orders-db", 5.0, 1)]
    watcher.compare_shadow(champion, _window())
    assert called == []   # challenger never queried when shadow disabled


def test_compare_shadow_records_agreement(monkeypatch):
    monkeypatch.setattr(watcher, "SHADOW_INFERENCE_URL", "http://shadow:8080")
    # Challenger agrees on top-1 but with a lower score.
    monkeypatch.setattr(watcher, "call_predict",
                        lambda w, url: [("orders-db", 4.0, 1), ("orders", 1.0, 2)])
    champion = [("orders-db", 5.0, 1), ("orders", 2.0, 2)]

    before = watcher.WATCHER_SHADOW_AGREE.labels(agree="true")._value.get()
    watcher.compare_shadow(champion, _window())
    after = watcher.WATCHER_SHADOW_AGREE.labels(agree="true")._value.get()
    assert after == before + 1
    assert watcher.WATCHER_SHADOW_SCORE_DELTA._value.get() == 1.0   # 5.0 - 4.0


def test_compare_shadow_records_disagreement(monkeypatch):
    monkeypatch.setattr(watcher, "SHADOW_INFERENCE_URL", "http://shadow:8080")
    monkeypatch.setattr(watcher, "call_predict",
                        lambda w, url: [("orders", 3.0, 1)])
    champion = [("orders-db", 5.0, 1)]
    before = watcher.WATCHER_SHADOW_AGREE.labels(agree="false")._value.get()
    watcher.compare_shadow(champion, _window())
    after = watcher.WATCHER_SHADOW_AGREE.labels(agree="false")._value.get()
    assert after == before + 1


def test_compare_shadow_survives_challenger_failure(monkeypatch):
    monkeypatch.setattr(watcher, "SHADOW_INFERENCE_URL", "http://shadow:8080")
    monkeypatch.setattr(watcher, "call_predict", lambda w, url: None)
    # Must not raise when the challenger is unreachable.
    watcher.compare_shadow([("orders-db", 5.0, 1)], _window())
