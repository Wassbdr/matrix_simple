"""Tests for the PSI drift fallback (rca_pipeline/monitoring/drift_job.py).

Covers the Evidently-free path so the job is testable in CI without the
[monitoring] extra.
"""

import numpy as np

from rca_pipeline.monitoring.drift_job import (
    _fetch_current,
    _flatten_windows,
    _psi,
    fallback_drift,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    # Same distribution, different draw -> PSI near 0.
    y = rng.normal(size=2000)
    assert _psi(x, y) < 0.05


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=2000)
    cur = rng.normal(3, 1, size=2000)   # mean shifted by 3 sigma
    assert _psi(ref, cur) > 0.5


def test_fallback_drift_flags_shifted_feature_only():
    rng = np.random.default_rng(1)
    n = 1500
    ref = np.column_stack([rng.normal(0, 1, n), rng.normal(5, 2, n), rng.normal(0, 1, n)])
    cur = np.column_stack([rng.normal(0, 1, n), rng.normal(5, 2, n), rng.normal(4, 1, n)])
    res = fallback_drift(ref, cur)
    assert res["n_features"] == 3
    assert res["n_drifted"] == 1                    # only feature 2 shifted
    assert 0.0 < res["drift_share"] < 0.5
    assert res["per_feature"]["network_rx"]["drifted"] is True   # 3rd metric name
    assert res["per_feature"]["cpu_usage"]["drifted"] is False


def test_flatten_windows_shapes():
    assert _flatten_windows(np.zeros((4, 14, 6, 30))).shape == (4, 6)   # [B,N,F,W]
    assert _flatten_windows(np.zeros((14, 6, 30))).shape == (14, 6)     # [N,F,W]
    assert _flatten_windows(np.zeros((20, 6))).shape == (20, 6)         # already 2-D


def test_fetch_current_stacks_sub_windows(monkeypatch):
    """The current sample must hold one row per (node, sub-window), not just
    N rows from a single window — otherwise PSI compares ~14 points against
    the reference's thousands."""
    N, F, W = 14, 6, 12

    class _FakeConn:
        def __init__(self, *a, **k):
            pass

        def fetch_window(self, start, end, window_size):
            # PrometheusConnector returns [W, N, F].
            return np.ones((window_size, N, F), dtype=np.float32), [f"p{i}" for i in range(N)]

    # _fetch_current imports PrometheusConnector lazily from this module.
    import rca_pipeline.prometheus_connector as pc
    monkeypatch.setattr(pc, "PrometheusConnector", _FakeConn)

    # 1h lookback / 6min sub-windows = 10 chunks -> 10 * N rows.
    current = _fetch_current("http://prom", "sock-shop", lookback_hours=1,
                             sub_window_minutes=6, window_size=W)
    assert current.shape == (10 * N, F)


def test_fetch_current_skips_empty_slices(monkeypatch):
    N, F = 14, 6

    class _FakeConn:
        def __init__(self, *a, **k):
            self.calls = 0

        def fetch_window(self, start, end, window_size):
            self.calls += 1
            if self.calls == 1:                       # first slice: no pods
                return np.zeros((window_size, 0, F), dtype=np.float32), []
            return np.ones((window_size, N, F), dtype=np.float32), ["p"] * N

    import rca_pipeline.prometheus_connector as pc
    monkeypatch.setattr(pc, "PrometheusConnector", _FakeConn)
    current = _fetch_current("http://prom", "sock-shop", lookback_hours=1,
                             sub_window_minutes=20)   # 3 slices, first empty
    assert current.shape == (2 * N, F)                # only the 2 non-empty slices
