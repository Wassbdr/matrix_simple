"""
gray_failure_detector.py - Gray failure detection inspired by GrayScope (FSE 2024).

Gray failures: 39% of production incidents (2025). Observable as collective
temporal-intermetric anomalies (Chandola 2009 / Zamanzadeh 2024) - individual
metrics stay within SLA thresholds while their joint correlation degrades.

Four detection mechanisms:
    1. P99/P50 ratio  (micro-stutter: CPU steal, L3 contention, GC pauses, CFS throttle)
    2. Inter-metric decorrelation  (Pearson corr(cpu, latency) break on sliding window)
    3. Monotone trend  (Mann-Kendall on memory - detects nascent Metastable Failures)
    4. Limplock signal  (queue_depth_rate / cpu_usage - Google 2023)

References:
    GrayScope (FSE 2024), GREYBOUND (ATC 2025), Limplock (Google 2023),
    Bronson et al. (OSDI 2022 - Metastable Failures)
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np


class GrayScore(NamedTuple):
    is_gray: bool
    confidence: float
    mechanism: str            # "p99_p50" | "decorrelation" | "trend" | "limplock" | "none"
    affected_nodes: List[int]


# ---------------------------------------------------------------------------
# 1. P99/P50 micro-stutter detector
# ---------------------------------------------------------------------------

def detect_p99_p50(
    window: np.ndarray,
    threshold: float = 3.0,
) -> Tuple[bool, float, List[int]]:
    """
    Per-node ratio of P99 to P50 across the time axis.

    Args:
        window: [N, W] - one metric (typically latency) for N nodes over W steps.
        threshold: ratio above which a node is flagged.
    Returns:
        (triggered, max_ratio, flagged_node_indices)
    """
    N, W = window.shape
    if W < 4:
        return False, 0.0, []
    p99 = np.percentile(window, 99, axis=1)
    p50 = np.percentile(window, 50, axis=1)
    p50_safe = np.where(np.abs(p50) < 1e-8, 1e-8, p50)
    ratios = p99 / p50_safe
    flagged = [int(i) for i in range(N) if ratios[i] > threshold]
    return len(flagged) > 0, float(ratios.max()), flagged


# ---------------------------------------------------------------------------
# 2. Inter-metric decorrelation
# ---------------------------------------------------------------------------

def detect_decorrelation(
    window_mts: np.ndarray,
    baseline_corr: Optional[np.ndarray] = None,
    drop_threshold: float = 0.4,
) -> Tuple[bool, float, List[int]]:
    """
    Detect inter-metric correlation break.

    Args:
        window_mts: [N, F, W] - all features for N nodes.
        baseline_corr: [F, F] - healthy baseline correlation.
        drop_threshold: mean absolute correlation drop trigger.
    Returns:
        (triggered, mean_drop, affected_nodes)
    """
    N, F, W = window_mts.shape
    if F < 2 or W < 4:
        return False, 0.0, []

    affected = []
    max_drop = 0.0
    for n in range(N):
        feat = window_mts[n]  # [F, W]
        corr = np.corrcoef(feat)
        corr = np.nan_to_num(corr, nan=0.0)
        if baseline_corr is not None:
            drop = float(np.abs(corr - baseline_corr).mean())
        else:
            half = W // 2
            c1 = np.corrcoef(feat[:, :half])
            c2 = np.corrcoef(feat[:, half:])
            c1, c2 = np.nan_to_num(c1), np.nan_to_num(c2)
            drop = float(np.abs(c1 - c2).mean())
        if drop > drop_threshold:
            affected.append(n)
        max_drop = max(max_drop, drop)
    return len(affected) > 0, max_drop, affected


# ---------------------------------------------------------------------------
# 3. Monotone trend (Mann-Kendall)
# ---------------------------------------------------------------------------

def _mann_kendall_s(x: np.ndarray) -> float:
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += int(np.sign(x[j] - x[i]))
    s_max = n * (n - 1) / 2
    return abs(s) / max(s_max, 1)


def detect_trend(
    window: np.ndarray,
    mk_threshold: float = 0.6,
) -> Tuple[bool, float, List[int]]:
    """
    Mann-Kendall trend on per-node metric (e.g. memory).

    Args:
        window: [N, W] - one metric for N nodes.
        mk_threshold: |S/Smax| above which trend is declared.
    Returns:
        (triggered, max_mk, affected_nodes)
    """
    N, W = window.shape
    if W < 4:
        return False, 0.0, []
    affected = []
    max_mk = 0.0
    for n in range(N):
        mk = _mann_kendall_s(window[n])
        if mk > mk_threshold:
            affected.append(n)
        max_mk = max(max_mk, mk)
    return len(affected) > 0, max_mk, affected


# ---------------------------------------------------------------------------
# 4. Limplock signal (Google 2023)
# ---------------------------------------------------------------------------

def detect_limplock(
    queue_depth_rate: np.ndarray,
    cpu_usage: np.ndarray,
    threshold: float = 5.0,
) -> Tuple[bool, float, List[int]]:
    """
    Limplock: queue depth grows while CPU stays flat.

    Args:
        queue_depth_rate: [N, W] - derivative of queue depth.
        cpu_usage: [N, W] - CPU utilization.
        threshold: ratio of mean(queue_depth_rate) / mean(cpu) above which flagged.
    """
    N, W = queue_depth_rate.shape
    affected = []
    max_ratio = 0.0
    for n in range(N):
        qd = float(np.mean(queue_depth_rate[n]))
        cpu = float(np.mean(cpu_usage[n]))
        ratio = qd / max(cpu, 1e-8)
        if ratio > threshold and qd > 0:
            affected.append(n)
        max_ratio = max(max_ratio, ratio)
    return len(affected) > 0, max_ratio, affected


# ---------------------------------------------------------------------------
# Composite detector
# ---------------------------------------------------------------------------

def detect_gray_failure(
    window_mts: np.ndarray,
    latency_idx: int = 4,
    memory_idx: int = 1,
    cpu_idx: int = 0,
    queue_depth_idx: Optional[int] = None,
    baseline_corr: Optional[np.ndarray] = None,
    p99_threshold: float = 3.0,
    decorr_threshold: float = 0.4,
    trend_threshold: float = 0.6,
    limplock_threshold: float = 5.0,
) -> GrayScore:
    """
    Run all 4 detectors and return the first triggered mechanism.

    Args:
        window_mts: [N, F, W] - node × feature × time.
        latency_idx, memory_idx, cpu_idx: feature indices.
        queue_depth_idx: feature index for queue depth (optional, enables Limplock).
        baseline_corr: [F, F] healthy baseline.
        limplock_threshold: ratio threshold for Limplock detection.
    """
    N, F, W = window_mts.shape

    # 1. P99/P50
    if latency_idx < F:
        ok, conf, nodes = detect_p99_p50(window_mts[:, latency_idx, :], p99_threshold)
        if ok:
            return GrayScore(True, conf, "p99_p50", nodes)

    # 2. Decorrelation
    ok, conf, nodes = detect_decorrelation(window_mts, baseline_corr, decorr_threshold)
    if ok:
        return GrayScore(True, conf, "decorrelation", nodes)

    # 3. Trend
    if memory_idx < F:
        ok, conf, nodes = detect_trend(window_mts[:, memory_idx, :], trend_threshold)
        if ok:
            return GrayScore(True, conf, "trend", nodes)

    # 4. Limplock (queue depth grows while CPU stays flat)
    if queue_depth_idx is not None and queue_depth_idx < F and cpu_idx < F:
        queue_depth = window_mts[:, queue_depth_idx, :]  # [N, W]
        if W >= 2:
            queue_depth_rate = np.diff(queue_depth, axis=1)  # [N, W-1]
            cpu_usage = window_mts[:, cpu_idx, 1:]  # [N, W-1] aligned with rate
            ok, conf, nodes = detect_limplock(queue_depth_rate, cpu_usage, limplock_threshold)
            if ok:
                return GrayScore(True, conf, "limplock", nodes)

    return GrayScore(False, 0.0, "none", [])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    N, F, W = 10, 5, 30

    # Test 1: No gray failure on uniform positive data (avoids P99/P50 noise near zero)
    clean = np.abs(np.random.randn(N, F, W)) + 1.0
    r = detect_gray_failure(clean, p99_threshold=5.0)
    assert not r.is_gray, f"false positive: {r}"
    print(f"  Test 1 PASS: clean → {r.mechanism}")

    # Test 2: P99/P50 spike
    lat = np.random.randn(N, W) * 0.1
    lat[3, -2:] = 20.0
    spiky = clean.copy()
    spiky[:, 4, :] = lat
    r = detect_gray_failure(spiky)
    assert r.is_gray and r.mechanism == "p99_p50", f"expected p99_p50: {r}"
    print(f"  Test 2 PASS: p99_p50 → nodes={r.affected_nodes}")

    # Test 3: Trend (memory leak)
    leaky = clean.copy()
    leaky[2, 1, :] = np.linspace(0, 10, W)
    r = detect_gray_failure(leaky)
    assert r.is_gray and r.mechanism == "trend", f"expected trend: {r}"
    print(f"  Test 3 PASS: trend → nodes={r.affected_nodes}")

    # Test 4: Limplock (standalone)
    qd = np.zeros((N, W))
    cpu = np.ones((N, W)) * 0.1
    qd[5] = np.linspace(0, 50, W)
    ok, conf, nodes = detect_limplock(qd, cpu, threshold=5.0)
    assert ok and 5 in nodes, f"expected limplock: {ok}, {nodes}"
    print(f"  Test 4 PASS: limplock (standalone) → nodes={nodes}")

    # Test 5: Limplock via composite detector (6 features: cpu, mem, net_rx, net_tx, latency, queue_depth)
    N6, F6, W6 = 10, 6, 30
    limplock_data = np.abs(np.random.randn(N6, F6, W6)) + 1.0
    # Node 7: queue depth grows (feature 5), CPU stays flat (feature 0)
    limplock_data[7, 0, :] = 0.1  # flat CPU
    limplock_data[7, 5, :] = np.linspace(0, 100, W6)  # growing queue depth
    r = detect_gray_failure(
        limplock_data,
        cpu_idx=0, memory_idx=1, latency_idx=4, queue_depth_idx=5,
        p99_threshold=10.0, decorr_threshold=0.8, trend_threshold=0.9,  # disable other detectors
        limplock_threshold=5.0,
    )
    assert r.is_gray and r.mechanism == "limplock", f"expected limplock via composite: {r}"
    assert 7 in r.affected_nodes, f"expected node 7 in affected: {r.affected_nodes}"
    print(f"  Test 5 PASS: limplock (composite) → nodes={r.affected_nodes}")

    print("\n=== gray_failure_detector tests passed ===")
