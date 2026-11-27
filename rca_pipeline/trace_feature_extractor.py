"""
trace_feature_extractor.py - 6 features from distributed traces (RUN ASE 2024).

Features per node per window:
    0. call_depth:        mean call depth across spans
    1. error_span_ratio:  fraction of spans with error status
    2. p99_span_ms:       P99 span duration in ms
    3. upstream_errors:    count of error spans from parent services
    4. downstream_errors:  count of error spans in child services
    5. retry_count:        estimated retries (spans with same operation > 1)

Fallback: np.zeros(6) when traces unavailable.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np

N_TRACE_FEATURES = 6


class Span(NamedTuple):
    service: str
    operation: str
    duration_ms: float
    has_error: bool
    parent_service: Optional[str]
    depth: int


def extract_trace_features(
    spans: Optional[List[Span]],
    node_name: str = "",
) -> np.ndarray:
    """
    Extract 6-dim feature vector from spans belonging to a single node.

    Args:
        spans: list of Span objects for this node in this window.
               If None or empty, returns zeros.
    Returns:
        np.ndarray [6], float32.
    """
    if not spans:
        return np.zeros(N_TRACE_FEATURES, dtype=np.float32)

    own_spans = [s for s in spans if s.service == node_name] if node_name else spans
    if not own_spans:
        return np.zeros(N_TRACE_FEATURES, dtype=np.float32)

    depths = [s.depth for s in own_spans]
    call_depth = float(np.mean(depths))

    n_errors = sum(1 for s in own_spans if s.has_error)
    error_ratio = n_errors / len(own_spans)

    durations = [s.duration_ms for s in own_spans]
    p99 = float(np.percentile(durations, 99)) if durations else 0.0

    upstream_errors = sum(
        1 for s in spans
        if s.has_error and s.parent_service and s.parent_service != node_name
    )

    child_services = {s.service for s in spans if s.parent_service == node_name and s.service != node_name}
    downstream_errors = sum(
        1 for s in spans if s.service in child_services and s.has_error
    )

    from collections import Counter
    op_counts = Counter(s.operation for s in own_spans)
    retries = sum(max(0, c - 1) for c in op_counts.values())

    return np.array([
        call_depth,
        error_ratio,
        p99,
        float(upstream_errors),
        float(downstream_errors),
        float(retries),
    ], dtype=np.float32)


def extract_trace_features_batch(
    all_spans: List[Span],
    node_names: List[str],
) -> np.ndarray:
    """
    Extract features for N nodes from all spans in a window.

    Returns:
        np.ndarray [N, 6].
    """
    return np.stack([
        extract_trace_features(all_spans, name) for name in node_names
    ])


if __name__ == "__main__":
    spans = [
        Span("svc-a", "GET /api", 15.0, False, None, 0),
        Span("svc-a", "GET /api", 120.0, True, None, 0),
        Span("svc-a", "GET /api", 20.0, False, None, 0),
        Span("svc-b", "POST /db", 50.0, True, "svc-a", 1),
        Span("svc-b", "POST /db", 45.0, False, "svc-a", 1),
    ]

    f_a = extract_trace_features(spans, "svc-a")
    assert f_a.shape == (6,)
    assert f_a[1] > 0  # error_ratio
    print(f"  svc-a: {f_a}")

    f_b = extract_trace_features(spans, "svc-b")
    print(f"  svc-b: {f_b}")

    empty = extract_trace_features(None)
    assert np.all(empty == 0.0)
    print("  Fallback: all zeros")

    batch = extract_trace_features_batch(spans, ["svc-a", "svc-b", "svc-c"])
    assert batch.shape == (3, 6)
    print(f"  Batch shape: {batch.shape}")

    print("\n=== trace_feature_extractor tests passed ===")
