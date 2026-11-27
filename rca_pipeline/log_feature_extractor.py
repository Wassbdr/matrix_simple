"""
log_feature_extractor.py - 8 features from log text (Chain-of-Event FSE 2024).

Features per node per window:
    0. error_freq:     count of ERROR/FATAL lines / total lines
    1. warn_freq:      count of WARN lines / total lines
    2. log_rate:       total lines / window duration
    3. error_burst:    max consecutive ERROR lines
    4. unique_errors:  unique error message count (normalized)
    5. oom_flag:       1.0 if "OOM" or "OutOfMemory" in logs
    6. restart_flag:   1.0 if "restart" or "CrashLoopBackOff" in logs
    7. timeout_flag:   1.0 if "timeout" or "deadline exceeded" in logs

Fallback: np.zeros(8) when logs unavailable.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

N_LOG_FEATURES = 8

_ERROR_RE = re.compile(r"\b(ERROR|FATAL|CRIT)\b", re.IGNORECASE)
_WARN_RE = re.compile(r"\bWARN(ING)?\b", re.IGNORECASE)
_OOM_RE = re.compile(r"\b(OOM|OutOfMemory|out.of.memory)\b", re.IGNORECASE)
_RESTART_RE = re.compile(r"\b(restart|CrashLoopBackOff)\b", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"\b(timeout|deadline.exceeded)\b", re.IGNORECASE)


def extract_log_features(
    log_lines: Optional[List[str]],
    window_seconds: float = 60.0,
    max_unique_errors: int = 50,
) -> np.ndarray:
    """
    Extract 8-dimensional feature vector from raw log lines.

    Args:
        log_lines: list of log strings for one node in one window.
                   If None or empty, returns zeros (graceful fallback).
        window_seconds: duration of the window (for log_rate).
        max_unique_errors: normalization cap for unique_errors.
    Returns:
        np.ndarray of shape [8], float32.
    """
    if not log_lines:
        return np.zeros(N_LOG_FEATURES, dtype=np.float32)

    total = len(log_lines)
    n_error = sum(1 for l in log_lines if _ERROR_RE.search(l))
    n_warn = sum(1 for l in log_lines if _WARN_RE.search(l))

    # Error burst: max consecutive error lines
    max_burst = 0
    current_burst = 0
    for line in log_lines:
        if _ERROR_RE.search(line):
            current_burst += 1
            max_burst = max(max_burst, current_burst)
        else:
            current_burst = 0

    unique_msgs = set()
    for line in log_lines:
        if _ERROR_RE.search(line):
            cleaned = re.sub(r"\d+", "#", line.strip()[:200])
            unique_msgs.add(cleaned)

    has_oom = 1.0 if any(_OOM_RE.search(l) for l in log_lines) else 0.0
    has_restart = 1.0 if any(_RESTART_RE.search(l) for l in log_lines) else 0.0
    has_timeout = 1.0 if any(_TIMEOUT_RE.search(l) for l in log_lines) else 0.0

    return np.array([
        n_error / max(total, 1),
        n_warn / max(total, 1),
        total / max(window_seconds, 1e-3),
        max_burst / max(total, 1),
        len(unique_msgs) / max_unique_errors,
        has_oom,
        has_restart,
        has_timeout,
    ], dtype=np.float32)


def extract_log_features_batch(
    logs_per_node: List[Optional[List[str]]],
    window_seconds: float = 60.0,
) -> np.ndarray:
    """
    Extract features for N nodes.

    Args:
        logs_per_node: list of length N, each element is a list of log lines or None.
    Returns:
        np.ndarray of shape [N, 8].
    """
    return np.stack([
        extract_log_features(logs, window_seconds) for logs in logs_per_node
    ])


if __name__ == "__main__":
    sample_logs = [
        "2025-01-01 INFO Starting service",
        "2025-01-01 ERROR Connection refused to db-001",
        "2025-01-01 ERROR Connection refused to db-001",
        "2025-01-01 ERROR OutOfMemory: heap space",
        "2025-01-01 WARN Slow query detected",
        "2025-01-01 INFO Request completed",
        "2025-01-01 ERROR timeout connecting to cache-002",
    ]
    feats = extract_log_features(sample_logs)
    assert feats.shape == (8,)
    assert feats[5] == 1.0  # oom_flag
    assert feats[7] == 1.0  # timeout_flag
    print(f"  Features: {feats}")

    empty = extract_log_features(None)
    assert np.all(empty == 0.0)
    print("  Fallback (None): all zeros")

    print("\n=== log_feature_extractor tests passed ===")
