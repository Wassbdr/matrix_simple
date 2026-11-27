"""Post-hoc model ensembling and score calibration for RCA rankings.

STA and GATv2 are trained on the same split per seed and evaluated on the
same test cases, so their per-case logits can be combined without any
retraining. Averaging happens in probability space (softmax) because the two
models produce logits on different scales.

Calibration utilities (temperature scaling, ECE) operate on (logits, labels)
pairs; fitting must be done on held-out *training* predictions to avoid
test leakage - callers are responsible for that split discipline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Run = Tuple[List[int], List[np.ndarray]]  # (y_true, per-case logits)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def soft_vote(logits_a: np.ndarray, logits_b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """Probability-domain average of two models' scores for one case."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError(f"weight_a must be in [0, 1], got {weight_a}")
    return weight_a * softmax(logits_a) + (1.0 - weight_a) * softmax(logits_b)


def confidence_weighted_vote(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    """Soft vote where each model's weight is its own top-1 confidence."""
    p_a, p_b = softmax(logits_a), softmax(logits_b)
    c_a, c_b = float(p_a.max()), float(p_b.max())
    total = c_a + c_b
    if total <= 0:
        return 0.5 * (p_a + p_b)
    return (c_a / total) * p_a + (c_b / total) * p_b


def ensemble_runs(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    mode: str = "soft",
    weight_a: float = 0.5,
) -> List[Run]:
    """Combine per-seed eval runs of two models into ensemble runs.

    Runs are paired by position and only combined when their ``y_true``
    sequences match exactly (same split); mismatched pairs are skipped.
    """
    if mode not in ("soft", "confidence"):
        raise ValueError(f"unknown ensemble mode: {mode}")
    out: List[Run] = []
    for (yt_a, logits_a), (yt_b, logits_b) in zip(runs_a, runs_b):
        if list(yt_a) != list(yt_b) or len(logits_a) != len(logits_b):
            continue
        combined = []
        for za, zb in zip(logits_a, logits_b):
            if mode == "soft":
                combined.append(soft_vote(za, zb, weight_a=weight_a))
            else:
                combined.append(confidence_weighted_vote(za, zb))
        out.append((list(yt_a), combined))
    return out


def disagreement_rate(runs_a: Sequence[Run], runs_b: Sequence[Run]) -> float:
    """Fraction of paired test cases where the two models' top-1 differ."""
    total, differ = 0, 0
    for (yt_a, logits_a), (yt_b, logits_b) in zip(runs_a, runs_b):
        if list(yt_a) != list(yt_b) or len(logits_a) != len(logits_b):
            continue
        for za, zb in zip(logits_a, logits_b):
            total += 1
            if int(np.argmax(za)) != int(np.argmax(zb)):
                differ += 1
    return differ / total if total else 0.0


def oracle_top1(runs_a: Sequence[Run], runs_b: Sequence[Run]) -> float:
    """Top-1 accuracy of a perfect per-case router (upper bound for routing).

    For each case the oracle picks whichever model ranks the true root cause
    first; this bounds what any cascade/confidence routing scheme can reach.
    """
    total, hits = 0, 0
    for (yt_a, logits_a), (yt_b, logits_b) in zip(runs_a, runs_b):
        if list(yt_a) != list(yt_b) or len(logits_a) != len(logits_b):
            continue
        for y, za, zb in zip(yt_a, logits_a, logits_b):
            total += 1
            if int(np.argmax(za)) == int(y) or int(np.argmax(zb)) == int(y):
                hits += 1
    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return np.asarray(logits, dtype=np.float64) / temperature


def _nll(logits_list: Sequence[np.ndarray], y_true: Sequence[int], temperature: float) -> float:
    total = 0.0
    for z, y in zip(logits_list, y_true):
        p = softmax(temperature_scale(z, temperature))
        total -= float(np.log(max(p[int(y)], 1e-12)))
    return total / max(len(y_true), 1)


def fit_temperature(
    logits_list: Sequence[np.ndarray],
    y_true: Sequence[int],
    grid: Optional[Sequence[float]] = None,
) -> float:
    """NLL-minimizing temperature over a log-spaced grid (consistent, no scipy dep).

    Fit on held-out training predictions, never on the test split.
    """
    if grid is None:
        grid = np.logspace(-1, 1, 81)  # 0.1 .. 10
    return float(min(grid, key=lambda t: _nll(logits_list, y_true, float(t))))


def expected_calibration_error(
    logits_list: Sequence[np.ndarray],
    y_true: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Standard top-1 ECE with equal-width confidence bins."""
    confidences, corrects = [], []
    for z, y in zip(logits_list, y_true):
        p = softmax(z)
        confidences.append(float(p.max()))
        corrects.append(1.0 if int(np.argmax(p)) == int(y) else 0.0)
    conf = np.asarray(confidences)
    corr = np.asarray(corrects)
    n = len(conf)
    if n == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)
