"""Tests for ensemble routing diagnostics (disagreement, oracle)."""

import numpy as np

from rca_pipeline.ensemble import disagreement_rate, oracle_top1


def _runs(y_true, top1s):
    logits = []
    for t in top1s:
        z = np.zeros(4)
        z[t] = 5.0
        logits.append(z)
    return [(y_true, logits)]


def test_disagreement_rate():
    a = _runs([0, 1, 2], [0, 1, 2])
    b = _runs([0, 1, 2], [0, 3, 2])
    assert disagreement_rate(a, b) == 1 / 3
    assert disagreement_rate(a, a) == 0.0
    assert disagreement_rate([], []) == 0.0


def test_oracle_picks_best_of_either():
    y = [0, 1, 2, 3]
    a = _runs(y, [0, 9 % 4, 2, 0])   # A right on cases 0, 1, 2
    b = _runs(y, [3, 3, 3, 3])       # B right on case 3 only
    assert oracle_top1(a, b) == 1.0   # perfect router fixes everything
    # Both wrong on a case → oracle cannot recover it.
    a2 = _runs(y, [1, 0, 0, 0])
    b2 = _runs(y, [2, 0, 0, 0])
    assert oracle_top1(a2, b2) == 0.0


def test_oracle_skips_mismatched_splits():
    a = _runs([0, 1], [0, 1])
    b = _runs([1, 0], [0, 1])
    assert oracle_top1(a, b) == 0.0
