"""Unit tests for failure_type_utils (pytest or python)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_type_utils import split_logits_by_failure_type


class _D:
    def __init__(self, failure_type):
        self.failure_type = failure_type


def test_split_logits_by_failure_type():
    data = [_D("hard"), _D("gray"), _D("cascade"), _D("real")]
    y_true = [0, 1, 2, 3]
    y_logits = [[0.1], [0.2], [0.3], [0.4]]
    out = split_logits_by_failure_type(data, y_true, y_logits)

    assert out["hard"]["y_true"] == [0]
    assert out["gray"]["y_true"] == [1]
    assert out["cascade"]["y_true"] == [2]
    assert out["other"]["y_true"] == [3]


def test_split_length_mismatch_raises():
    data = [_D("hard")]
    try:
        split_logits_by_failure_type(data, [1, 2], [[0.1]])
    except ValueError:
        return
    raise AssertionError("Expected ValueError on mismatched lengths")


if __name__ == "__main__":
    test_split_logits_by_failure_type()
    test_split_length_mismatch_raises()
    print("ok: test_failure_type_utils")
