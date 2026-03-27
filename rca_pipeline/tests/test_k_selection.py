"""Unit tests for k_selection helpers (pytest or `python tests/test_k_selection.py`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from k_selection import get_optimal_k, adaptive_gib_beta


def test_get_optimal_k():
    assert get_optimal_k(20) == max(5, round(20 * 0.25))
    assert get_optimal_k(100) == max(5, round(100 * 0.15))
    assert get_optimal_k(500) == max(10, round(500 * 0.10))


def test_adaptive_gib_beta_ratio_n100_vs_n20():
    b20 = adaptive_gib_beta(1.0, 20)
    b100 = adaptive_gib_beta(1.0, 100)
    assert abs(b20 - 1.0) < 1e-6
    ratio = b20 / b100
    assert 4.4 < ratio < 4.6, f"expected ~4.5x smaller at N=100, got ratio {ratio}"


if __name__ == "__main__":
    test_get_optimal_k()
    test_adaptive_gib_beta_ratio_n100_vs_n20()
    print("ok: test_k_selection")
