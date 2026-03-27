"""
Shared k-selection and GIB beta scaling for STA / Late Bottleneck training.

Single source of truth for benchmark and standalone train scripts.
"""

from __future__ import annotations

import math


def get_optimal_k(n_nodes: int) -> int:
    """
    Empirical k selection (ablation-validated):
        N <= 20  : ~25% of nodes (min 5)
        20 < N <= 100: ~15% of nodes (sweet spot at N=100)
        N > 100  : ~10% of nodes (large-scale)
    """
    if n_nodes <= 20:
        return max(5, int(round(n_nodes * 0.25)))
    if n_nodes <= 100:
        return max(5, int(round(n_nodes * 0.15)))
    return max(10, int(round(n_nodes * 0.10)))


def adaptive_gib_beta(beta_max: float, n_nodes: int, n_ref: int = 20) -> float:
    """
    Scale GIB regularization so larger N (more root-cause classes) is not over-sparsified.

    Exponent chosen so (N_ref/N)^exp gives ~4.5× reduction at N=100 vs N=20:
        (20/100)^exp = 1/4.5  =>  exp = log(4.5)/log(5) ≈ 0.88
    (using 0.873 for a ratio closer to exactly 4.5)
    """
    exp = math.log(4.5) / math.log(5.0)  # ~0.88; at N=100 vs N=20 ratio is ~4.5
    return float(beta_max * (n_ref / float(n_nodes)) ** exp)
