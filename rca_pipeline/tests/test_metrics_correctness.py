"""Small correctness checks for core RCAEval metrics."""

import numpy as np

from rca_pipeline.metrics_rcaeval import avg_at_k, mrr, ndcg_at_k


def test_avg_at_1_perfect():
    y_true = [0, 1, 2]
    y_logits = [np.array([10, 0, 0]), np.array([0, 10, 0]), np.array([0, 0, 10])]
    assert avg_at_k(y_true, y_logits, k=1) == 1.0


def test_avg_at_1_all_wrong():
    y_true = [0, 1, 2]
    y_logits = [np.array([0, 10, 0]), np.array([10, 0, 0]), np.array([10, 0, 0])]
    assert avg_at_k(y_true, y_logits, k=1) == 0.0


def test_mrr_rank_2():
    y_true = [0]
    y_logits = [np.array([5, 10, 0])]
    assert mrr(y_true, y_logits) == 0.5


def test_ndcg_at_3_rank_1():
    y_true = [1]
    y_logits = [np.array([0, 10, 0])]
    assert ndcg_at_k(y_true, y_logits, k=3) == 1.0


def test_ndcg_at_3_rank_3():
    y_true = [2]
    y_logits = [np.array([10, 5, 1])]
    expected = (1.0 / np.log2(4)) / (1.0 / np.log2(2))
    assert abs(ndcg_at_k(y_true, y_logits, k=3) - expected) < 1e-6
