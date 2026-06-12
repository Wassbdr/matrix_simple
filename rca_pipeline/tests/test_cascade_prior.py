"""Tests for the cascade propagation prior (rca_pipeline/cascade_prior.py)."""

import numpy as np
import pytest
import torch

from rca_pipeline.cascade_prior import (
    apply_cascade_prior,
    node_anomaly_scores,
    propagation_scores,
    undirected_neighbors,
)

# Chain topology: front-end(0) -> orders(1) -> orders-db(2), plus isolated(3).
EDGE_INDEX = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
N, F, W = 4, 2, 8


def _window(anomaly_per_node):
    x = torch.zeros(N, F, W)
    for node, amplitude in enumerate(anomaly_per_node):
        x[node] = amplitude
    return x


def test_node_anomaly_scores_normalized():
    x = _window([0.0, 4.0, 0.4, 0.0])
    a = node_anomaly_scores(x)
    assert a.max() == pytest.approx(1.0)
    assert np.argmax(a) == 1
    assert a[3] == 0.0


def test_undirected_neighbors_union():
    neigh = undirected_neighbors(EDGE_INDEX, N)
    assert neigh[0] == [1]
    assert neigh[1] == [0, 2]
    assert neigh[2] == [1]
    assert neigh[3] == []


def test_propagation_prior_boosts_quiet_node_with_loud_neighbors():
    # Cascade signature: orders(1) screams, orders-db(2) is silent.
    a = np.array([0.5, 1.0, 0.05, 0.0])
    prior = propagation_scores(a, EDGE_INDEX)
    assert np.argmax(prior) == 2          # quiet node next to the loudest one
    assert prior[3] == 0.0                # isolated node untouched
    # The loud symptom node itself gets little prior (1 - anomaly ≈ 0).
    assert prior[1] < prior[2]


def test_cascade_blend_flips_symptom_to_root_cause():
    # Model is confidently wrong: ranks the loud symptom node (1) first.
    model_logits = np.array([0.0, 6.0, 1.0, -2.0])
    x = _window([1.0, 4.0, 0.1, 0.0])     # node 1 loud, node 2 (true RC) quiet
    p0 = apply_cascade_prior(model_logits, x, EDGE_INDEX, gamma=0.0)
    assert np.argmax(p0) == 1
    p = apply_cascade_prior(model_logits, x, EDGE_INDEX, gamma=0.6)
    assert np.argmax(p) == 2              # prior recovers the quiet root cause
    assert p.sum() == pytest.approx(1.0)


def test_hard_failure_unaffected_when_root_is_loud():
    # Hard failure: the root cause (1) is itself the loudest node and the
    # model already ranks it first — the prior must not flip it away.
    model_logits = np.array([0.0, 8.0, 0.0, 0.0])
    x = _window([0.3, 4.0, 0.2, 0.0])
    for gamma in (0.2, 0.4, 0.6):
        p = apply_cascade_prior(model_logits, x, EDGE_INDEX, gamma=gamma)
        assert np.argmax(p) == 1, f"hard-failure flip at gamma={gamma}"


def test_gamma_bounds():
    with pytest.raises(ValueError):
        apply_cascade_prior(np.zeros(N), _window([1, 0, 0, 0]), EDGE_INDEX, gamma=1.5)
