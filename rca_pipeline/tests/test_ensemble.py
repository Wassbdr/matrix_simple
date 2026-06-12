"""Tests for post-hoc ensembling and calibration (rca_pipeline/ensemble.py)."""

import numpy as np
import pytest

from rca_pipeline.ensemble import (
    confidence_weighted_vote,
    ensemble_runs,
    expected_calibration_error,
    fit_temperature,
    soft_vote,
    softmax,
    temperature_scale,
)


def test_softmax_is_a_distribution():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    assert np.argmax(p) == 2


def test_soft_vote_averages_probabilities():
    za = np.array([10.0, 0.0, 0.0])   # model A certain of node 0
    zb = np.array([0.0, 10.0, 0.0])   # model B certain of node 1
    p = soft_vote(za, zb)
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(p[0], p[1])      # equal weights → tie between 0 and 1
    assert p[2] < p[0]


def test_soft_vote_weighting():
    za = np.array([10.0, 0.0])
    zb = np.array([0.0, 10.0])
    assert np.argmax(soft_vote(za, zb, weight_a=0.9)) == 0
    assert np.argmax(soft_vote(za, zb, weight_a=0.1)) == 1
    with pytest.raises(ValueError):
        soft_vote(za, zb, weight_a=1.5)


def test_soft_vote_scale_invariance_motivation():
    # Same ranking but wildly different logit scales: probability-domain
    # averaging keeps the confident model from drowning the other.
    za = np.array([100.0, 0.0, 0.0])
    zb = np.array([0.0, 0.1, 0.0])
    p = soft_vote(za, zb)
    assert np.argmax(p) == 0
    assert p[1] > p[2]  # model B's mild preference survives


def test_confidence_weighted_vote_favors_confident_model():
    za = np.array([10.0, 0.0])       # very confident → weight ≈ 1
    zb = np.array([0.0, 0.1])        # nearly uniform → weight ≈ 0.5
    p = confidence_weighted_vote(za, zb)
    assert np.argmax(p) == 0


def test_ensemble_runs_pairs_by_matching_y_true():
    za = [np.array([5.0, 0.0]), np.array([0.0, 5.0])]
    zb = [np.array([4.0, 0.0]), np.array([0.0, 4.0])]
    runs_a = [([0, 1], za)]
    runs_b = [([0, 1], zb)]
    out = ensemble_runs(runs_a, runs_b)
    assert len(out) == 1
    y_true, probs = out[0]
    assert y_true == [0, 1]
    assert all(np.isclose(p.sum(), 1.0) for p in probs)

    # Mismatched splits are skipped, not silently mixed.
    runs_b_bad = [([1, 0], zb)]
    assert ensemble_runs(runs_a, runs_b_bad) == []


def test_ensemble_runs_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ensemble_runs([], [], mode="hard")


def test_temperature_scaling_softens_confidence():
    z = np.array([4.0, 0.0, 0.0])
    p_hot = softmax(temperature_scale(z, 0.5))
    p_cold = softmax(temperature_scale(z, 4.0))
    assert p_hot[0] > softmax(z)[0] > p_cold[0]
    with pytest.raises(ValueError):
        temperature_scale(z, 0.0)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    # Overconfident model: correct only 60% of the time but logit gap is huge.
    logits, y = [], []
    for _ in range(200):
        true = int(rng.integers(0, 4))
        pred = true if rng.random() < 0.6 else int(rng.integers(0, 4))
        z = np.zeros(4)
        z[pred] = 12.0
        logits.append(z)
        y.append(true)
    t = fit_temperature(logits, y)
    assert t > 1.0  # softening required
    ece_before = expected_calibration_error(logits, y)
    ece_after = expected_calibration_error([z / t for z in logits], y)
    assert ece_after < ece_before


def test_ece_zero_for_perfectly_calibrated_certainty():
    logits = [np.array([50.0, 0.0]), np.array([0.0, 50.0])]
    y = [0, 1]
    assert expected_calibration_error(logits, y) < 1e-6
    assert expected_calibration_error([], []) == 0.0
