"""Determinism and column-order guards for the Sock Shop demo (STA bundle vs BARO)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import RobustScaler

from rca_pipeline.baseline_baro import baro_scores_from_arrays
from rca_pipeline.demo_incident import (
    BUNDLE_DIR,
    FEATURES,
    INCIDENT_JSON,
    run_demo,
    sta_full_ranking,
)
from rca_pipeline.model_bundle import align_window, load_bundle
from rca_pipeline.simulated_data_scalable import (
    generate_scalable_dataset,
    topology_from_service_graph,
)
from rca_pipeline.topologies import SOCK_SHOP

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / BUNDLE_DIR
INCIDENT = ROOT / INCIDENT_JSON


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(str(BUNDLE))


@pytest.fixture(scope="module")
def incident():
    return json.loads(INCIDENT.read_text())


def test_bundle_and_incident_share_service_and_feature_order(bundle, incident):
    assert bundle.node_names == SOCK_SHOP.node_names() == incident["services"]
    assert bundle.metadata["feature_names"] == FEATURES == incident["features"]
    assert np.asarray(incident["window"]).shape == (14, len(FEATURES), 12)


def test_two_runs_give_same_topk():
    a = run_demo(str(BUNDLE), str(INCIDENT), repeats=1)
    b = run_demo(str(BUNDLE), str(INCIDENT), repeats=1)
    for method in ("STA", "BARO"):
        assert a[method]["ranking"] == b[method]["ranking"]
        assert a[method]["top3"] == b[method]["top3"]
        assert a[method]["true_rank"] == b[method]["true_rank"]


def test_sta_and_baro_rank_the_same_candidates():
    res = run_demo(str(BUNDLE), str(INCIDENT), repeats=1)
    assert (
        sorted(res["STA"]["ranking"]) == sorted(res["BARO"]["ranking"]) == sorted(res["candidates"])
    )


def test_shuffled_services_and_features_are_realigned(bundle, incident):
    window = np.asarray(incident["window"], dtype=np.float32)
    ref = sta_full_ranking(
        bundle, align_window(bundle, window, incident["services"], incident["features"])
    )
    rng = np.random.RandomState(0)
    s_perm = rng.permutation(len(incident["services"]))
    f_perm = rng.permutation(len(incident["features"]))
    shuffled = window[s_perm][:, f_perm]
    aligned = align_window(
        bundle,
        shuffled,
        services=[incident["services"][i] for i in s_perm],
        features=[incident["features"][i] for i in f_perm],
    )
    np.testing.assert_array_equal(aligned, window)
    assert sta_full_ranking(bundle, aligned) == ref


def test_feature_order_matters_without_names(bundle, incident):
    """Guards the guard: swapping columns silently changes the model's input."""
    window = np.asarray(incident["window"], dtype=np.float32)
    swapped = window[:, ::-1].copy()
    out_ref = bundle.model(torch.from_numpy(window)[None], bundle.edge_index)[3]
    out_sw = bundle.model(torch.from_numpy(swapped)[None], bundle.edge_index)[3]
    assert not torch.allclose(out_ref, out_sw)


@pytest.mark.parametrize(
    "services,features",
    [
        (["front-end"] * 14, None),
        (None, ["cpu", "memory", "latency_p99", "error_rate", "rps"]),
        (None, ["cpu", "cpu", "latency_p99", "error_rate", "throughput"]),
    ],
)
def test_align_window_rejects_mismatched_names(bundle, incident, services, features):
    window = np.asarray(incident["window"], dtype=np.float32)
    with pytest.raises(ValueError):
        align_window(bundle, window, services=services, features=features)


def test_baro_arrays_matches_sklearn_robustscaler():
    rng = np.random.RandomState(1)
    normal = rng.randn(4, 3, 40)
    normal[2, 1] = 5.0  # IQR == 0 column
    anomal = rng.randn(4, 3, 12) + np.array([0, 3, 0, 1])[:, None, None]
    expected = []
    for i in range(4):
        best = -np.inf
        for f in range(3):
            sc = RobustScaler().fit(normal[i, f].reshape(-1, 1))
            best = max(best, float(sc.transform(anomal[i, f].reshape(-1, 1)).max()))
        expected.append(best)
    np.testing.assert_allclose(baro_scores_from_arrays(normal, anomal), expected, rtol=1e-5)


def test_service_graph_generator_is_seeded():
    a, names, _ = generate_scalable_dataset(
        topology=SOCK_SHOP, num_samples=30, seed=3, generate_multimodal=False
    )
    b, _, _ = generate_scalable_dataset(
        topology=SOCK_SHOP, num_samples=30, seed=3, generate_multimodal=False
    )
    assert names == SOCK_SHOP.node_names()
    assert all(torch.equal(x.x, y.x) and x.root_cause == y.root_cause for x, y in zip(a, b))
    _, ei, P, G = topology_from_service_graph(SOCK_SHOP)
    assert G.number_of_edges() == len(SOCK_SHOP.edges)
    assert P[names.index("carts-db"), names.index("carts")] == 1.0  # callee fault reaches caller
