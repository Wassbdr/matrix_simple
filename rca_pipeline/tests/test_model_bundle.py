"""Round-trip and inference-contract tests for the serving model bundle."""

import numpy as np
import pytest
import torch

from rca_pipeline.model_bundle import load_bundle, predict, save_bundle
from rca_pipeline.model_sta import SelectThenAnalyze

NODE_NAMES = ["svc-a", "svc-b", "svc-c", "svc-d"]
N, F, W, H, K = len(NODE_NAMES), 2, 8, 8, 2
EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


def _make_bundle(tmp_path, scaler_scale=None):
    torch.manual_seed(0)
    model = SelectThenAnalyze(num_features=F, hidden_dim=H, num_nodes=N, k=K)
    if scaler_scale is None:
        scaler_scale = np.ones(F, dtype=np.float32)
    save_bundle(
        str(tmp_path),
        model=model,
        node_names=NODE_NAMES,
        edge_index=EDGE_INDEX,
        scaler_mean=np.zeros(F, dtype=np.float32),
        scaler_scale=scaler_scale,
        num_features=F,
        hidden_dim=H,
        k=K,
        training_info={"avg1_mean": 0.5, "seeds": [42]},
        hpo_provenance={"source": "test"},
    )
    return model


def test_save_load_round_trip(tmp_path):
    model = _make_bundle(tmp_path)
    bundle = load_bundle(str(tmp_path))

    assert bundle.node_names == NODE_NAMES
    assert torch.equal(bundle.edge_index, EDGE_INDEX)
    assert bundle.metadata["config"] == {
        "num_features": F, "hidden_dim": H, "num_nodes": N, "k": K,
    }
    for key_src, key_dst in zip(model.state_dict(), bundle.model.state_dict()):
        assert key_src == key_dst
    for p_src, p_dst in zip(model.state_dict().values(), bundle.model.state_dict().values()):
        assert torch.equal(p_src, p_dst)


def test_metadata_provenance_fields(tmp_path):
    _make_bundle(tmp_path)
    bundle = load_bundle(str(tmp_path))
    card = bundle.metadata["model_card"]
    for field in ("created_at_utc", "git_commit", "python_version", "torch_version", "platform"):
        assert field in card
    assert bundle.metadata["hpo_provenance"] == {"source": "test"}
    assert bundle.metadata["training"]["seeds"] == [42]
    # run_id is omitted when not provided (offline training)
    assert "mlflow_run_id" not in card


def test_mlflow_run_id_cross_link(tmp_path):
    torch.manual_seed(0)
    model = SelectThenAnalyze(num_features=F, hidden_dim=H, num_nodes=N, k=K)
    save_bundle(
        str(tmp_path), model=model, node_names=NODE_NAMES, edge_index=EDGE_INDEX,
        scaler_mean=np.zeros(F, dtype=np.float32),
        scaler_scale=np.ones(F, dtype=np.float32),
        num_features=F, hidden_dim=H, k=K,
        training_info={"avg1_mean": 0.5}, mlflow_run_id="abc123def456",
    )
    bundle = load_bundle(str(tmp_path))
    assert bundle.metadata["model_card"]["mlflow_run_id"] == "abc123def456"


def test_predict_ranking_shape_and_order(tmp_path):
    _make_bundle(tmp_path)
    bundle = load_bundle(str(tmp_path))
    window = np.random.default_rng(1).normal(size=(N, F, W)).astype(np.float32)

    ranked = predict(bundle, window, top_k=3)
    assert len(ranked) == 3
    services = [svc for svc, _ in ranked]
    scores = [score for _, score in ranked]
    assert all(svc in NODE_NAMES for svc in services)
    assert len(set(services)) == 3
    assert scores == sorted(scores, reverse=True)


def test_predict_is_deterministic(tmp_path):
    _make_bundle(tmp_path)
    bundle = load_bundle(str(tmp_path))
    window = np.random.default_rng(2).normal(size=(N, F, W)).astype(np.float32)
    assert predict(bundle, window, top_k=N) == predict(bundle, window, top_k=N)


def test_predict_zero_scale_guard(tmp_path):
    _make_bundle(tmp_path, scaler_scale=np.zeros(F, dtype=np.float32))
    bundle = load_bundle(str(tmp_path))
    window = np.random.default_rng(3).normal(size=(N, F, W)).astype(np.float32)

    ranked = predict(bundle, window, top_k=1)
    assert np.isfinite(ranked[0][1])


def test_predict_shape_mismatch_raises(tmp_path):
    _make_bundle(tmp_path)
    bundle = load_bundle(str(tmp_path))

    with pytest.raises(ValueError, match="N="):
        predict(bundle, np.zeros((N + 1, F, W), dtype=np.float32))
    with pytest.raises(ValueError, match="features"):
        predict(bundle, np.zeros((N, F + 1, W), dtype=np.float32))
