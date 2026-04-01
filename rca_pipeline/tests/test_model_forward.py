"""Forward-pass smoke tests for key RCA models."""

import torch

from rca_pipeline.model_late_gat import SpatioTemporalRCA_GAT
from rca_pipeline.model_sta import SelectThenAnalyze


def test_sta_forward_shapes():
    bsz, num_nodes, num_features, window, k = 2, 20, 5, 12, 5
    model = SelectThenAnalyze(
        num_features=num_features, hidden_dim=32, num_nodes=num_nodes, k=k
    )
    x = torch.randn(bsz, num_nodes, num_features, window)
    edge_index = torch.randint(0, num_nodes, (2, 40))

    logits, m_soft, top_k, sel = model(x, edge_index)
    assert logits.shape == (bsz, num_nodes)
    assert m_soft.shape == (bsz, num_nodes)
    assert top_k.shape == (bsz, k)
    assert sel.shape == (bsz, num_nodes)


def test_gat_forward_shapes():
    bsz, num_nodes, num_features, window = 2, 20, 5, 12
    model = SpatioTemporalRCA_GAT(
        num_features=num_features, hidden_dim=32, num_nodes=num_nodes
    )
    x = torch.randn(bsz, num_nodes, num_features, window)
    edge_index = torch.randint(0, num_nodes, (2, 40))

    logits, imp, h = model(x, edge_index)
    assert logits.shape == (bsz, num_nodes)
    assert imp.shape == (bsz, num_nodes)
    assert h.shape == (bsz, num_nodes, 32)
