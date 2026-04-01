"""Unit tests for sta_fusion_utils (pytest or python)."""
import torch

from rca_pipeline.sta_fusion_utils import gated_selector_scorer_fusion


def test_non_selected_nodes_unchanged():
    sel = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    scorer = torch.tensor([float("-inf"), 3.0, -1.0, float("-inf")], dtype=torch.float32)
    top_k = torch.tensor([1, 2], dtype=torch.long)
    out = gated_selector_scorer_fusion(sel, scorer, top_k, gamma=0.35)
    assert torch.isclose(out[0], sel[0])
    assert torch.isclose(out[3], sel[3])


def test_gamma_zero_returns_selector():
    sel = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    scorer = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    top_k = torch.tensor([0, 1, 2], dtype=torch.long)
    out = gated_selector_scorer_fusion(sel, scorer, top_k, gamma=0.0)
    assert torch.allclose(out, sel)


def test_handles_non_finite_scorer():
    sel = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    scorer = torch.tensor([float("-inf"), float("inf"), 1.0], dtype=torch.float32)
    top_k = torch.tensor([0, 1, 2], dtype=torch.long)
    out = gated_selector_scorer_fusion(sel, scorer, top_k, gamma=0.5)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_non_selected_nodes_unchanged()
    test_gamma_zero_returns_selector()
    test_handles_non_finite_scorer()
    print("ok: test_sta_fusion_utils")
