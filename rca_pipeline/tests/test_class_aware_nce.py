"""Tests for class-aware InfoNCE (rca_pipeline/losses.py)."""

import torch

from rca_pipeline.losses import class_aware_info_nce_loss, info_nce_loss


def _clustered_embeddings(n_per_class=4, dim=8, noise=0.05, seed=0):
    """Two well-separated classes; same-class members are near-duplicates."""
    g = torch.Generator().manual_seed(seed)
    c0 = torch.zeros(dim); c0[0] = 1.0
    c1 = torch.zeros(dim); c1[1] = 1.0
    z, y = [], []
    for label, center in ((0, c0), (1, c1)):
        for _ in range(n_per_class):
            z.append(center + noise * torch.randn(dim, generator=g))
            y.append(label)
    return torch.stack(z), torch.tensor(y)


def test_class_aware_loss_is_finite_and_positive():
    z, y = _clustered_embeddings()
    z_pos = z + 0.05 * torch.randn_like(z)
    loss = class_aware_info_nce_loss(z, z_pos, y)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_class_aware_does_not_punish_clustered_same_class():
    """On clustered same-class embeddings the legacy loss is large (it treats
    same-class near-duplicates as negatives); the class-aware loss is small."""
    z, y = _clustered_embeddings()
    z_pos = z + 0.01 * torch.randn_like(z)
    legacy = info_nce_loss(z, z_pos, z)  # all samples as negatives, self included
    aware = class_aware_info_nce_loss(z, z_pos, y)
    assert aware.item() < legacy.item()


def test_single_class_batch_degenerates_gracefully():
    z, _ = _clustered_embeddings(n_per_class=4)
    y = torch.zeros(z.size(0), dtype=torch.long)  # everyone same class
    z_pos = z + 0.01 * torch.randn_like(z)
    loss = class_aware_info_nce_loss(z, z_pos, y)
    # No valid negative anywhere: positive wins by construction, loss ≈ 0.
    assert torch.isfinite(loss)
    assert loss.item() < 1e-3


def test_gradients_flow():
    z, y = _clustered_embeddings()
    z = z.clone().requires_grad_(True)
    z_pos = z.detach() + 0.05 * torch.randn_like(z)
    loss = class_aware_info_nce_loss(z, z_pos, y)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
