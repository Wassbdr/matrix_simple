"""Regression tests for the GIB warmup collapse (README §5.4, fixed in A3).

With k == N the GIB target rate is 1.0: the KL term against Bernoulli(1-eps)
explodes for any mask probability < 1 and saturates the selector. These
tests pin the hazard so the k>=N skip in benchmark_real never regresses.
"""

import torch

from rca_pipeline.losses import variational_gib_loss


def test_target_rate_one_stays_finite_and_explodes_vs_sane_target():
    # Before the float32-safe clamp, target_rate=1.0 gave loss=inf and NaN
    # gradients (1.0 - 1e-8 == 1.0 in float32, so the clamp was a no-op).
    m = torch.full((8, 5), 0.5)
    sane = variational_gib_loss(m, target_rate=0.5)
    degenerate = variational_gib_loss(m, target_rate=1.0)
    assert torch.isfinite(degenerate)
    assert degenerate.item() > 100 * max(sane.item(), 1e-6)
    assert degenerate.item() > 5.0


def test_target_rate_one_gradient_finite_and_saturates_masks_upward():
    logits = torch.zeros(4, 5, requires_grad=True)
    m = torch.sigmoid(logits)
    loss = variational_gib_loss(m, target_rate=1.0)
    loss.backward()
    # Finite (no more NaN weights after one optimizer step) but huge and
    # uniformly negative: pushes every mask probability toward 1, which is
    # why the GIB term must be skipped entirely when k >= N.
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad < 0).all()
    assert logits.grad.abs().mean().item() > 0.1


def test_sane_target_rate_keeps_moderate_gradients():
    logits = torch.zeros(4, 5, requires_grad=True)
    m = torch.sigmoid(logits)
    loss = variational_gib_loss(m, target_rate=0.4)
    loss.backward()
    assert logits.grad.abs().mean().item() < 0.1
