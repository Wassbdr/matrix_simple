"""
Fusion helpers for STA selector/scorer calibration.
"""


import numpy as np
import torch


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    return x.std(unbiased=False).clamp(min=1e-6)


def gated_selector_scorer_fusion(
    selector_logits: torch.Tensor,
    scorer_logits: torch.Tensor,
    top_k_indices: torch.Tensor,
    gamma: float = 0.35,
) -> torch.Tensor:
    """
    Fuse selector and scorer only on selected top-k nodes.

    - Non selected nodes keep selector logits unchanged.
    - Selected nodes are fused via convex blending:
        (1-gamma) * selector + gamma * scorer_rescaled
      where scorer is calibrated to selector scale on the top-k subset.
    """
    combined = selector_logits.clone()
    idx = top_k_indices.long().view(-1)
    if idx.numel() == 0:
        return combined

    sel_k = selector_logits[idx]
    scorer_k = scorer_logits[idx]
    finite = torch.isfinite(scorer_k)
    if not finite.any():
        return combined

    scorer_k = scorer_k.clone()
    fallback = sel_k.mean()
    scorer_k[~finite] = fallback

    scorer_norm = (scorer_k - scorer_k.mean()) / _safe_std(scorer_k)
    scorer_rescaled = scorer_norm * _safe_std(sel_k) + sel_k.mean()

    g = max(0.0, min(1.0, float(gamma)))
    combined[idx] = (1.0 - g) * sel_k + g * scorer_rescaled
    return combined


def apply_gray_prior(
    selector_logits: torch.Tensor,
    x_window: np.ndarray,
    lambda_gray: float = 1.0,
) -> torch.Tensor:
    """
    Apply gray failure detection scores as a Bayesian prior on selector logits.

    P(M_i = 1 | gray_score) ∝ sigmoid(logit_i) * (1 + lambda * gray_score_i)

    In log-space (logits): logit_i += lambda * gray_score_i

    Args:
        selector_logits: [N] or [B, N] raw selector logits
        x_window: [N, F, W] numpy array for gray failure detection
        lambda_gray: strength of the gray prior
    Returns:
        adjusted logits with gray prior applied
    """
    try:
        from gray_failure_detector import detect_gray_failure
    except ImportError:
        return selector_logits

    result = detect_gray_failure(x_window)

    if not result.is_gray or not result.affected_nodes:
        return selector_logits

    gray_scores = torch.zeros(x_window.shape[0], device=selector_logits.device)
    for node_idx in result.affected_nodes:
        if node_idx < gray_scores.size(0):
            gray_scores[node_idx] = result.confidence

    if selector_logits.dim() == 2:
        gray_scores = gray_scores.unsqueeze(0).expand_as(selector_logits)

    return selector_logits + lambda_gray * gray_scores
