"""
Contrastive data augmentation for Spatio-Temporal RCA training.

Provides differentiable and non-differentiable augmentations to
generate more diverse positive pairs for the InfoNCE loss, improving
the robustness of the learned latent space.

Augmentation strategies:
    1. Gaussian noise injection on node features
    2. Temporal jittering (random shift within the window)
    3. Feature masking (randomly zero out feature channels)
    4. Edge dropout (temporarily remove edges for spatial diversity)
"""

import torch
import logging
from typing import Optional
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Adds isotropic Gaussian noise to node features.

    This is the simplest augmentation for contrastive learning.
    It produces a slightly perturbed version of the input that should
    still be recognized as a positive (same system state).

    Args:
        x: shape [batch, num_nodes, num_features, W] or [num_nodes, num_features, W]
        sigma: Standard deviation of the noise.

    Returns:
        Noisy copy of x with the same shape.
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


def temporal_jitter(
    x: torch.Tensor,
    max_shift: int = 1,
) -> torch.Tensor:
    """
    Applies a random temporal shift along the last dimension with zero-padding.

    Unlike torch.roll (circular), this implementation pads shifted positions
    with zeros to preserve temporal causality — future timesteps never wrap
    to the beginning of the window.

    Args:
        x: shape [batch, num_nodes, num_features, W] or [num_nodes, num_features, W]
        max_shift: Maximum number of timesteps to shift.

    Returns:
        Temporally shifted copy of x (zero-padded, not circular).
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x.clone()
    out = torch.zeros_like(x)
    if shift > 0:
        # Shift right: early timesteps become zero
        out[..., shift:] = x[..., :-shift]
    else:
        # Shift left: late timesteps become zero
        out[..., :shift] = x[..., -shift:]
    return out


def feature_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Randomly zeros out a fraction of the feature channels.

    Forces the model to learn redundant representations across
    features rather than relying on a single metric (e.g. CPU only).

    Args:
        x: shape [batch, num_nodes, num_features, W] or [num_nodes, num_features, W]
        mask_ratio: Fraction of features to mask out.

    Returns:
        Feature-masked copy of x.
    """
    x_aug = x.clone()
    num_features = x.size(-2)  # num_features dimension
    num_to_mask = max(1, int(num_features * mask_ratio))
    mask_indices = torch.randperm(num_features)[:num_to_mask]
    # Zero out selected feature channels across all nodes and timesteps
    x_aug[..., mask_indices, :] = 0.0
    return x_aug


def augment_positive(
    x: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    noise_sigma: float = 0.1,
    jitter_shift: int = 1,
    feature_mask_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Applies a composition of augmentations to generate a positive sample.

    The augmented sample should be semantically equivalent to the original
    (same system state) but structurally different enough to prevent
    the model from learning trivial shortcuts.

    Composition order:
        1. Gaussian noise
        2. Temporal jitter
        3. Feature masking

    Args:
        x: shape [batch, num_nodes, num_features, W]
        edge_index: Optional, shape [2, num_edges] (unused, kept for API compat)
        noise_sigma: Gaussian noise standard deviation.
        jitter_shift: Max temporal shift in timesteps.
        feature_mask_ratio: Fraction of features to mask.

    Returns:
        Augmented x tensor.
    """
    x_aug = add_gaussian_noise(x, sigma=noise_sigma)
    x_aug = temporal_jitter(x_aug, max_shift=jitter_shift)
    x_aug = feature_masking(x_aug, mask_ratio=feature_mask_ratio)
    return x_aug
