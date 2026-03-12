"""
Loss functions for the Spatio-Temporal RCA model.

InfoNCE (Noise Contrastive Estimation):
    Pulls healthy (anchor, positive) graph embeddings together while
    pushing anomalous (negative) embeddings apart in representation space.

GIB Sparsity (Asymmetric):
    For healthy windows: push mask M toward zero (no root cause).
    For anomalous windows: push mask M toward sparse one-hot (single root cause).
    This directly encodes the dynamical system assumption:
        U=0 → M≈0, ||U||_0=1 → M is one-hot.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def info_nce_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Asymmetric InfoNCE contrastive loss.

    Args:
        z_anchor   : shape [batch, embedding_dim]
        z_positive : shape [batch, embedding_dim]
        z_negatives: shape [num_negatives, embedding_dim]
        temperature: Scalar temperature for the softmax.

    Returns:
        Scalar InfoNCE loss.
    """
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_positive = F.normalize(z_positive, dim=-1)
    z_negatives = F.normalize(z_negatives, dim=-1)

    pos_sim = torch.sum(z_anchor * z_positive, dim=-1, keepdim=True) / temperature
    neg_sim = torch.mm(z_anchor, z_negatives.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def asymmetric_gib_loss(
    probs_healthy: torch.Tensor,
    probs_anomalous: torch.Tensor,
    target_sparsity: float = 0.2,
) -> torch.Tensor:
    """
    Asymmetric GIB loss that encodes the dynamical system assumption:
        - Healthy windows (U=0): mask should be near zero → no root cause
        - Anomalous windows (||U||_0 = 1): mask should be concentrated → one node active

    Scales properly to any N:
        - target_sparsity should be set to 1/N by the caller.
        - Concentration weight is scaled by 1/log(N) to prevent it from
          dominating at large N where max_entropy grows.

    Args:
        probs_healthy:   shape [batch_h, num_nodes, 1] — mask probs for healthy windows
        probs_anomalous: shape [batch_a, num_nodes, 1] — mask probs for anomalous windows
        target_sparsity: desired mean activation for anomalous masks (e.g. 1/N)

    Returns:
        Scalar asymmetric GIB loss.
    """
    # --- Healthy: push all mask probabilities toward 0 ---
    loss_healthy = (probs_healthy ** 2).mean()

    # --- Anomalous: push mask toward sparse, concentrated distribution ---
    p_anom = probs_anomalous.squeeze(-1)  # [batch, num_nodes]
    N = p_anom.size(-1)

    # 1. Sparsity: mean activation should be close to target_sparsity
    mean_activation = p_anom.mean(dim=-1)  # [batch]
    loss_sparsity = ((mean_activation - target_sparsity) ** 2).mean()

    # 2. Concentration: negative entropy pushes toward one-hot
    #    Scaled by 1/log(N) so it doesn't dominate at large N
    p_norm = p_anom / (p_anom.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p_norm * torch.log(p_norm + 1e-8)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(N), device=p_anom.device))
    loss_concentration = (entropy / max_entropy).mean() / max(1.0, torch.log(torch.tensor(float(N))).item())

    loss_anomalous = loss_sparsity + loss_concentration

    return loss_healthy + loss_anomalous


def root_cause_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for root cause classification.
    """
    return F.cross_entropy(logits, targets)

