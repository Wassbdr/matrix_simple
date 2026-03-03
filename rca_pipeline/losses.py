"""
Loss functions for the Spatio-Temporal RCA model.

InfoNCE (Noise Contrastive Estimation):
    Pulls healthy (anchor, positive) graph embeddings together while
    pushing anomalous (negative) embeddings apart in representation space.

GIB Sparsity:
    Penalizes the Graph Information Bottleneck mask probabilities to
    encourage sparse causal sub-graph selection.
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

    Healthy graph windows (anchor, positive) are pulled together, while
    anomalous windows (negatives) are repelled. The loss is computed as
    the negative log-likelihood of the positive pair among all candidates.

    Args:
        z_anchor   : shape [batch, embedding_dim]
                     Graph-level embeddings of anchor (healthy baseline) windows.
        z_positive : shape [batch, embedding_dim]
                     Graph-level embeddings of positive (healthy shifted) windows.
        z_negatives: shape [num_negatives, embedding_dim]
                     Graph-level embeddings of negative (chaos-injected) windows.
        temperature: Scalar temperature for the softmax. Lower values
                     sharpen the distribution.

    Returns:
        Scalar InfoNCE loss.
    """
    # L2-normalize all embeddings to the unit hypersphere
    # shape: [batch, embedding_dim]
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_positive = F.normalize(z_positive, dim=-1)
    # shape: [num_negatives, embedding_dim]
    z_negatives = F.normalize(z_negatives, dim=-1)

    # Positive similarity: dot product between anchor and positive
    # shape: [batch, 1]
    pos_sim = torch.sum(z_anchor * z_positive, dim=-1, keepdim=True) / temperature

    # Negative similarity: dot product between each anchor and all negatives
    # shape: [batch, num_negatives]
    neg_sim = torch.mm(z_anchor, z_negatives.t()) / temperature

    # Concatenate: the positive is always at index 0
    # shape: [batch, 1 + num_negatives]
    logits = torch.cat([pos_sim, neg_sim], dim=-1)

    # Target: the positive is always at position 0
    # shape: [batch]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Standard cross-entropy over the softmax distribution
    loss = F.cross_entropy(logits, labels)

    return loss


def gib_sparsity_loss(mask_probabilities: torch.Tensor) -> torch.Tensor:
    """
    Graph Information Bottleneck sparsity regularizer.

    Encourages the node mask M to be as sparse as possible by penalizing
    its mean activation. This is equivalent to a relaxed L1 norm on the
    sigmoid probabilities and can also be interpreted as a KL divergence
    between the predicted Bernoulli distribution and a sparse prior
    Bernoulli(epsilon ~ 0).

    Formally:
        L_sparse = KL( Bernoulli(p) || Bernoulli(epsilon) )
                 ~ p * log(p / eps) + (1-p) * log((1-p) / (1-eps))

    For simplicity and numerical stability, we use the mean of the
    probabilities as a differentiable L1 proxy.

    Args:
        mask_probabilities: shape [batch, num_nodes, 1]
                            Sigmoid probabilities from the GIB module.

    Returns:
        Scalar sparsity loss.
    """
    # shape: scalar
    # Mean activation across batch and nodes penalizes dense masks
    loss = mask_probabilities.mean()

    return loss


def total_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negatives: torch.Tensor,
    mask_probs: torch.Tensor,
    beta: float = 0.0,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Combined objective: L_total = L_InfoNCE + beta * L_GIB.

    The beta coefficient controls the information bottleneck pressure.
    During warm-up, beta=0 lets the contrastive loss structure the
    latent space first. Then beta is annealed upward to activate
    the causal sub-graph compression.

    Args:
        z_anchor        : shape [batch, embedding_dim]
        z_positive      : shape [batch, embedding_dim]
        z_negatives     : shape [num_negatives, embedding_dim]
        mask_probs      : shape [batch, num_nodes, 1]
        beta            : GIB coefficient (annealed during training)
        temperature     : InfoNCE temperature

    Returns:
        Scalar total loss.
    """
    l_contrastive = info_nce_loss(z_anchor, z_positive, z_negatives, temperature)
    l_sparsity = gib_sparsity_loss(mask_probs)

    l_total = l_contrastive + beta * l_sparsity

    return l_total
