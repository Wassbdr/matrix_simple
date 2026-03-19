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
    min_active_nodes: int = 1,
    dispersion_weight: float = 0.25,
) -> torch.Tensor:
    """
    Asymmetric GIB loss with optional multi-root support:
        - Healthy windows (U=0): mask should be near zero → no root cause
        - Anomalous windows: mask should be sparse and optionally spread over
          at least a small set of nodes (for combinatorial failures)

    Scales properly to any N:
        - target_sparsity should be set to 1/N by the caller.
        - Concentration weight is scaled by 1/log(N) to prevent it from
          dominating at large N where max_entropy grows.

    Args:
        probs_healthy:   shape [batch_h, num_nodes, 1] — mask probs for healthy windows
        probs_anomalous: shape [batch_a, num_nodes, 1] — mask probs for anomalous windows
        target_sparsity: desired mean activation for anomalous masks (e.g. 1/N)
        min_active_nodes: minimum number of active nodes encouraged in anomalous masks.
            Keep at 1 for single-root behavior; set >1 to relax one-root assumption.
        dispersion_weight: weight of anti-collapse penalty on anomalous masks.

    Returns:
        Scalar asymmetric GIB loss.
    """
    # --- Healthy: push all mask probabilities toward 0 ---
    loss_healthy = (probs_healthy ** 2).mean()

    # --- Anomalous: sparse, optionally multi-node distribution ---
    p_anom = probs_anomalous.squeeze(-1)  # [batch, num_nodes]
    N = p_anom.size(-1)

    # 1. Sparsity: mean activation should be close to target_sparsity
    mean_activation = p_anom.mean(dim=-1)  # [batch]
    loss_sparsity = ((mean_activation - target_sparsity) ** 2).mean()

    # 2. Anti-collapse term:
    #    Encourage anomalous masks not to collapse onto a single node when
    #    min_active_nodes > 1. Uses normalized entropy floor.
    p_norm = p_anom / (p_anom.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p_norm * torch.log(p_norm + 1e-8)).sum(dim=-1)  # [batch]

    max_entropy = torch.log(torch.tensor(float(max(N, 2)), device=p_anom.device))
    entropy_norm = entropy / max_entropy

    target_nodes = max(1, min(min_active_nodes, N))
    target_entropy = torch.log(torch.tensor(float(target_nodes), device=p_anom.device)) / max_entropy
    loss_dispersion = F.relu(target_entropy - entropy_norm).mean()

    # 3. Binary sharpness keeps probabilities near {0,1} without forcing one-hot.
    loss_sharpness = (p_anom * (1.0 - p_anom)).mean()

    loss_anomalous = loss_sparsity + dispersion_weight * loss_dispersion + 0.1 * loss_sharpness

    return loss_healthy + loss_anomalous


def root_cause_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for root cause classification.
    """
    return F.cross_entropy(logits, targets)


# =============================================================================
# SOTA: Multi-Root and Recurrence Losses
# =============================================================================


def multi_root_cause_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_gamma: float = 2.0,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """
    Unified loss for single-root and multi-root cause analysis.

    Two modes:
    1. Multi-hot targets [B, N]: BCE with focal weighting for hard examples
    2. Index targets [B]: Standard cross-entropy for single root cause

    Focal Loss helps with class imbalance by down-weighting easy examples.

    Args:
        logits: [B, N] raw scores (not probabilities)
        targets: [B, N] multi-hot or [B] indices
        focal_gamma: Focal loss exponent (0 = standard BCE, 2 = strong focal)
        pos_weight: Weight for positive class (handles imbalance)

    Returns:
        Scalar loss
    """
    B, N = logits.shape

    # Detect multi-root mode
    is_multi_root = targets.dim() == 2 and targets.size(1) == N

    if is_multi_root:
        # Multi-label: Focal BCE
        probs = torch.sigmoid(logits)
        targets_float = targets.float()

        # Focal weight: reduce weight for easy (well-classified) examples
        pt = torch.where(targets_float == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** focal_gamma

        # BCE loss with pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_float,
            reduction="none",
            pos_weight=torch.tensor([pos_weight], device=logits.device),
        )

        # Apply focal weighting
        loss = (focal_weight * bce).mean()

    else:
        # Single-label: Cross-entropy
        if targets.dim() == 2:
            # Convert multi-hot to index (take first positive)
            targets = targets.argmax(dim=1)
        loss = F.cross_entropy(logits, targets)

    return loss


def recurrence_classification_loss(
    recurrence_logits: torch.Tensor,
    is_recurrent: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary loss for novel vs recurrent classification.

    This helps the model learn to distinguish patterns it has seen before
    from completely new failure modes.

    Args:
        recurrence_logits: [B, 2] logits for [novel, recurrent]
        is_recurrent: [B] boolean tensor (True = recurrent pattern)

    Returns:
        Scalar cross-entropy loss
    """
    targets = is_recurrent.long()
    return F.cross_entropy(recurrence_logits, targets)


def memory_regularization_loss(
    memory_attn: torch.Tensor,
    gate_value: torch.Tensor,
    entropy_weight: float = 0.1,
) -> torch.Tensor:
    """
    Regularization for memory module.

    Encourages:
    1. Diverse memory slot usage (entropy on attention)
    2. Appropriate gating (not always 0 or 1)

    Args:
        memory_attn: [B, M] attention weights over memory slots
        gate_value: [B, 1] gate values
        entropy_weight: Weight for attention entropy term

    Returns:
        Scalar regularization loss
    """
    # Entropy of attention (encourage diversity)
    attn_entropy = -(memory_attn * torch.log(memory_attn + 1e-8)).sum(dim=-1)
    # Negative because we want to maximize entropy (diverse usage)
    loss_entropy = -attn_entropy.mean()

    # Gate variance (encourage meaningful gating, not always 0 or 1)
    loss_gate = -gate_value.var()

    return entropy_weight * loss_entropy + 0.01 * loss_gate


def combined_rca_loss(
    outputs: dict,
    targets: torch.Tensor,
    is_recurrent: Optional[torch.Tensor] = None,
    is_anomaly: Optional[torch.Tensor] = None,
    lambda_recurrence: float = 0.1,
    lambda_sparsity: float = 0.1,
    lambda_memory: float = 0.01,
) -> tuple:
    """
    Combined loss for memory-augmented RCA model.

    Components:
    1. Main RCA loss (multi-root or single-root)
    2. Sparsity loss on mask (continued regularization)
    3. Recurrence classification loss (if labels available)
    4. Memory regularization loss (if memory outputs available)

    Args:
        outputs: Dict from MemoryAugmentedRCA with:
            - logits_full: [B, N] root cause logits
            - M_soft: [B, N] selection mask
            - recurrence_logits: [B, 2] (optional)
            - memory_attn: [B, M] (optional)
            - memory_gate: [B, 1] (optional)
        targets: [B] or [B, N] root cause targets
        is_recurrent: [B] boolean (optional, for recurrence loss)
        is_anomaly: [B] boolean (optional, to filter healthy samples)
        lambda_recurrence: Weight for recurrence classification loss
        lambda_sparsity: Weight for mask sparsity loss
        lambda_memory: Weight for memory regularization

    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    device = outputs["logits_full"].device
    components = {}

    # Filter to anomaly samples only for most losses
    if is_anomaly is not None:
        anom_mask = is_anomaly.bool()
        logits = outputs["logits_full"][anom_mask]
        targets_filtered = targets[anom_mask] if targets.dim() == 1 else targets[anom_mask]
    else:
        logits = outputs["logits_full"]
        targets_filtered = targets

    # 1. Main RCA loss
    if logits.numel() > 0:
        loss_rca = multi_root_cause_loss(logits, targets_filtered)
    else:
        loss_rca = torch.tensor(0.0, device=device)
    components["loss_rca"] = loss_rca.item()

    # 2. Sparsity loss on mask
    loss_sparsity = outputs["M_soft"].abs().sum(dim=-1).mean()
    components["loss_sparsity"] = loss_sparsity.item()

    # 3. Recurrence classification loss
    loss_recurrence = torch.tensor(0.0, device=device)
    if (
        is_recurrent is not None
        and "recurrence_logits" in outputs
        and outputs["recurrence_logits"] is not None
    ):
        # Filter to anomaly samples
        if is_anomaly is not None:
            rec_logits = outputs["recurrence_logits"][anom_mask]
            rec_targets = is_recurrent[anom_mask]
        else:
            rec_logits = outputs["recurrence_logits"]
            rec_targets = is_recurrent

        if rec_logits.numel() > 0:
            loss_recurrence = recurrence_classification_loss(rec_logits, rec_targets)
    components["loss_recurrence"] = loss_recurrence.item()

    # 4. Memory regularization loss
    loss_memory = torch.tensor(0.0, device=device)
    if "memory_attn" in outputs and "memory_gate" in outputs:
        if outputs["memory_attn"] is not None and outputs["memory_gate"] is not None:
            loss_memory = memory_regularization_loss(
                outputs["memory_attn"],
                outputs["memory_gate"],
            )
    components["loss_memory"] = loss_memory.item()

    # Total loss
    total_loss = (
        loss_rca
        + lambda_sparsity * loss_sparsity
        + lambda_recurrence * loss_recurrence
        + lambda_memory * loss_memory
    )
    components["total"] = total_loss.item()

    return total_loss, components


