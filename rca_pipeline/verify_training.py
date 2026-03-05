"""
Sanity-check script for the Spatio-Temporal RCA training pipeline.

This script validates gradient integrity, learning capacity, and mask
stability before committing to large-scale experiments.

Three checks are performed:
    1. Single-batch overfitting (100 epochs) -- proves capacity.
    2. Gradient flow inspection (epoch 1 and 50) -- proves
       reparameterization does not block backpropagation.
    3. Mask collapse detection -- proves the GIB mask remains in a
       non-degenerate regime during aggressive beta annealing.

Verified equation:
    L_total = L_InfoNCE + beta * L_GIB
    where beta is annealed from 0.0 to 1.0 over 50 epochs.
"""

import torch
import torch.optim as optim
import logging
import sys
from typing import List, Dict, Tuple

from torch_geometric.data import Data

from model import SpatioTemporalRCA
from losses import info_nce_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> List[Data]:
    """Loads the windowed PyTorch Geometric dataset from disk."""
    try:
        dataset: List[Data] = torch.load(filepath, weights_only=False)
        logger.info("Loaded %d windowed graph samples from %s", len(dataset), filepath)
        return dataset
    except Exception as e:
        logger.error("Failed to load dataset from %s: %s", filepath, e)
        sys.exit(1)


def extract_mini_batch(
    dataset: List[Data],
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts a tiny contrastive batch from the dataset for overfitting.

    Split logic (mirrors dataset_builder.py protocol):
        - First half  = healthy (baseline)
        - Second half = anomalous (chaos)

    Anchors   = healthy[0 .. batch_size-1]
    Positives = healthy[1 .. batch_size]
    Negatives = anomalous[0 .. batch_size-1]

    Returns:
        x_anchor   : [batch_size, num_nodes, num_features, W]
        x_positive : [batch_size, num_nodes, num_features, W]
        x_negative : [batch_size, num_nodes, num_features, W]
        edge_index : [2, num_edges]
    """
    midpoint = len(dataset) // 2
    healthy = dataset[:midpoint]
    anomalous = dataset[midpoint:]

    if len(healthy) < batch_size + 1:
        logger.error(
            "Not enough healthy windows (%d) to form a batch of %d anchor/positive pairs.",
            len(healthy),
            batch_size,
        )
        sys.exit(1)

    if len(anomalous) < batch_size:
        logger.error(
            "Not enough anomalous windows (%d) to form %d negatives.",
            len(anomalous),
            batch_size,
        )
        sys.exit(1)

    anchors = torch.stack([healthy[i].x for i in range(batch_size)]).to(device)
    positives = torch.stack([healthy[i + 1].x for i in range(batch_size)]).to(device)
    negatives = torch.stack([anomalous[i].x for i in range(batch_size)]).to(device)
    edge_index = dataset[0].edge_index.to(device)

    logger.info("Mini-batch extracted:")
    logger.info("  Anchors   : %s", anchors.shape)
    logger.info("  Positives : %s", positives.shape)
    logger.info("  Negatives : %s", negatives.shape)
    logger.info("  Edge index: %s", edge_index.shape)

    return anchors, positives, negatives, edge_index


# ---------------------------------------------------------------------------
# Gradient flow inspection
# ---------------------------------------------------------------------------

def inspect_gradient_flow(model: SpatioTemporalRCA, epoch: int) -> None:
    """
    Computes and logs the L2 norm of gradients for the three major
    architectural components:
        - Temporal convolution (Conv1d weights)
        - Spatial convolution  (GCNConv weights)
        - GIB mask generator   (MLP weights)

    Raises a CRITICAL alert if any gradient is None or exactly zero,
    which would indicate that the relaxed Bernoulli reparameterization
    is blocking backpropagation through the information bottleneck.
    """
    component_map: Dict[str, List[str]] = {
        "temporal_block": [],
        "spatial_block": [],
        "gib": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for component_key in component_map:
            if name.startswith(component_key):
                component_map[component_key].append(name)
                break

    logger.info("--- Gradient Flow Inspection at Epoch %d ---", epoch)

    for component_name, param_names in component_map.items():
        total_norm: float = 0.0
        has_none_grad: bool = False

        for pname in param_names:
            param = dict(model.named_parameters())[pname]
            if param.grad is None:
                has_none_grad = True
                logger.critical(
                    "GRADIENT IS NONE for parameter '%s' in component '%s' at epoch %d. "
                    "The reparameterization trick may be broken.",
                    pname,
                    component_name,
                    epoch,
                )
            else:
                total_norm += param.grad.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5

        if not has_none_grad and total_norm == 0.0:
            logger.critical(
                "GRADIENT NORM IS EXACTLY 0.0 for component '%s' at epoch %d. "
                "Backpropagation is not flowing through this component.",
                component_name,
                epoch,
            )
        else:
            logger.info(
                "  Component %-20s | Gradient L2 Norm: %.6f",
                component_name,
                total_norm,
            )


# ---------------------------------------------------------------------------
# Mask collapse check
# ---------------------------------------------------------------------------

def check_mask_collapse(mask_probs: torch.Tensor, epoch: int) -> None:
    """
    Detects degenerate mask behavior (collapse to all-0 or all-1).

    A healthy GIB mask should have a mean probability strictly between
    0.0 and 1.0. Exact boundary values indicate that the bottleneck
    has collapsed and is no longer performing meaningful compression
    or has degenerated into a pass-through identity.

    Args:
        mask_probs: shape [batch, num_nodes, 1]
        epoch: current epoch number
    """
    mean_prob = mask_probs.mean().item()
    min_prob = mask_probs.min().item()
    max_prob = mask_probs.max().item()

    logger.info(
        "  Mask stats at epoch %03d | Mean: %.4f | Min: %.4f | Max: %.4f",
        epoch,
        mean_prob,
        min_prob,
        max_prob,
    )

    if mean_prob <= 1e-6:
        logger.warning(
            "MASK COLLAPSE DETECTED at epoch %d: mean probability is %.6f (near 0.0). "
            "The GIB is suppressing ALL nodes. Consider reducing beta or beta_max.",
            epoch,
            mean_prob,
        )
    elif mean_prob >= 1.0 - 1e-6:
        logger.warning(
            "MASK COLLAPSE DETECTED at epoch %d: mean probability is %.6f (near 1.0). "
            "The GIB is passing ALL nodes. The bottleneck is not compressing.",
            epoch,
            mean_prob,
        )


# ---------------------------------------------------------------------------
# Main sanity check procedure
# ---------------------------------------------------------------------------

def run_sanity_check(
    dataset_path: str = "dataset_windows.pt",
    num_epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    beta_max: float = 1.0,
    annealing_epochs: int = 50,
) -> None:
    """
    Executes the three sanity checks:
        1. Single-batch overfitting for 100 epochs.
        2. Gradient flow at epochs 1 and 50.
        3. Mask collapse monitoring every 10 epochs under aggressive
           beta annealing (0 -> 1 over 50 epochs).

    Verified equation:
        L_total = L_InfoNCE + beta * L_GIB
        beta(t) = min(1.0, t / annealing_epochs) * beta_max
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -- Load and extract --
    dataset = load_dataset(dataset_path)
    x_anchor, x_positive, x_negative, edge_index = extract_mini_batch(
        dataset, batch_size=batch_size, device=device
    )

    num_features = x_anchor.size(2)

    # -- Model and optimizer --
    model = SpatioTemporalRCA(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_nodes=5,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info("=" * 70)
    logger.info("SANITY CHECK: Overfitting on a single mini-batch of %d samples", batch_size)
    logger.info("  Epochs: %d | LR: %.1e | Hidden: %d", num_epochs, lr, hidden_dim)
    logger.info("  Beta annealing: 0.0 -> %.2f over %d epochs", beta_max, annealing_epochs)
    logger.info("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # -- Beta annealing schedule --
        # beta(t) = min(1.0, t / annealing_epochs) * beta_max
        beta = min(1.0, epoch / max(annealing_epochs, 1)) * beta_max

        # -- Forward pass --
        # z shape: [batch, num_nodes, hidden_dim]
        # probs shape: [batch, num_nodes, 1]
        z_a, probs_a, _ = model(x_anchor, edge_index)
        z_p, _, _ = model(x_positive, edge_index)
        z_n, _, _ = model(x_negative, edge_index)

        # -- Graph-level aggregation (mean pooling) --
        # shape: [batch, hidden_dim]
        z_a_graph = z_a.mean(dim=1)
        z_p_graph = z_p.mean(dim=1)
        z_n_graph = z_n.mean(dim=1)

        # -- Compute losses --
        # L_total = L_InfoNCE + beta * L_GIB
        l_nce = info_nce_loss(z_a_graph, z_p_graph, z_n_graph, temperature=0.07)
        l_gib = probs_a.mean()  # L1 sparsity
        loss = l_nce + beta * l_gib

        # -- Backward pass --
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # -- Gradient flow inspection at epoch 1 and epoch 50 --
        if epoch == 0 or epoch == 49:
            inspect_gradient_flow(model, epoch + 1)

        optimizer.step()

        # -- Logging every epoch --
        logger.info(
            "Epoch %03d/%03d | L_total: %.4f | L_InfoNCE: %.4f | L_GIB: %.4f | Beta: %.4f",
            epoch + 1,
            num_epochs,
            loss.item(),
            l_nce.item(),
            l_gib.item(),
            beta,
        )

        # -- Mask collapse check every 10 epochs --
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                check_mask_collapse(probs_a, epoch + 1)

    # -- Final summary --
    logger.info("=" * 70)
    logger.info("SANITY CHECK COMPLETE")
    logger.info("  Final L_InfoNCE : %.6f", l_nce.item())
    logger.info("  Final L_GIB     : %.6f", l_gib.item())
    logger.info("  Final Beta      : %.4f", beta)
    logger.info(
        "  If L_InfoNCE dropped significantly, the model has learning capacity."
    )
    logger.info(
        "  If no CRITICAL gradient alerts appeared, reparameterization is intact."
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity check for the Spatio-Temporal RCA training pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_windows.pt",
        help="Path to the .pt dataset file.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--annealing-epochs", type=int, default=50)
    args = parser.parse_args()

    run_sanity_check(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        beta_max=args.beta_max,
        annealing_epochs=args.annealing_epochs,
    )
