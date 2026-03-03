"""
Training loop for the Spatio-Temporal RCA model with Graph Information
Bottleneck and InfoNCE contrastive learning.

Workflow:
    1. Load windowed PyTorch Geometric Data objects from disk.
    2. Split into healthy (baseline) and anomalous (chaos) windows.
    3. Construct contrastive triplets: (anchor, positive, negative).
    4. Train the SpatioTemporalRCA model with beta-annealing warm-up.
"""

import torch
import torch.optim as optim
from torch_geometric.data import Data
from typing import List, Tuple
import logging
import argparse
import os

from model import SpatioTemporalRCA
from losses import total_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> List[Data]:
    """Loads a list of Data objects from a PyTorch file."""
    logger.info("Loading dataset from %s", filepath)
    try:
        dataset: List[Data] = torch.load(filepath, weights_only=False)
        logger.info("Loaded %d windowed graph samples.", len(dataset))
        return dataset
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise


def split_healthy_anomalous(
    dataset: List[Data],
    healthy_ratio: float = 0.5,
) -> Tuple[List[Data], List[Data]]:
    """
    Splits the dataset into healthy (first half = baseline period) and
    anomalous (second half = chaos injection period) windows.

    This corresponds to the dataset_builder.py protocol:
        - Minutes 0-5  : healthy baseline
        - Minutes 5-10 : chaos injection

    With 10s steps and a window of W, the midpoint is at index ~30.
    We use a configurable ratio for flexibility.

    Args:
        dataset: Full list of Data windows in chronological order.
        healthy_ratio: Fraction of the dataset considered healthy.

    Returns:
        (healthy_windows, anomalous_windows)
    """
    split_idx = int(len(dataset) * healthy_ratio)
    healthy = dataset[:split_idx]
    anomalous = dataset[split_idx:]
    logger.info(
        "Split dataset: %d healthy windows, %d anomalous windows.",
        len(healthy),
        len(anomalous),
    )
    return healthy, anomalous


def build_contrastive_batch(
    healthy: List[Data],
    anomalous: List[Data],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs contrastive triplets from the dataset.

    Anchors   : healthy windows at index i
    Positives : healthy windows at index i+1 (temporally adjacent)
    Negatives : all anomalous windows

    Returns:
        x_anchor   : shape [batch, num_nodes, num_features, W]
        x_positive : shape [batch, num_nodes, num_features, W]
        x_negative : shape [num_neg, num_nodes, num_features, W]
        edge_index : shape [2, num_edges]
    """
    # Anchors and positives are consecutive healthy pairs
    anchors_x = []
    positives_x = []
    for i in range(len(healthy) - 1):
        anchors_x.append(healthy[i].x)
        positives_x.append(healthy[i + 1].x)

    negatives_x = [w.x for w in anomalous]

    # shape: [batch, num_nodes, num_features, W]
    x_anchor = torch.stack(anchors_x).to(device)
    x_positive = torch.stack(positives_x).to(device)

    if negatives_x:
        x_negative = torch.stack(negatives_x).to(device)
    else:
        # Fallback: if no anomalous windows, use a zero tensor
        x_negative = torch.zeros_like(x_anchor[:1]).to(device)
        logger.warning("No anomalous windows found. Using zero tensor as negative.")

    # Edge index is static and shared across all samples
    edge_index = healthy[0].edge_index.to(device)

    return x_anchor, x_positive, x_negative, edge_index


def graph_level_embedding(z: torch.Tensor) -> torch.Tensor:
    """
    Aggregates node-level embeddings into a single graph-level vector
    via mean pooling.

    Args:
        z: shape [batch, num_nodes, hidden_dim]
    Returns:
        shape [batch, hidden_dim]
    """
    # shape: [batch, hidden_dim]
    return z.mean(dim=1)


# ---------------------------------------------------------------------------
# Beta annealing
# ---------------------------------------------------------------------------

def compute_beta(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """
    Linear warm-up schedule for the GIB coefficient beta.

    During the first `warmup_epochs`, beta increases linearly from 0
    to beta_max. This allows the contrastive loss to first structure
    the latent space before the information bottleneck compresses it.

    Args:
        epoch: Current epoch index (0-based).
        warmup_epochs: Number of epochs for the warm-up phase.
        beta_max: Maximum beta value after warm-up.

    Returns:
        Current beta value.
    """
    if warmup_epochs <= 0:
        return beta_max
    beta = min(1.0, epoch / warmup_epochs) * beta_max
    return beta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 32,
    beta_max: float = 1.0,
    warmup_epochs: int = 20,
    temperature: float = 0.07,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Main training procedure.

    Args:
        dataset_path: Path to the .pt file containing windowed Data objects.
        num_epochs: Total number of training epochs.
        lr: Learning rate for AdamW.
        weight_decay: L2 regularization for AdamW.
        hidden_dim: Hidden dimension of the STGCN encoder.
        beta_max: Maximum GIB coefficient after warm-up.
        warmup_epochs: Number of epochs to linearly ramp beta from 0 to beta_max.
        temperature: InfoNCE temperature parameter.
        checkpoint_dir: Directory to save model checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # -- Load and split dataset --
    dataset = load_dataset(dataset_path)
    healthy, anomalous = split_healthy_anomalous(dataset)

    if len(healthy) < 2:
        logger.error("Need at least 2 healthy windows to form anchor/positive pairs.")
        return

    # -- Build contrastive batch --
    x_anchor, x_positive, x_negative, edge_index = build_contrastive_batch(
        healthy, anomalous, device
    )

    logger.info("Anchor batch shape  : %s", x_anchor.shape)
    logger.info("Positive batch shape: %s", x_positive.shape)
    logger.info("Negative batch shape: %s", x_negative.shape)

    # Infer dimensions from the data
    num_features = x_anchor.size(2)

    # -- Initialize model and optimizer --
    model = SpatioTemporalRCA(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_nodes=5,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # -- Training loop --
    logger.info(
        "Starting training: epochs=%d, lr=%.1e, beta_max=%.2f, warmup=%d",
        num_epochs,
        lr,
        beta_max,
        warmup_epochs,
    )

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        beta = compute_beta(epoch, warmup_epochs, beta_max)

        # Forward pass on anchors
        # shape: z_a [batch, num_nodes, hidden_dim], probs_a [batch, num_nodes, 1]
        z_a, probs_a = model(x_anchor, edge_index)

        # Forward pass on positives
        # shape: z_p [batch, num_nodes, hidden_dim], _ (probs not used)
        z_p, _ = model(x_positive, edge_index)

        # Forward pass on negatives
        # shape: z_n [num_neg, num_nodes, hidden_dim], _ (probs not used)
        z_n, _ = model(x_negative, edge_index)

        # Aggregate to graph-level embeddings
        # shape: [batch, hidden_dim], [batch, hidden_dim], [num_neg, hidden_dim]
        z_a_graph = graph_level_embedding(z_a)
        z_p_graph = graph_level_embedding(z_p)
        z_n_graph = graph_level_embedding(z_n)

        # Compute combined loss
        loss = total_loss(
            z_anchor=z_a_graph,
            z_positive=z_p_graph,
            z_negatives=z_n_graph,
            mask_probs=probs_a,
            beta=beta,
            temperature=temperature,
        )

        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            mask_density = probs_a.mean().item()
            logger.info(
                "Epoch %04d/%04d | Loss: %.4f | Beta: %.4f | Mask density: %.4f",
                epoch,
                num_epochs,
                loss.item(),
                beta,
                mask_density,
            )

    # -- Save final checkpoint --
    checkpoint_path = os.path.join(checkpoint_dir, "stgcn_rca_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "beta_max": beta_max,
        },
        checkpoint_path,
    )
    logger.info("Final checkpoint saved to %s", checkpoint_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal RCA model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_windows.pt",
        help="Path to the .pt dataset file.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        temperature=args.temperature,
    )
