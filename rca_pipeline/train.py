"""
Training loop for the Spatio-Temporal RCA model with Graph Information
Bottleneck and InfoNCE contrastive learning.

Two-phase training:
    Phase 1 — Contrastive pre-training (InfoNCE + GIB sparsity):
        Learns to separate healthy vs anomalous embeddings.
    Phase 2 — Supervised fine-tuning (root cause classification):
        Trains the model to predict which node is the root cause
        using labeled anomalous windows.
"""

import torch
import torch.optim as optim
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import logging
import argparse
import os

from model import SpatioTemporalRCA
from losses import total_loss, root_cause_loss
from augment import augment_positive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}


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


def split_by_labels(
    dataset: List[Data],
) -> Tuple[List[Data], List[Data]]:
    """
    Splits the dataset into healthy and anomalous windows based on actual
    labels (y==0 for healthy, y==1 for anomalous), NOT by position.

    Returns:
        (healthy_windows, anomalous_windows)
    """
    healthy = [d for d in dataset if d.y is not None and d.y.item() == 0]
    anomalous = [d for d in dataset if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in NAME_TO_IDX]
    logger.info(
        "Split dataset by labels: %d healthy windows, %d anomalous windows (with valid root cause).",
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
    supervised_epochs: int = 100,
    supervised_lr: float = 5e-4,
) -> None:
    """
    Two-phase training procedure.

    Phase 1: Contrastive pre-training (InfoNCE + GIB sparsity)
    Phase 2: Supervised fine-tuning on anomalous windows for root cause prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # -- Load and split dataset by actual labels --
    dataset = load_dataset(dataset_path)
    healthy, anomalous = split_by_labels(dataset)

    if len(healthy) < 2:
        logger.error("Need at least 2 healthy windows to form anchor/positive pairs.")
        return

    if len(anomalous) < 1:
        logger.error("Need at least 1 anomalous window with root cause label.")
        return

    # Log root cause distribution
    rc_dist: Dict[str, int] = {}
    for d in anomalous:
        rc = d.root_cause[0]
        rc_dist[rc] = rc_dist.get(rc, 0) + 1
    logger.info("Root cause distribution in anomalous windows: %s", rc_dist)

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

    # ===================================================================
    # Phase 1: Contrastive pre-training
    # ===================================================================
    logger.info(
        "=== PHASE 1: Contrastive pre-training (%d epochs) ===",
        num_epochs,
    )

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        beta = compute_beta(epoch, warmup_epochs, beta_max)

        # Forward pass on anchors
        z_a, probs_a = model(x_anchor, edge_index)

        # Apply Contrastive Data Augmentation to positives dynamically
        x_positive_aug = augment_positive(
            x_positive, 
            noise_sigma=0.05, 
            jitter_shift=1, 
            feature_mask_ratio=0.1
        )
        # Forward pass on augmented positives
        z_p, _ = model(x_positive_aug, edge_index)

        # Forward pass on negatives
        z_n, _ = model(x_negative, edge_index)

        # Aggregate to graph-level embeddings
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            mask_density = probs_a.mean().item()
            logger.info(
                "[Phase1] Epoch %04d/%04d | Loss: %.4f | Beta: %.4f | Mask density: %.4f",
                epoch,
                num_epochs,
                loss.item(),
                beta,
                mask_density,
            )

    # ===================================================================
    # Phase 2: Supervised fine-tuning for root cause classification
    # ===================================================================
    logger.info(
        "=== PHASE 2: Supervised fine-tuning (%d epochs) ===",
        supervised_epochs,
    )

    # Build supervised batch from anomalous windows
    sup_x = torch.stack([d.x for d in anomalous]).to(device)  # [N, V, F, W]
    sup_targets = torch.tensor(
        [NAME_TO_IDX[d.root_cause[0]] for d in anomalous],
        dtype=torch.long,
    ).to(device)  # [N]

    logger.info("Supervised batch shape: %s, targets shape: %s", sup_x.shape, sup_targets.shape)

    # Use a separate optimizer for fine-tuning with lower LR
    ft_optimizer = optim.AdamW(model.parameters(), lr=supervised_lr, weight_decay=weight_decay)

    for epoch in range(supervised_epochs):
        model.train()
        ft_optimizer.zero_grad()

        # Forward pass
        z, probs = model(sup_x, edge_index)

        # Root cause classification
        rc_logits = model.classify_root_cause(z)  # [N, num_nodes]

        # Combined loss: classification + sparsity (to keep mask sparse)
        cls_loss = root_cause_loss(rc_logits, sup_targets)
        sparsity = probs.mean()
        loss = cls_loss + 0.1 * sparsity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ft_optimizer.step()

        if epoch % 20 == 0 or epoch == supervised_epochs - 1:
            # Compute accuracy
            preds = rc_logits.argmax(dim=1)
            acc = (preds == sup_targets).float().mean().item()
            logger.info(
                "[Phase2] Epoch %04d/%04d | Loss: %.4f | Accuracy: %.4f | Mask density: %.4f",
                epoch,
                supervised_epochs,
                loss.item(),
                acc,
                probs.mean().item(),
            )

    # -- Save final checkpoint --
    checkpoint_path = os.path.join(checkpoint_dir, "stgcn_rca_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": ft_optimizer.state_dict(),
            "epoch": num_epochs + supervised_epochs,
            "beta_max": beta_max,
            "num_features": num_features,
            "hidden_dim": hidden_dim,
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
    parser.add_argument("--epochs", type=int, default=100, help="Contrastive pre-training epochs")
    parser.add_argument("--supervised-epochs", type=int, default=150, help="Supervised fine-tuning epochs")
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
        supervised_epochs=args.supervised_epochs,
    )
