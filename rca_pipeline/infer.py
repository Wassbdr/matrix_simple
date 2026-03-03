"""
RCA Inference Script.

Loads a trained SpatioTemporalRCA checkpoint, passes an anomalous
window through the model, and identifies the root cause node based
on the Graph Information Bottleneck mask probabilities.

The node with the highest mask activation is reported as the most
likely root cause -- the GIB has learned to isolate the causal
sub-graph during training.
"""

import torch
import argparse
import logging
from typing import List, Dict, Tuple

from torch_geometric.data import Data
from model import SpatioTemporalRCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NODE_NAMES: Dict[int, str] = {
    0: "frontend",
    1: "auth",
    2: "backend",
    3: "cache",
    4: "database",
}


def load_model(checkpoint_path: str, hidden_dim: int = 32) -> SpatioTemporalRCA:
    """Loads a trained model from a checkpoint file."""
    model = SpatioTemporalRCA(num_features=5, hidden_dim=hidden_dim, num_nodes=5)
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (epoch %d)",
            checkpoint_path,
            checkpoint.get("epoch", -1),
        )
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        raise
    model.eval()
    return model


def load_windows(dataset_path: str) -> List[Data]:
    """Loads all windowed Data objects from a .pt file."""
    try:
        windows: List[Data] = torch.load(dataset_path, weights_only=False)
        logger.info("Loaded %d windows from %s", len(windows), dataset_path)
        return windows
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise


def identify_root_cause(
    model: SpatioTemporalRCA,
    window: Data,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Runs inference on a single window and returns the identified root cause.

    The GIB mask probabilities indicate which nodes the bottleneck
    considers causally relevant. During training, the mask was forced
    to be sparse -- so high-probability nodes in an anomalous window
    are the ones the model identifies as root cause.

    Args:
        model: Trained SpatioTemporalRCA (in eval mode).
        window: A single Data object with x shape [num_nodes, num_features, W].

    Returns:
        (root_cause_name, root_cause_probability, all_node_probabilities)
    """
    with torch.no_grad():
        # Add batch dimension: [1, num_nodes, num_features, W]
        x = window.x.unsqueeze(0)
        edge_index = window.edge_index

        # Forward pass
        # z: [1, num_nodes, hidden_dim], probs: [1, num_nodes, 1]
        _, probs = model(x, edge_index)

        # Remove batch dim: [num_nodes, 1] -> [num_nodes]
        node_probs = probs.squeeze(0).squeeze(-1)

    # Build probability map
    prob_map: Dict[str, float] = {}
    for idx, name in NODE_NAMES.items():
        prob_map[name] = round(float(node_probs[idx]), 4)

    # Root cause = node with highest mask probability
    root_idx = int(torch.argmax(node_probs))
    root_name = NODE_NAMES[root_idx]
    root_prob = float(node_probs[root_idx])

    return root_name, root_prob, prob_map


def run_inference(
    checkpoint_path: str,
    dataset_path: str,
    hidden_dim: int = 32,
    num_windows: int = 5,
) -> None:
    """
    Loads model and dataset, then runs inference on the last N windows
    (which correspond to the anomaly period).

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        dataset_path: Path to the windowed dataset .pt file.
        hidden_dim: Must match the training hidden_dim.
        num_windows: Number of anomalous windows to analyze.
    """
    model = load_model(checkpoint_path, hidden_dim)
    windows = load_windows(dataset_path)

    # Take the last N windows (anomaly region)
    anomaly_windows = windows[-num_windows:]
    logger.info("Analyzing the last %d windows (anomaly region)...", len(anomaly_windows))

    logger.info("=" * 60)
    for i, window in enumerate(anomaly_windows):
        root_name, root_prob, prob_map = identify_root_cause(model, window)
        logger.info("Window %d/%d:", i + 1, len(anomaly_windows))
        logger.info("  Mask probabilities: %s", prob_map)
        logger.info(
            "  --> Root cause: %s (probability: %.4f)",
            root_name,
            root_prob,
        )
    logger.info("=" * 60)

    # Majority vote across all analyzed windows
    vote_count: Dict[str, int] = {}
    for window in anomaly_windows:
        name, _, _ = identify_root_cause(model, window)
        vote_count[name] = vote_count.get(name, 0) + 1

    final_root_cause = max(vote_count, key=lambda k: vote_count[k])
    logger.info(
        "FINAL VERDICT (majority vote over %d windows): Root cause = %s (votes: %s)",
        len(anomaly_windows),
        final_root_cause,
        vote_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCA Inference -- Identify root cause nodes.")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/stgcn_rca_final.pt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset_windows.pt",
        help="Path to the windowed dataset .pt file.",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--num-windows", type=int, default=5,
        help="Number of anomalous windows to analyze.",
    )
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        hidden_dim=args.hidden_dim,
        num_windows=args.num_windows,
    )
