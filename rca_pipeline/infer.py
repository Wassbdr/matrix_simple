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

from common import NODE_NAMES as _DEFAULT_NODE_NAMES, NAME_TO_IDX as _DEFAULT_NAME_TO_IDX, load_dataset_payload

# Default int->str mapping for 5-node K8s topology (overridden at runtime by dataset payload)
_DEFAULT_NODE_IDX_NAMES: Dict[int, str] = {i: name for i, name in enumerate(_DEFAULT_NODE_NAMES)}


def load_model(checkpoint_path: str, hidden_dim: int = 32) -> Tuple[SpatioTemporalRCA, bool]:
    """Loads a trained model from a checkpoint file.
    Returns the model and the use_gib flag stored at training time."""
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        # Read dimensions from checkpoint
        num_features = checkpoint.get('num_features',
            checkpoint['model_state_dict']['temporal_block.conv1.weight'].size(1))
        hd = checkpoint.get('hidden_dim', hidden_dim)
        num_nodes = checkpoint.get('num_nodes', 5)
        use_gib = checkpoint.get('use_gib', True)
        model = SpatioTemporalRCA(num_features=num_features, hidden_dim=hd, num_nodes=num_nodes)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (epoch %d, features=%d, hidden=%d, use_gib=%s)",
            checkpoint_path,
            checkpoint.get("epoch", -1),
            num_features,
            hd,
            use_gib,
        )
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e)
        raise
    model.eval()
    return model, use_gib


def load_windows(dataset_path: str) -> Tuple[List[Data], Dict[int, str]]:
    """Loads all windowed Data objects plus node name mapping from a .pt file."""
    try:
        windows, node_names = load_dataset_payload(dataset_path)
        node_idx_names: Dict[int, str] = {i: name for i, name in enumerate(node_names)}
        logger.info("Loaded %d windows from %s (%d nodes: %s)",
                    len(windows), dataset_path, len(node_idx_names), list(node_idx_names.values()))
        return windows, node_idx_names
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise


def identify_root_cause(
    model: SpatioTemporalRCA,
    window: Data,
    use_gib: bool = True,
    node_idx_names: Dict[int, str] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Runs inference on a single window and returns the identified root cause.

    Uses the GIB raw logits (use_gib=True) or the NodeScorer head (use_gib=False)
    depending on how the model was trained — matching the training objective.

    Args:
        model: Trained SpatioTemporalRCA (in eval mode).
        window: A single Data object with x shape [num_nodes, num_features, W].
        use_gib: Whether to use the GIB logit head (must match checkpoint).
        node_idx_names: int→str mapping; defaults to 5-node K8s topology.

    Returns:
        (root_cause_name, root_cause_score, all_node_scores)
    """
    names = node_idx_names if node_idx_names is not None else _DEFAULT_NODE_IDX_NAMES
    with torch.no_grad():
        # Add batch dimension: [1, num_nodes, num_features, W]
        x = window.x.unsqueeze(0)
        edge_index = window.edge_index

        # Forward pass
        _, _, h_spatial = model(x, edge_index)

        if use_gib:
            logits = model.gib_logits(h_spatial)  # [1, num_nodes]
        else:
            logits = model.classify_root_cause(h_spatial)  # [1, num_nodes]

        # Shape [num_nodes]
        node_scores = logits.squeeze(0)

    # Build score map
    score_map: Dict[str, float] = {}
    for idx, name in names.items():
        score_map[name] = round(float(node_scores[idx]), 4)

    # Root cause = node with highest score
    root_idx = int(torch.argmax(node_scores))
    root_name = names[root_idx]
    root_score = float(node_scores[root_idx])

    return root_name, root_score, score_map


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
    model, use_gib = load_model(checkpoint_path, hidden_dim)
    windows, node_idx_names = load_windows(dataset_path)

    # Select anomalous windows explicitly by label (y==1)
    anomaly_windows = [w for w in windows if w.y is not None and w.y.item() == 1]
    if not anomaly_windows:
        logger.warning("No windows with y==1 found; falling back to the last %d windows.", num_windows)
        anomaly_windows = windows[-num_windows:]
    else:
        anomaly_windows = anomaly_windows[-num_windows:]
    logger.info("Analyzing %d anomalous windows...", len(anomaly_windows))

    logger.info("=" * 60)
    for i, window in enumerate(anomaly_windows):
        root_name, root_score, score_map = identify_root_cause(model, window, use_gib, node_idx_names)
        logger.info("Window %d/%d:", i + 1, len(anomaly_windows))
        logger.info("  Scores: %s", score_map)
        logger.info(
            "  --> Root cause: %s (score: %.4f)",
            root_name,
            root_score,
        )
    logger.info("=" * 60)

    # Majority vote across all analyzed windows
    vote_count: Dict[str, int] = {}
    for window in anomaly_windows:
        name, _, _ = identify_root_cause(model, window, use_gib, node_idx_names)
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
