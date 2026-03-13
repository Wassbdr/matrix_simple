"""
Evaluation script for the RCA model.

Computes Precision, Recall, and F1-score for root cause identification
by comparing the model's predicted root cause node against the known
ground truth labels in the dataset.

Uses the supervised root cause classifier head for predictions.

Usage:
    python3 evaluate.py \
        --checkpoint checkpoints/stgcn_rca_final.pt \
        --dataset ../dataset_windows.pt
"""

import torch
import torch.nn.functional as F
import argparse
import logging
from typing import List, Dict, Tuple
from collections import Counter

from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from model import SpatioTemporalRCA

from common import NODE_NAMES, NAME_TO_IDX, stratified_split, load_dataset_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str) -> Tuple[SpatioTemporalRCA, bool]:
    """Loads a trained model from a checkpoint file.
    Returns the model and the use_gib flag stored at training time."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    num_features = checkpoint.get('num_features',
        checkpoint['model_state_dict']['temporal_block.conv1.weight'].size(1))
    hidden_dim = checkpoint.get('hidden_dim',
        checkpoint['model_state_dict']['temporal_block.conv1.weight'].size(0))
    num_nodes = checkpoint.get('num_nodes', 5)
    use_gib = checkpoint.get('use_gib', True)

    model = SpatioTemporalRCA(num_features=num_features, hidden_dim=hidden_dim, num_nodes=num_nodes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s (features=%d, hidden=%d, nodes=%d, use_gib=%s)",
                checkpoint_path, num_features, hidden_dim, num_nodes, use_gib)
    return model, use_gib


def get_anomalies(dataset, name_to_idx):
    """Extract anomalous windows with valid root cause labels."""
    return [d for d in dataset
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and len(d.root_cause) > 0
            and d.root_cause[0] in name_to_idx]


def evaluate(
    checkpoint_path: str,
    dataset_path: str,
    train_ratio: float = 0.8,
) -> None:
    """
    Evaluates the model on anomalous windows using stratified split.
    Reports per-class and weighted metrics.
    """
    model, use_gib = load_model(checkpoint_path)

    logger.info("Loading dataset from %s", dataset_path)
    dataset, node_names = load_dataset_payload(dataset_path)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    node_idx_to_name = {i: name for i, name in enumerate(node_names)}

    anomalies = get_anomalies(dataset, name_to_idx)
    logger.info("Total anomalous windows with valid labels: %d", len(anomalies))

    if not anomalies:
        logger.error("No anomalous windows with root cause labels found.")
        return

    # Stratified split: ensure each root cause type appears in train and test
    by_rc: Dict[str, List[Data]] = {}
    for d in anomalies:
        rc = d.root_cause[0]
        by_rc.setdefault(rc, []).append(d)

    test_anomalies: List[Data] = []
    for rc, items in by_rc.items():
        split = max(1, int(len(items) * train_ratio))
        test_anomalies.extend(items[split:])

    logger.info("Evaluating on %d test anomalous windows.", len(test_anomalies))

    # Show distribution
    test_dist = Counter([d.root_cause[0] for d in test_anomalies])
    logger.info("Test root cause distribution: %s", dict(test_dist))

    # Predict root cause for each test window
    y_true, y_pred = [], []
    edge_index = test_anomalies[0].edge_index

    with torch.no_grad():
        for data in test_anomalies:
            x = data.x.unsqueeze(0)  # [1, V, F, W]
            z, probs, h_spatial = model(x, data.edge_index)
            if use_gib:
                logits = model.gib_logits(h_spatial)  # [1, num_nodes]
            else:
                logits = model.classify_root_cause(h_spatial)  # [1, num_nodes]
            pred_idx = logits.argmax(dim=1).item()

            y_true.append(name_to_idx[data.root_cause[0]])
            y_pred.append(pred_idx)

    # Compute metrics
    p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Display results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("  Total test windows    : %d", len(test_anomalies))
    logger.info("  Precision (weighted)  : %.4f", p)
    logger.info("  Recall    (weighted)  : %.4f", r)
    logger.info("  F1-score  (weighted)  : %.4f", f1)
    logger.info("=" * 60)

    # Per-class report
    target_names = [node_idx_to_name[i] for i in sorted(set(y_true) | set(y_pred))]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    logger.info("\n%s", report)

    # Prediction distribution
    pred_names = [node_idx_to_name[p] for p in y_pred]
    logger.info("Prediction distribution: %s", dict(Counter(pred_names)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RCA model against ground truth.")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/stgcn_rca_final.pt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, default="../dataset_windows.pt",
        help="Path to the windowed dataset .pt file.",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
    )
