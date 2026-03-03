"""
Evaluation script for the RCA model.

Computes Precision, Recall, and F1-score for root cause identification
by comparing the model's predicted root cause node against the known
ground truth (the pod targeted by the Chaos Mesh experiment).

Usage:
    python3 evaluate.py \
        --checkpoint checkpoints/stgcn_rca_final.pt \
        --dataset dataset_windows.pt \
        --ground-truth cache
"""

import torch
import argparse
import logging
from typing import List, Dict, Tuple

from torch_geometric.data import Data
from model import SpatioTemporalRCA
from infer import load_model, load_windows, identify_root_cause, NODE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    """
    Computes per-class Precision, Recall, and F1 for the root cause
    identification task.

    In the RCA setting:
        - True Positive  = model predicts the correct root cause node
        - False Positive = model predicts a wrong node as root cause
        - False Negative = model fails to predict the correct root cause

    Args:
        predictions: List of predicted root cause node names.
        ground_truth: The actual root cause node name.

    Returns:
        Dictionary with precision, recall, f1, and accuracy.
    """
    if not predictions:
        logger.warning("No predictions to evaluate.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    true_positives = sum(1 for p in predictions if p == ground_truth)
    false_positives = sum(1 for p in predictions if p != ground_truth)
    total = len(predictions)

    # In a single-label multi-class setting with one ground truth:
    # Precision = TP / (TP + FP) = fraction of correct predictions
    # Recall = TP / total_anomalous_windows (all should predict ground_truth)
    precision = true_positives / total if total > 0 else 0.0
    recall = true_positives / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = true_positives / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate(
    checkpoint_path: str,
    dataset_path: str,
    ground_truth: str,
    hidden_dim: int = 32,
    healthy_ratio: float = 0.5,
) -> None:
    """
    Evaluates the model on anomalous windows and reports metrics.

    Args:
        checkpoint_path: Path to the trained checkpoint.
        dataset_path: Path to the windowed .pt dataset.
        ground_truth: Name of the actual root cause node (e.g. 'cache', 'database').
        hidden_dim: Must match the training configuration.
        healthy_ratio: Fraction of windows that are healthy (first half).
    """
    if ground_truth not in NODE_NAMES.values():
        logger.error(
            "Unknown ground truth '%s'. Must be one of: %s",
            ground_truth,
            list(NODE_NAMES.values()),
        )
        return

    model = load_model(checkpoint_path, hidden_dim)
    windows = load_windows(dataset_path)

    # Split: anomalous windows are the second half
    split_idx = int(len(windows) * healthy_ratio)
    anomalous_windows = windows[split_idx:]
    logger.info("Evaluating on %d anomalous windows. Ground truth: %s", len(anomalous_windows), ground_truth)

    # Predict root cause for each anomalous window
    predictions: List[str] = []
    for window in anomalous_windows:
        name, prob, _ = identify_root_cause(model, window)
        predictions.append(name)

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    # Display results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("  Ground truth root cause : %s", ground_truth)
    logger.info("  Total anomalous windows : %d", len(anomalous_windows))
    logger.info("  Precision               : %.4f", metrics["precision"])
    logger.info("  Recall                  : %.4f", metrics["recall"])
    logger.info("  F1-score                : %.4f", metrics["f1"])
    logger.info("  Accuracy                : %.4f", metrics["accuracy"])
    logger.info("=" * 60)

    # Distribution of predictions
    vote_count: Dict[str, int] = {}
    for p in predictions:
        vote_count[p] = vote_count.get(p, 0) + 1
    logger.info("  Prediction distribution : %s", vote_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RCA model against ground truth.")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/stgcn_rca_final.pt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset_windows.pt",
        help="Path to the windowed dataset .pt file.",
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Name of the actual root cause node (frontend, auth, backend, cache, database).",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        ground_truth=args.ground_truth,
        hidden_dim=args.hidden_dim,
    )
