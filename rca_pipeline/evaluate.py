"""
Evaluation script for the RCA model.

Computes Precision, Recall, and F1-score for root cause identification
by comparing the model's predicted root cause node against the known
ground truth labels in the dataset.

SOTA metrics added (V2):
- Top-k Accuracy (k=1, 2, 5)
- Mean Reciprocal Rank (MRR)
- Precision@k
- Breakdown by pattern type and recurrence status

Uses the supervised root cause classifier head for predictions.

Usage:
    python3 evaluate.py \
        --checkpoint checkpoints/stgcn_rca_final.pt \
        --dataset ../dataset_windows.pt

    # With breakdown (V2 datasets)
    python3 evaluate.py --checkpoint checkpoints/sta.pt --dataset scalable_N100_v2.pt --breakdown
"""

import torch
import argparse
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict

from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from model import SpatioTemporalRCA
from model_sta import SelectThenAnalyze

from common import load_dataset_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# SOTA Metrics Class
# =============================================================================


class RCAMetrics:
    """
    SOTA metrics for Root Cause Analysis evaluation.

    Computes:
    - Top-k Accuracy (k=1, 2, 5): Is the true RC in the top-k predictions?
    - Mean Reciprocal Rank (MRR): Average of 1/rank across samples
    - Precision@k: |correct in top-k| / k
    - Per-category breakdown: Metrics split by pattern_type and is_recurrent

    Usage:
        metrics = RCAMetrics(k_values=[1, 2, 5])
        for batch in dataloader:
            logits = model(batch)
            metrics.update(logits, targets, metadata)
        results = metrics.compute_metrics()
        breakdown = metrics.compute_breakdown()
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics tracker.

        Args:
            k_values: List of k values for top-k and P@k (default: [1, 2, 5])
        """
        self.k_values = k_values or [1, 2, 5]
        self.reset()

    def reset(self):
        """Clear accumulated predictions."""
        self.predictions: List[np.ndarray] = []  # Raw logits/scores
        self.targets: List[np.ndarray] = []  # True RC indices
        self.metadata: List[Dict] = []  # pattern_type, is_recurrent, etc.

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """
        Add a batch of predictions.

        Args:
            logits: [B, N] raw scores (higher = more likely RC)
            targets: [B] indices or [B, N] multi-hot
            metadata: List of dicts with 'pattern_type', 'is_recurrent', etc.
        """
        B = logits.size(0)
        logits_np = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # Convert multi-hot targets to indices
        if targets_np.ndim == 2:
            # Multi-hot: get all positive indices
            for i in range(B):
                self.targets.append(np.where(targets_np[i] > 0)[0])
                self.predictions.append(logits_np[i])
        else:
            # Single index
            for i in range(B):
                self.targets.append(np.array([targets_np[i]]))
                self.predictions.append(logits_np[i])

        # Add metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * B)

    def _get_rank(self, pred: np.ndarray, target: int) -> int:
        """
        Get the rank (1-indexed) of target in predictions.

        Args:
            pred: [N] scores
            target: Index of true RC

        Returns:
            Rank (1 = top prediction, 2 = second, etc.)
        """
        sorted_indices = np.argsort(pred)[::-1]  # Descending
        rank = np.where(sorted_indices == target)[0]
        if len(rank) == 0:
            return len(pred) + 1  # Not found
        return int(rank[0]) + 1

    def _get_best_rank(self, pred: np.ndarray, targets: np.ndarray) -> int:
        """Get the best (lowest) rank among all true targets."""
        return min(self._get_rank(pred, t) for t in targets)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all SOTA metrics.

        Returns:
            Dict with:
            - top_1_acc, top_2_acc, top_5_acc
            - mrr (Mean Reciprocal Rank)
            - precision_at_1, precision_at_2, precision_at_5
        """
        if not self.predictions:
            return {}

        results = {}
        n_samples = len(self.predictions)
        N = self.predictions[0].shape[0] if self.predictions else 0

        # Top-k accuracy
        for k in self.k_values:
            k_eff = min(k, N)
            correct = 0
            for pred, target in zip(self.predictions, self.targets):
                top_k_preds = np.argsort(pred)[-k_eff:][::-1]
                if any(t in top_k_preds for t in target):
                    correct += 1
            results[f"top_{k}_acc"] = correct / n_samples

        # MRR and Severity-Weighted MRR
        rr_sum = 0.0
        sw_rr_sum = 0.0
        total_severity = 0.0
        has_severity = False

        for pred, target, meta in zip(self.predictions, self.targets, self.metadata):
            best_rank = self._get_best_rank(pred, target)
            rr = 1.0 / best_rank
            rr_sum += rr

            sev = meta.get("severity", None)
            if sev is not None:
                has_severity = True
                sw_rr_sum += rr * sev
                total_severity += sev

        results["mrr"] = rr_sum / n_samples
        if has_severity and total_severity > 0:
            results["sw_mrr"] = sw_rr_sum / total_severity
        elif has_severity:
            results["sw_mrr"] = 0.0

        # Precision@k
        for k in self.k_values:
            k_eff = min(k, N)
            precision_sum = 0.0
            for pred, target in zip(self.predictions, self.targets):
                top_k_preds = set(np.argsort(pred)[-k_eff:][::-1])
                target_set = set(target)
                hits = len(top_k_preds & target_set)
                precision_sum += hits / k_eff
            results[f"precision_at_{k}"] = precision_sum / n_samples

        return results

    def compute_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics broken down by category.

        Returns:
            Dict with:
            - by_pattern_type: {pattern_name: {mrr, top1, count}}
            - by_recurrence: {novel: {...}, recurrent: {...}}
            - by_difficulty: {hard: {...}, medium: {...}, normal: {...}, easy: {...}}
        """
        if not self.predictions:
            return {}

        # Organize by category
        by_pattern: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        by_recurrence: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "novel": [],
            "recurrent": [],
        }
        by_difficulty: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "hard": [],
            "medium": [],
            "normal": [],
            "easy": [],
        }

        for pred, target, meta in zip(self.predictions, self.targets, self.metadata):
            pattern = meta.get("pattern_type", "unknown")
            is_rec = meta.get("is_recurrent", False)
            difficulty = meta.get("pattern_difficulty", "normal")

            by_pattern[pattern].append((pred, target))
            key = "recurrent" if is_rec else "novel"
            by_recurrence[key].append((pred, target))
            if difficulty in by_difficulty:
                by_difficulty[difficulty].append((pred, target))

        # Compute metrics per category
        result = {"by_pattern_type": {}, "by_recurrence": {}, "by_difficulty": {}}

        # Pattern breakdown
        for pattern, samples in by_pattern.items():
            if not samples:
                continue
            mrr_sum = sum(1.0 / self._get_best_rank(p, t) for p, t in samples)
            top1_sum = sum(
                1 if t[0] == np.argmax(p) else 0 for p, t in samples
            )
            result["by_pattern_type"][pattern] = {
                "mrr": mrr_sum / len(samples),
                "top_1_acc": top1_sum / len(samples),
                "count": len(samples),
            }

        # Recurrence breakdown
        for rec_type, samples in by_recurrence.items():
            if not samples:
                continue
            mrr_sum = sum(1.0 / self._get_best_rank(p, t) for p, t in samples)
            top1_sum = sum(
                1 if t[0] == np.argmax(p) else 0 for p, t in samples
            )
            result["by_recurrence"][rec_type] = {
                "mrr": mrr_sum / len(samples),
                "top_1_acc": top1_sum / len(samples),
                "count": len(samples),
            }

        # Difficulty breakdown
        for diff, samples in by_difficulty.items():
            if not samples:
                continue
            mrr_sum = sum(1.0 / self._get_best_rank(p, t) for p, t in samples)
            top1_sum = sum(
                1 if t[0] == np.argmax(p) else 0 for p, t in samples
            )
            result["by_difficulty"][diff] = {
                "mrr": mrr_sum / len(samples),
                "top_1_acc": top1_sum / len(samples),
                "count": len(samples),
            }

        return result


def evaluate_with_sota_metrics(
    model: torch.nn.Module,
    test_data: List[Data],
    name_to_idx: Dict[str, int],
    device: torch.device,
    batch_size: int = 256,
    mode: str = "select_then_analyze",
    use_gib: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """
    Evaluate model with SOTA metrics including breakdown.

    Args:
        model: Trained RCA model
        test_data: List of test Data objects (anomalies only)
        name_to_idx: Mapping from node name to index
        device: Torch device
        batch_size: Batch size for inference
        mode: Model mode ("select_then_analyze" or "semi_supervised")
        use_gib: Whether to use GIB logits (for semi_supervised mode)

    Returns:
        Tuple of (global_metrics, breakdown)
    """
    model.eval()
    metrics = RCAMetrics(k_values=[1, 2, 5])

    # Filter to valid anomalies
    valid_data = [
        d for d in test_data
        if d.y is not None and d.y.item() == 1
        and hasattr(d, "root_cause") and d.root_cause[0] in name_to_idx
    ]

    if not valid_data:
        logger.warning("No valid anomalies for evaluation")
        return {}, {}

    # Check for same topology
    same_topology = all(
        torch.equal(d.edge_index, valid_data[0].edge_index) for d in valid_data
    )

    with torch.no_grad():
        if same_topology:
            X = torch.stack([d.x for d in valid_data]).to(device)
            edge_index = valid_data[0].edge_index.to(device)

            for start in range(0, len(valid_data), batch_size):
                end = min(start + batch_size, len(valid_data))
                x_batch = X[start:end]
                batch_data = valid_data[start:end]

                # Get logits
                if mode == "select_then_analyze":
                    logits_full, _, _, sel_logits = model(x_batch, edge_index)
                    scorer = logits_full.clone()
                    scorer[scorer == float("-inf")] = 0.0
                    logits = sel_logits + 0.5 * scorer
                else:
                    _, _, h_spatial = model(x_batch, edge_index)
                    if use_gib:
                        logits = model.gib_logits(h_spatial)
                    else:
                        logits = model.classify_root_cause(h_spatial)

                # Targets
                targets = torch.tensor(
                    [name_to_idx[d.root_cause[0]] for d in batch_data],
                    dtype=torch.long,
                    device=device,
                )

                # Metadata
                metadata = []
                for d in batch_data:
                    meta = {
                        "pattern_type": getattr(d, "pattern_type", "unknown"),
                        "is_recurrent": getattr(d, "is_recurrent", False),
                        "severity": getattr(d, "severity", None),
                        "pattern_difficulty": getattr(d, "pattern_difficulty", "normal"),
                    }
                    metadata.append(meta)

                metrics.update(logits, targets, metadata)
        else:
            # Per-sample inference
            for d in valid_data:
                x = d.x.unsqueeze(0).to(device)
                edge_index = d.edge_index.to(device)

                if mode == "select_then_analyze":
                    logits_full, _, _, sel_logits = model(x, edge_index)
                    scorer = logits_full.clone()
                    scorer[scorer == float("-inf")] = 0.0
                    logits = sel_logits + 0.5 * scorer
                else:
                    _, _, h_spatial = model(x, edge_index)
                    if use_gib:
                        logits = model.gib_logits(h_spatial)
                    else:
                        logits = model.classify_root_cause(h_spatial)

                targets = torch.tensor(
                    [name_to_idx[d.root_cause[0]]],
                    dtype=torch.long,
                    device=device,
                )

                meta = {
                    "pattern_type": getattr(d, "pattern_type", "unknown"),
                    "is_recurrent": getattr(d, "is_recurrent", False),
                    "severity": getattr(d, "severity", None),
                    "pattern_difficulty": getattr(d, "pattern_difficulty", "normal"),
                }
                metrics.update(logits, targets, [meta])

    global_metrics = metrics.compute_metrics()
    breakdown = metrics.compute_breakdown()

    return global_metrics, breakdown


def load_model(checkpoint_path: str):
    """Loads a trained model from a checkpoint file.
    Returns (model, use_gib, mode)."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    mode = checkpoint.get("mode", "semi_supervised")

    if mode == "select_then_analyze":
        num_features = checkpoint.get("num_features", 5)
        hidden_dim = checkpoint.get("hidden_dim", 32)
        num_nodes = checkpoint.get("num_nodes", 5)
        k = checkpoint.get("k", max(1, min(3, num_nodes)))

        model = SelectThenAnalyze(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            k=k,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(
            "Loaded STA checkpoint from %s (features=%d, hidden=%d, nodes=%d, k=%d)",
            checkpoint_path,
            num_features,
            hidden_dim,
            num_nodes,
            k,
        )
        return model, False, mode

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
    return model, use_gib, mode


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
    batch_size: int = 256,
) -> None:
    """
    Evaluates the model on anomalous windows using stratified split.
    Reports per-class and weighted metrics.
    """
    model, use_gib, mode = load_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
    same_topology = all(torch.equal(d.edge_index, test_anomalies[0].edge_index) for d in test_anomalies)

    with torch.no_grad():
        if mode == "select_then_analyze":
            if same_topology:
                X_test = torch.stack([d.x for d in test_anomalies]).to(device)
                y_true = [name_to_idx[d.root_cause[0]] for d in test_anomalies]
                edge_index = test_anomalies[0].edge_index.to(device)

                for start in range(0, len(test_anomalies), batch_size):
                    xb = X_test[start:start + batch_size]
                    logits_full, _, _, sel_logits = model(xb, edge_index)
                    scorer = logits_full.clone()
                    scorer[scorer == float('-inf')] = 0.0
                    combined = sel_logits + 0.5 * scorer
                    y_pred.extend(combined.argmax(dim=1).detach().cpu().tolist())
            else:
                logger.warning("Mixed edge_index detected; falling back to per-window inference (STA).")
                for data in test_anomalies:
                    x = data.x.unsqueeze(0).to(device)
                    edge_index = data.edge_index.to(device)
                    logits_full, _, _, sel_logits = model(x, edge_index)
                    scorer = logits_full.clone()
                    scorer[scorer == float('-inf')] = 0.0
                    combined = sel_logits + 0.5 * scorer
                    y_true.append(name_to_idx[data.root_cause[0]])
                    y_pred.append(int(combined.argmax(dim=1).item()))

        elif same_topology:
            X_test = torch.stack([d.x for d in test_anomalies]).to(device)
            y_true = [name_to_idx[d.root_cause[0]] for d in test_anomalies]
            edge_index = test_anomalies[0].edge_index.to(device)

            for start in range(0, len(test_anomalies), batch_size):
                xb = X_test[start:start + batch_size]
                _, _, h_spatial = model(xb, edge_index)
                if use_gib:
                    logits = model.gib_logits(h_spatial)
                else:
                    logits = model.classify_root_cause(h_spatial)
                y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        else:
            logger.warning("Mixed edge_index detected; falling back to per-window inference.")
            for data in test_anomalies:
                x = data.x.unsqueeze(0).to(device)
                edge_index = data.edge_index.to(device)
                _, _, h_spatial = model(x, edge_index)
                if use_gib:
                    logits = model.gib_logits(h_spatial)
                else:
                    logits = model.classify_root_cause(h_spatial)
                y_true.append(name_to_idx[data.root_cause[0]])
                y_pred.append(int(logits.argmax(dim=1).item()))

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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Show SOTA metrics with breakdown by pattern type and recurrence",
    )
    parser.add_argument(
        "--sota-only", action="store_true",
        help="Only show SOTA metrics (MRR, top-k, P@k), skip legacy metrics",
    )
    args = parser.parse_args()

    if args.breakdown or args.sota_only:
        # SOTA evaluation with breakdown
        model, use_gib, mode = load_model(args.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dataset, node_names = load_dataset_payload(args.dataset)
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        anomalies = get_anomalies(dataset, name_to_idx)

        # Stratified split for test
        by_rc: Dict[str, List[Data]] = {}
        for d in anomalies:
            rc = d.root_cause[0]
            by_rc.setdefault(rc, []).append(d)

        test_anomalies: List[Data] = []
        for rc, items in by_rc.items():
            split = max(1, int(len(items) * 0.8))
            test_anomalies.extend(items[split:])

        logger.info("Evaluating on %d test anomalous windows.", len(test_anomalies))

        global_metrics, breakdown = evaluate_with_sota_metrics(
            model=model,
            test_data=test_anomalies,
            name_to_idx=name_to_idx,
            device=device,
            batch_size=args.batch_size,
            mode=mode,
            use_gib=use_gib,
        )

        # Display SOTA metrics
        logger.info("=" * 60)
        logger.info("SOTA EVALUATION RESULTS")
        logger.info("-" * 60)
        for k, v in global_metrics.items():
            logger.info("  %-20s: %.4f", k, v)

        if args.breakdown and breakdown:
            logger.info("-" * 60)
            logger.info("BREAKDOWN BY PATTERN TYPE:")
            for pattern, stats in breakdown.get("by_pattern_type", {}).items():
                logger.info(
                    "  %-25s | MRR: %.3f | Top-1: %.3f | n=%d",
                    pattern, stats["mrr"], stats["top_1_acc"], stats["count"]
                )

            logger.info("-" * 60)
            logger.info("BREAKDOWN BY RECURRENCE:")
            for rec_type, stats in breakdown.get("by_recurrence", {}).items():
                logger.info(
                    "  %-25s | MRR: %.3f | Top-1: %.3f | n=%d",
                    rec_type, stats["mrr"], stats["top_1_acc"], stats["count"]
                )

            logger.info("-" * 60)
            logger.info("BREAKDOWN BY DIFFICULTY:")
            for diff, stats in breakdown.get("by_difficulty", {}).items():
                if stats["count"] > 0:
                    logger.info(
                        "  %-25s | MRR: %.3f | Top-1: %.3f | n=%d",
                        diff, stats["mrr"], stats["top_1_acc"], stats["count"]
                    )
        logger.info("=" * 60)

        if not args.sota_only:
            # Also run legacy evaluation
            evaluate(
                checkpoint_path=args.checkpoint,
                dataset_path=args.dataset,
                batch_size=args.batch_size,
            )
    else:
        evaluate(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset,
            batch_size=args.batch_size,
        )
