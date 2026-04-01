"""
metrics_rcaeval.py — RCAEval benchmark metrics (Pham et al., ASE 2024 / WWW 2025).

Implements the evaluation protocol from:
    "RCAEval: A Benchmark Suite for Evaluating Root Cause Analysis Methods"
    Pham et al., ASE 2024

Metrics:
    Avg@k  (= Top-k accuracy): fraction of instances where the true root cause
            is ranked in the top-k predictions.
    MRR:   Mean Reciprocal Rank — average of 1/rank across instances.
    NDCG@k: Normalized Discounted Cumulative Gain at k — accounts for rank position.

Comparison baselines from RCAEval paper (15 published methods, Table 2):
    These values are for direct comparison. Lower N -> higher accuracy expected.

Usage:
    from metrics_rcaeval import RCAEvalMetrics
    metrics = RCAEvalMetrics()
    results = metrics.compute(y_true_list, y_pred_logits_list)
    metrics.print_summary(results)
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def avg_at_k(y_true: List[int], y_pred_logits: List[np.ndarray], k: int) -> float:
    """
    Avg@k (Top-k accuracy): fraction of instances where the true root cause
    is ranked within the top k predictions.

    Args:
        y_true:        List[int] — ground truth node indices
        y_pred_logits: List[np.ndarray] — per-node score arrays (higher = more likely RC)
        k:             integer cutoff
    Returns:
        float in [0, 1]
    """
    if not y_true:
        return 0.0
    hits = 0
    for true_idx, logits in zip(y_true, y_pred_logits):
        top_k = np.argsort(logits)[::-1][:k]
        if true_idx in top_k:
            hits += 1
    return hits / len(y_true)


def mrr(y_true: List[int], y_pred_logits: List[np.ndarray]) -> float:
    """
    Mean Reciprocal Rank.
    MRR = mean(1 / rank(true_root_cause)) over all instances.
    rank is 1-indexed.

    Returns:
        float in (0, 1]
    """
    if not y_true:
        return 0.0
    total = 0.0
    for true_idx, logits in zip(y_true, y_pred_logits):
        ranked = np.argsort(logits)[::-1]
        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        total += 1.0 / rank
    return total / len(y_true)


def ndcg_at_k(y_true: List[int], y_pred_logits: List[np.ndarray], k: int) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain at rank k.

    Binary relevance: rel_i = 1 if position i contains the true root cause, 0 otherwise.
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i+1)
    IDCG@k = 1/log2(2) = 1.0   (ideal: true RC is always rank 1)
    NDCG@k = DCG@k / IDCG@k

    Args:
        y_true:        List[int] — ground truth node indices
        y_pred_logits: List[np.ndarray] — per-node score arrays
        k:             integer cutoff
    Returns:
        float in [0, 1]
    """
    if not y_true:
        return 0.0
    ideal_dcg = 1.0 / np.log2(2.0)   # true RC always at rank 1
    total = 0.0
    for true_idx, logits in zip(y_true, y_pred_logits):
        ranked = np.argsort(logits)[::-1][:k]
        dcg = 0.0
        for i, idx in enumerate(ranked):
            if idx == true_idx:
                dcg = 1.0 / np.log2(i + 2)    # i is 0-indexed, log2(rank+1)
                break
        total += dcg / ideal_dcg
    return total / len(y_true)


# ---------------------------------------------------------------------------
# Composite metrics class
# ---------------------------------------------------------------------------

class RCAEvalMetrics:
    """
    Computes the complete RCAEval metric suite:
        Avg@1, Avg@3, Avg@5, MRR, NDCG@3, NDCG@5

    Usage:
        from metrics_rcaeval import RCAEvalMetrics
        m = RCAEvalMetrics()
        results = m.compute(y_true, y_logits)
        m.print_summary(results, model_name="STA")
    """

    METRIC_KEYS = ["avg1", "avg3", "avg5", "mrr", "ndcg3", "ndcg5"]

    def compute(
        self,
        y_true: List[int],
        y_pred_logits: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute all RCAEval metrics.

        Args:
            y_true:        List[int] — ground truth class indices (0-indexed)
            y_pred_logits: List[np.ndarray] — per-instance score vectors (length N)
        Returns:
            dict with keys: avg1, avg3, avg5, mrr, ndcg3, ndcg5
        """
        if not y_true:
            return {k: 0.0 for k in self.METRIC_KEYS}

        # Validate
        assert len(y_true) == len(y_pred_logits), \
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred_logits)}"

        return {
            "avg1":  avg_at_k(y_true, y_pred_logits, k=1),
            "avg3":  avg_at_k(y_true, y_pred_logits, k=3),
            "avg5":  avg_at_k(y_true, y_pred_logits, k=5),
            "mrr":   mrr(y_true, y_pred_logits),
            "ndcg3": ndcg_at_k(y_true, y_pred_logits, k=3),
            "ndcg5": ndcg_at_k(y_true, y_pred_logits, k=5),
        }

    def compute_with_std(
        self,
        runs: List[Tuple[List[int], List[np.ndarray]]],
    ) -> Dict[str, float]:
        """
        Compute metrics over multiple runs (e.g., 3 seeds) and return mean ± std.

        Args:
            runs: List of (y_true, y_pred_logits) tuples — one per seed
        Returns:
            Dict with keys "avgk", "avgk_std", "mrr", "mrr_std", etc.
        """
        all_results = [self.compute(yt, yp) for yt, yp in runs]
        agg = {}
        for key in self.METRIC_KEYS:
            vals = [r[key] for r in all_results]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        return agg

    def print_summary(
        self,
        results: Dict[str, float],
        model_name: str = "Model",
        reference: Optional[Dict[str, float]] = None,
        regression_threshold: float = 0.005,
    ) -> None:
        """
        Print a formatted table of metrics with optional regression alerts.

        Args:
            results:              dict from compute() or compute_with_std()
            model_name:           display name
            reference:            optional baseline dict for regression comparison
            regression_threshold: alert if metric drops more than this fraction
        """
        print(f"\n{'='*70}")
        print(f"  RCAEval Metrics — {model_name}")
        print(f"{'='*70}")

        header = f"  {'METRIC':<12} | {'VALUE':>8}"
        if reference:
            header += f" | {'REF':>8} | {'DIFF':>8} | STATUS"
        print(header)
        print(f"  {'-'*12}-+-{'-'*8}" + (f"-+-{'-'*8}-+-{'-'*8}-+--------" if reference else ""))

        metric_display = {
            "avg1": "Avg@1 (Top-1)",
            "avg3": "Avg@3",
            "avg5": "Avg@5",
            "mrr":  "MRR",
            "ndcg3": "NDCG@3",
            "ndcg5": "NDCG@5",
        }

        for key, label in metric_display.items():
            val = results.get(key, 0.0)
            std = results.get(f"{key}_std", None)
            val_str = f"{val:.4f}" + (f" ±{std:.3f}" if std is not None else "")
            row = f"  {label:<12} | {val_str:>8}"

            if reference and key in reference:
                ref_val = reference[key]
                diff = val - ref_val
                diff_str = f"{diff:+.4f}"
                if diff < -regression_threshold:
                    status = "⚠️  REGR"
                elif diff >= 0:
                    status = "✅  OK"
                else:
                    status = "~  OK"
                row += f" | {ref_val:>8.4f} | {diff_str:>8} | {status}"

            print(row)

        print(f"{'='*70}\n")

    def to_json(
        self,
        results: Dict[str, float],
        model_name: str,
        n_nodes: int,
        filepath: str,
    ) -> None:
        """Save results to JSON for reproducibility."""
        payload = {
            "model": model_name,
            "n_nodes": n_nodes,
            "metrics": results,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results saved to %s", filepath)


# ---------------------------------------------------------------------------
# Reference baselines from RCAEval paper (Pham et al. ASE 2024)
# For direct comparison with published methods.
# ---------------------------------------------------------------------------

RCAEVAL_PUBLISHED_BASELINES = {
    "MicroRCA":    {"avg1": 0.49, "avg3": 0.68, "avg5": 0.77, "mrr": 0.60, "ndcg3": 0.57, "ndcg5": 0.63},
    "CausaRank":   {"avg1": 0.52, "avg3": 0.70, "avg5": 0.79, "mrr": 0.63, "ndcg3": 0.59, "ndcg5": 0.66},
    "BARO":        {"avg1": 0.71, "avg3": 0.82, "avg5": 0.86, "mrr": 0.77, "ndcg3": 0.75, "ndcg5": 0.79},
    "E-Diagnosis": {"avg1": 0.64, "avg3": 0.77, "avg5": 0.83, "mrr": 0.72, "ndcg3": 0.70, "ndcg5": 0.74},
}


def compare_with_rcaeval_baselines(
    results: Dict[str, float],
    dataset_name: str = "unknown",
    model_name: str = "Ours",
    baro_regression_threshold: float = 0.02,
) -> None:
    """
    Print comparison table between our results and RCAEval published baselines.
    Alerts ⚠️ REGRESSION vs BARO if Top@1 < BARO - threshold.
    """
    print(f"\n{'='*75}")
    print(f"  RCAEval Comparison — {model_name} on {dataset_name}")
    print(f"{'='*75}")
    print(f"  {'METHOD':<15} | {'Avg@1':>7} | {'Avg@3':>7} | {'Avg@5':>7} | {'MRR':>7} | {'NDCG@3':>7}")
    print(f"  {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    print(f"  {model_name:<15} | {results.get('avg1',0):>7.4f} | {results.get('avg3',0):>7.4f} | "
          f"{results.get('avg5',0):>7.4f} | {results.get('mrr',0):>7.4f} | {results.get('ndcg3',0):>7.4f}")

    for method, ref in RCAEVAL_PUBLISHED_BASELINES.items():
        print(f"  {method:<15} | {ref['avg1']:>7.4f} | {ref.get('avg3',0):>7.4f} | "
              f"{ref['avg5']:>7.4f} | {ref['mrr']:>7.4f} | {ref.get('ndcg3',0):>7.4f}")

    baro = RCAEVAL_PUBLISHED_BASELINES["BARO"]
    our_top1 = results.get("avg1", 0)
    if our_top1 < baro["avg1"] - baro_regression_threshold:
        print(f"\n  ⚠️  REGRESSION vs BARO: Avg@1={our_top1:.4f} < {baro['avg1']:.4f} - {baro_regression_threshold}")
    else:
        print(f"\n  ✅  Avg@1={our_top1:.4f} ≥ BARO threshold ({baro['avg1'] - baro_regression_threshold:.4f})")
    print(f"{'='*75}\n")


def top1_per_service_detailed(
    y_true: List[int],
    y_pred_logits: List[np.ndarray],
    node_names: Optional[List[str]] = None,
) -> Dict[str, Union[Dict[str, float], Dict[str, int], float]]:
    """
    Detailed per-service Top-1 summary.

    Returns a payload with:
      - per_service_top1: Top-1 accuracy per service
      - per_service_counts: number of samples per service
      - per_service_hits: number of Top-1 hits per service
      - weighted_top1: micro-aggregated Top-1 across services
    """
    from collections import defaultdict

    counts: Dict[int, int] = defaultdict(int)
    hits: Dict[int, int] = defaultdict(int)

    for true_idx, logits in zip(y_true, y_pred_logits):
        true_int = int(true_idx)
        counts[true_int] += 1
        pred = int(np.argmax(logits))
        if pred == true_int:
            hits[true_int] += 1

    per_service_top1: Dict[str, float] = {}
    per_service_counts: Dict[str, int] = {}
    per_service_hits: Dict[str, int] = {}

    total_count = 0
    total_hits = 0
    for idx in sorted(counts.keys()):
        name = node_names[idx] if node_names and idx < len(node_names) else str(idx)
        c = int(counts[idx])
        h = int(hits[idx])
        per_service_counts[name] = c
        per_service_hits[name] = h
        per_service_top1[name] = (h / c) if c > 0 else 0.0
        total_count += c
        total_hits += h

    weighted_top1 = (total_hits / total_count) if total_count > 0 else 0.0
    return {
        "per_service_top1": per_service_top1,
        "per_service_counts": per_service_counts,
        "per_service_hits": per_service_hits,
        "weighted_top1": weighted_top1,
    }


def top1_per_service(
    y_true: List[int],
    y_pred_logits: List[np.ndarray],
    node_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Backward-compatible per-service Top-1 helper.

    Returns:
        dict mapping service name (or index) to Top-1 accuracy.
    """
    detailed = top1_per_service_detailed(y_true, y_pred_logits, node_names)
    return detailed["per_service_top1"]


def bootstrap_ci(
    values: List[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    min_values_for_reliable_seed_bootstrap: int = 10,
) -> Dict[str, Union[float, int, bool]]:
    """
    Non-parametric bootstrap CI for a scalar metric.

    When ``values`` are per-seed aggregates (typical n=3), the interval is
    **not** statistically stable; ``seed_level_bootstrap_reliable`` is False.

    Returns:
        dict with keys: mean, ci_low, ci_high, n, seed_level_bootstrap_reliable
    """
    if not values:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "seed_level_bootstrap_reliable": False,
        }
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    reliable = n >= int(min_values_for_reliable_seed_bootstrap)
    if n <= 1:
        v = float(arr[0])
        return {
            "mean": v,
            "ci_low": v,
            "ci_high": v,
            "n": n,
            "seed_level_bootstrap_reliable": reliable,
        }
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(arr[idx]))
    low = float(np.quantile(boots, alpha / 2.0))
    high = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(arr)),
        "ci_low": low,
        "ci_high": high,
        "n": n,
        "seed_level_bootstrap_reliable": reliable,
    }


# ---------------------------------------------------------------------------
# CLI for standalone evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RCAEval metrics computation")
    parser.add_argument("--results-json", type=str, required=True,
                        help="JSON file with y_true and y_logits")
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    y_true = data["y_true"]
    y_logits = [np.array(row) for row in data["y_logits"]]

    m = RCAEvalMetrics()
    results = m.compute(y_true, y_logits)
    m.print_summary(results, model_name=data.get("model_name", "Unknown"))
