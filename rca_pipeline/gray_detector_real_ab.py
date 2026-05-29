"""
A/B evaluation of the gray_failure_detector on REAL cluster data.

Loads existing ObservIT sock-shop sliding windows (no fresh chaos
collection needed — uses the already-collected CSVs under
``datasets/sockshop_runs/``), trains STA once on a stratified split,
then evaluates twice on the same test windows:

    A. STA predictions alone
    B. STA predictions + gray_failure_detector boost on flagged nodes

Reports per-scenario (root cause) Avg@1 for A and B, plus aggregate.

This is the real-data counterpart of the synthetic harness in
``rca_pipeline/detector_ab_harness.py``. Together they provide both a
controlled validation of the detector design and an empirical check
of its uplift on the actual cluster failure modes.

Usage:
    python -m rca_pipeline.gray_detector_real_ab \\
        --dataset-dir datasets/sockshop_runs \\
        --seeds 42,123,7 \\
        --epochs 50 \\
        --boost 3.0 \\
        --output-json results_transmission/gray_detector_real_ab.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from rca_pipeline.benchmark_real import (
    _clone_pyg_data,
    train_and_eval_sta,
)
from rca_pipeline.common import stratified_split_shuffle
from rca_pipeline.graph_builder import GraphBuilder
from rca_pipeline.gray_failure_detector import detect_gray_failure
from rca_pipeline.loaders.observit_loader import load_observit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SCENARIO_FOR_RC = {
    "catalogue":     ("stress-cpu-catalogue",                       "hard"),
    "carts":         ("stress-mem-carts",                           "hard"),
    "orders-db":     ("network-delay-orders-to-orders-db",          "cascade"),
    "catalogue-db":  ("network-loss-catalogue-to-catalogue-db",     "gray"),
    "user":          ("pod-kill-user",                              "hard"),
}


def make_split(data_list, name_to_idx, seed: int):
    """Match benchmark_real's ObservIT pipeline: clone, stratified split,
    StandardScaler fit on train only, applied to both train and test."""
    base = [_clone_pyg_data(d) for d in data_list]
    train_data, test_data = stratified_split_shuffle(
        base,
        train_ratio=0.8,
        name_to_idx=name_to_idx,
        seed=seed,
        keep_at_least_one_in_test=True,
    )
    scaler = GraphBuilder.fit_scaler(train_data)
    train_data = GraphBuilder.apply_scaler(train_data, scaler)
    test_data = GraphBuilder.apply_scaler(test_data, scaler)
    return train_data, test_data


def detector_boost_for_test(test_data, name_to_idx, node_names, boost: float):
    """Run gray_failure_detector on each test window. Returns parallel list
    of {affected_nodes, triggered, mechanism} aligned with the order in which
    train_and_eval_sta iterates over test data (filtered to known RCs)."""
    out = []
    for d in test_data:
        if d.root_cause[0] not in name_to_idx:
            continue
        window_mts = d.x.cpu().numpy()  # [N, F, W]
        gray = detect_gray_failure(
            window_mts,
            latency_idx=4,
            memory_idx=1,
            cpu_idx=0,
        )
        out.append({
            "rc_name": d.root_cause[0],
            "triggered": gray.is_gray,
            "mechanism": gray.mechanism,
            "affected_nodes": [n for n in gray.affected_nodes if 0 <= n < len(node_names)],
        })
    return out


def evaluate_ab(y_true, y_logits, detector_records, boost: float) -> Dict[str, List[dict]]:
    """Compute A and B top-1 per window using existing logits + detector flags.
    Groups results by root cause name."""
    by_rc: Dict[str, List[dict]] = defaultdict(list)
    assert len(y_true) == len(y_logits) == len(detector_records), (
        f"length mismatch: y_true={len(y_true)} logits={len(y_logits)} "
        f"detector={len(detector_records)}"
    )
    for rc_idx, logits, det in zip(y_true, y_logits, detector_records):
        merged = np.asarray(logits, dtype=float).copy()
        boosted = merged.copy()
        if det["triggered"]:
            for n in det["affected_nodes"]:
                if np.isfinite(boosted[n]):
                    boosted[n] += boost
        top1_a = int(np.argmax(merged)) == rc_idx
        top1_b = int(np.argmax(boosted)) == rc_idx
        by_rc[det["rc_name"]].append({
            "top1_a": top1_a,
            "top1_b": top1_b,
            "detector_triggered": bool(det["triggered"]),
            "detector_mechanism": det["mechanism"],
        })
    return by_rc


def aggregate(by_rc: Dict[str, List[dict]]) -> dict:
    rows = []
    for rc, recs in sorted(by_rc.items()):
        n = len(recs)
        if n == 0:
            continue
        avg1_a = sum(r["top1_a"] for r in recs) / n
        avg1_b = sum(r["top1_b"] for r in recs) / n
        trig = sum(r["detector_triggered"] for r in recs) / n
        scenario, ftype = SCENARIO_FOR_RC.get(rc, ("unknown", "unknown"))
        rows.append({
            "root_cause": rc,
            "scenario": scenario,
            "failure_type": ftype,
            "n_windows": n,
            "avg1_baseline": avg1_a,
            "avg1_boosted": avg1_b,
            "uplift": avg1_b - avg1_a,
            "detector_trigger_rate": trig,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/sockshop_runs")
    parser.add_argument("--topology", default="sock_shop")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--seeds", default="42,123,7",
                        help="Comma-separated seeds for repeated runs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--boost", type=float, default=3.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")  # CPU is fine for sockshop N=14
    seeds = [int(s) for s in args.seeds.split(",")]

    logger.info("Loading ObservIT sock-shop windows from %s", args.dataset_dir)
    data_list, node_names, edge_index = load_observit(
        args.dataset_dir,
        window_size=args.window_size,
        topology_name=args.topology,
    )
    logger.info("Loaded %d windows, N=%d nodes", len(data_list), len(node_names))

    # Aggregate per-RC results across seeds
    all_seeds_rows: Dict[Tuple[str, int], dict] = {}
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    for seed in seeds:
        logger.info("=== seed=%d ===", seed)
        train, test = make_split(data_list, name_to_idx, seed)
        logger.info("split: train=%d test=%d", len(train), len(test))
        # Train using benchmark_real's full STA training (with contrastive
        # warmup disabled, matching the publication-grade real-data protocol).
        result = train_and_eval_sta(
            train, test, node_names, edge_index, device,
            hidden_dim=32, epochs=args.epochs, seed=seed,
            contrastive_warmup=0,
        )
        if result is None:
            logger.warning("train_and_eval_sta returned None for seed=%d; skipping", seed)
            continue
        y_true, y_logits = result
        det_records = detector_boost_for_test(test, name_to_idx, node_names, args.boost)
        by_rc = evaluate_ab(y_true, y_logits, det_records, boost=args.boost)
        rows = aggregate(by_rc)
        for r in rows:
            all_seeds_rows[(r["root_cause"], seed)] = r

    # Aggregate across seeds
    per_rc_summary = []
    for rc in sorted({k[0] for k in all_seeds_rows}):
        per_seed = [v for (r, _), v in all_seeds_rows.items() if r == rc]
        if not per_seed:
            continue
        avg1_a_vals = [p["avg1_baseline"] for p in per_seed]
        avg1_b_vals = [p["avg1_boosted"] for p in per_seed]
        trig_vals = [p["detector_trigger_rate"] for p in per_seed]
        per_rc_summary.append({
            "root_cause": rc,
            "scenario": per_seed[0]["scenario"],
            "failure_type": per_seed[0]["failure_type"],
            "n_seeds": len(per_seed),
            "n_windows_total": sum(p["n_windows"] for p in per_seed),
            "avg1_baseline_mean": float(np.mean(avg1_a_vals)),
            "avg1_baseline_std": float(np.std(avg1_a_vals, ddof=1)) if len(avg1_a_vals) > 1 else 0.0,
            "avg1_boosted_mean": float(np.mean(avg1_b_vals)),
            "avg1_boosted_std": float(np.std(avg1_b_vals, ddof=1)) if len(avg1_b_vals) > 1 else 0.0,
            "uplift_mean": float(np.mean(avg1_b_vals) - np.mean(avg1_a_vals)),
            "detector_trigger_rate_mean": float(np.mean(trig_vals)),
        })

    out = {
        "config": {
            "dataset_dir": args.dataset_dir,
            "topology": args.topology,
            "window_size": args.window_size,
            "seeds": seeds,
            "epochs": args.epochs,
            "boost": args.boost,
            "n_windows_total": len(data_list),
            "n_nodes": len(node_names),
        },
        "per_root_cause": per_rc_summary,
        "per_seed_detail": {f"{k[0]}__seed{k[1]}": v for k, v in all_seeds_rows.items()},
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)

    # Print summary
    print("\n=== Per-scenario A/B (real cluster sock-shop data) ===")
    header = f"{'scenario':<45} {'type':<8} {'baseline':>14} {'boosted':>14} {'uplift':>8} {'trigger':>8}"
    print(header)
    print("-" * len(header))
    for r in per_rc_summary:
        print(f"{r['scenario']:<45} {r['failure_type']:<8} "
              f"{r['avg1_baseline_mean']:.3f} +/- {r['avg1_baseline_std']:.3f}  "
              f"{r['avg1_boosted_mean']:.3f} +/- {r['avg1_boosted_std']:.3f}  "
              f"{r['uplift_mean']:>+8.3f} {r['detector_trigger_rate_mean']:>8.3f}")


if __name__ == "__main__":
    main()
