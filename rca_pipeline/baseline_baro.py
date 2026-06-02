"""Isoperimetric BARO baseline on RCAEval, under *our* protocol.

BARO (Pham et al., RCAEval) ranks services by the maximum RobustScaler z-score
of the anomalous period relative to the normal period (split at the injection
time). Published numbers use RCAEval's own split/preprocessing, so they are only
*partially comparable* to ours.

This script runs BARO's core mechanism on the **exact same cases, candidate
node set, train/test split (per seed) and metric functions** as
`benchmark_real.py`, turning the comparison isoperimetric. Concretely, for each
test case it reads the raw ``metrics.csv`` + ``inject_time.txt`` (exposed as
``Data.case_path`` by the loader), scores each candidate node by BARO, and feeds
``(y_true, y_logits)`` to the same ``RCAEvalMetrics`` used for STA / GATv2.

Differences from the published BARO that we keep explicit:
- BARO ranks over the **same candidate node set** our models do (the
  fault-injectable services), not all instrumented services.
- BARO sees the **same per-node feature columns** our loader selects
  (``num_features`` columns), but uses the full pre/post-injection series for
  its normal baseline (as BARO requires).

Usage:
    python -m rca_pipeline.baseline_baro --output-json results_transmission/baselines_rcaeval_ourprotocol.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from rca_pipeline.common import stratified_split_shuffle
from rca_pipeline.loaders.rcaeval_loader import RCAEvalLoader
from rca_pipeline.metrics_rcaeval import RCAEVAL_PUBLISHED_BASELINES, RCAEvalMetrics

DATASETS = ["trainticket", "sockshop", "online-boutique"]
DEFAULT_SEEDS = [42, 123, 7, 1337, 2025]


def _read_inject_time(case_path: str) -> Optional[int]:
    p = os.path.join(case_path, "inject_time.txt")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        txt = fh.read().strip()
    try:
        return int(float(txt))
    except ValueError:
        return None


def _read_metrics(case_path: str) -> Optional[pd.DataFrame]:
    for fname in ("metrics.csv", "simple_metrics.csv"):
        p = os.path.join(case_path, fname)
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def _node_columns(df: pd.DataFrame, node: str, num_features: int) -> List[str]:
    """Mirror the loader's column selection: ts-<node>_* then <node>_*."""
    cols = [c for c in df.columns if c.startswith(f"ts-{node}_")]
    if not cols:
        cols = [c for c in df.columns if c.startswith(f"{node}_")]
    return cols[:num_features]


def baro_scores(
    case_path: str, node_names: List[str], num_features: int
) -> Optional[np.ndarray]:
    """Per-node BARO score vector [N] (max RobustScaler z over the node's cols)."""
    df = _read_metrics(case_path)
    if df is None or "time" not in df.columns:
        return None
    inject = _read_inject_time(case_path)
    if inject is None:
        return None

    t = pd.to_numeric(df["time"], errors="coerce")
    normal = df[t < inject]
    anomal = df[t >= inject]
    if len(normal) < 2 or len(anomal) < 1:  # degenerate split → midpoint fallback
        mid = len(df) // 2
        normal, anomal = df.iloc[:mid], df.iloc[mid:]
    if len(normal) < 2 or len(anomal) < 1:
        return None

    scores = np.full(len(node_names), -np.inf, dtype=np.float64)
    for i, node in enumerate(node_names):
        best = -np.inf
        for col in _node_columns(df, node, num_features):
            a = pd.to_numeric(normal[col], errors="coerce").to_numpy()
            b = pd.to_numeric(anomal[col], errors="coerce").to_numpy()
            a = np.nan_to_num(a, nan=0.0)
            b = np.nan_to_num(b, nan=0.0)
            scaler = RobustScaler().fit(a.reshape(-1, 1))  # sklearn guards IQR=0
            z = scaler.transform(b.reshape(-1, 1))[:, 0]
            best = max(best, float(np.max(z)))
        scores[i] = best if np.isfinite(best) else 0.0
    # nodes with no matching columns → 0 (kept rankable, below any real score)
    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(np.float32)


def run_dataset(
    loader: RCAEvalLoader, dataset: str, seeds: List[int]
) -> Dict:
    data_list, node_names, _ = loader.load(dataset)
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    runs = []  # list of (y_true, y_logits) per seed
    n_test_per_seed = []
    n_missing = 0
    for seed in seeds:
        _, test_data = stratified_split_shuffle(
            data_list, train_ratio=0.8, name_to_idx=name_to_idx,
            seed=seed, keep_at_least_one_in_test=True,
        )
        y_true, y_logits = [], []
        for d in test_data:
            rc = d.root_cause[0] if d.root_cause else ""
            if rc not in name_to_idx:
                continue
            cp = getattr(d, "case_path", None)
            s = baro_scores(cp, node_names, loader.num_features) if cp else None
            if s is None:
                n_missing += 1
                continue
            y_true.append(name_to_idx[rc])
            y_logits.append(s)
        if y_true:
            runs.append((y_true, y_logits))
            n_test_per_seed.append(len(y_true))

    metrics = RCAEvalMetrics().compute_with_std(runs)
    metrics["n_nodes"] = len(node_names)
    metrics["node_names"] = node_names
    metrics["n_test_cases_per_seed"] = n_test_per_seed
    metrics["n_seeds"] = len(runs)
    metrics["n_cases_missing_inject_or_metrics"] = n_missing
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Isoperimetric BARO baseline (our protocol).")
    ap.add_argument("--rcaeval-root", default="rcaeval")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--window-size", type=int, default=12)
    ap.add_argument("--num-features", type=int, default=5)
    ap.add_argument("--our-results", default="results_transmission/paper_benchmark_real.json",
                    help="our STA/GATv2 results for side-by-side context")
    ap.add_argument("--output-json", default="results_transmission/baselines_rcaeval_ourprotocol.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    loader = RCAEvalLoader(
        rcaeval_root=args.rcaeval_root,
        window_size=args.window_size,
        num_features=args.num_features,
    )

    ours = {}
    if os.path.exists(args.our_results):
        with open(args.our_results) as fh:
            ours = json.load(fh)

    out: Dict[str, Dict] = {
        "_meta": {
            "method": "BARO (RobustScaler max-zscore ranking), our protocol",
            "isoperimetric": True,
            "note": (
                "Same cases, candidate node set, per-seed stratified split and "
                "metric functions as benchmark_real.py. BARO uses the full "
                "pre/post-injection series (split at inject_time.txt) over the "
                "same num_features columns our models see."
            ),
            "seeds": seeds,
        }
    }
    for ds in datasets:
        baro_m = run_dataset(loader, ds, seeds)
        entry = {
            "BARO_ourprotocol": baro_m,
            "BARO_published": RCAEVAL_PUBLISHED_BASELINES["BARO"],
        }
        ds_ours = ours.get(ds, {})
        for mk in ("STA", "GATv2"):
            if mk in ds_ours:
                entry[f"{mk}_ours"] = {
                    k: ds_ours[mk][k] for k in ("avg1", "avg1_std", "mrr", "mrr_std")
                    if k in ds_ours[mk]
                }
        out[ds] = entry
        b1 = baro_m.get("avg1", 0.0)
        print(f"{ds:16s} BARO(ours)={b1:.3f}  BARO(pub)={RCAEVAL_PUBLISHED_BASELINES['BARO']['avg1']:.2f}"
              f"  STA(ours)={ds_ours.get('STA', {}).get('avg1', float('nan')):.3f}"
              f"  n_test/seed={baro_m.get('n_test_cases_per_seed')}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
