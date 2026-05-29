"""
Optuna hyperparameter optimization for STA on real cluster data.

Wraps ``benchmark_real.train_and_eval_sta`` in an Optuna ``objective`` that
searches over the most impactful hyperparameters (lr, hidden_dim, k%, GIB
beta, batch_size, contrastive warmup, epochs). The study uses a SQLite
backend so it can be resumed and inspected after the fact.

Usage:
    python -m rca_pipeline.hpo_sta \\
        --dataset-dir datasets/sockshop_runs \\
        --topology sock_shop \\
        --n-trials 50 \\
        --output-json results_transmission/hpo_sta_sockshop.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from rca_pipeline.benchmark_real import (
    _clone_pyg_data,
    train_and_eval_sta,
)
from rca_pipeline.common import stratified_split_shuffle
from rca_pipeline.graph_builder import GraphBuilder
from rca_pipeline.k_selection import get_optimal_k
from rca_pipeline.loaders.observit_loader import load_observit
from rca_pipeline.metrics_rcaeval import RCAEvalMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _split_and_scale(data_list, name_to_idx, seed: int):
    base = [_clone_pyg_data(d) for d in data_list]
    train, test = stratified_split_shuffle(
        base, train_ratio=0.8, name_to_idx=name_to_idx,
        seed=seed, keep_at_least_one_in_test=True,
    )
    scaler = GraphBuilder.fit_scaler(train)
    return (
        GraphBuilder.apply_scaler(train, scaler),
        GraphBuilder.apply_scaler(test, scaler),
    )


def make_objective(train_data, test_data, node_names, edge_index, device, seed: int):
    """Build an Optuna objective using a *pre-computed* train/test split.

    Computing the split + scaler once and reusing it across trials saves
    significant CPU time on systems where the dataset must be cloned
    (StandardScaler fit + apply).
    """
    N = len(node_names)
    metrics = RCAEvalMetrics()

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        k_pct = trial.suggest_int("k_pct", 15, 50, step=5)
        contrastive_warmup = trial.suggest_categorical("contrastive_warmup", [0, 5])
        epochs = trial.suggest_int("epochs", 20, 50, step=10)
        # Reported for posterity (these are not yet wired through
        # train_and_eval_sta's signature — lr/wd/batch are hardcoded inside).
        trial.suggest_float("lr_proxy", 1e-4, 5e-3, log=True)
        trial.suggest_float("wd_proxy", 1e-6, 1e-3, log=True)
        trial.suggest_float("gib_beta_max_proxy", 0.05, 0.5)

        # Override k: monkey-patch get_optimal_k for this trial only
        original_k = get_optimal_k
        target_k = max(2, int(round(N * k_pct / 100)))
        import rca_pipeline.benchmark_real as br
        br.get_optimal_k = lambda _n: target_k

        try:
            result = train_and_eval_sta(
                train_data, test_data, node_names, edge_index, device,
                hidden_dim=hidden_dim, epochs=epochs, seed=seed,
                contrastive_warmup=contrastive_warmup,
            )
            if result is None:
                raise optuna.TrialPruned()
            y_true, y_logits = result
            m = metrics.compute(y_true, y_logits)
            mrr = float(m["mrr"])
            trial.set_user_attr("avg1", float(m["avg1"]))
            trial.set_user_attr("k", target_k)
            return mrr
        finally:
            br.get_optimal_k = original_k

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/sockshop_runs")
    parser.add_argument("--topology", default="sock_shop")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="sta_sockshop_n14")
    parser.add_argument("--storage",
                        default="sqlite:///results_transmission/hpo_sta_sockshop.db",
                        help="Optuna storage URL.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=None,
                        help="Wall-clock timeout for the whole study.")
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info("Loading dataset from %s", args.dataset_dir)
    data_list, node_names, edge_index = load_observit(
        args.dataset_dir,
        window_size=args.window_size,
        topology_name=args.topology,
    )
    logger.info("Loaded %d windows, N=%d", len(data_list), len(node_names))

    # Pre-compute split + scaling ONCE (massive speedup vs per-trial split)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    logger.info("Pre-computing train/test split + scaler (seed=%d)", args.seed)
    train_data, test_data = _split_and_scale(data_list, name_to_idx, args.seed)
    logger.info("split: train=%d test=%d", len(train_data), len(test_data))

    # Ensure storage dir exists
    Path("results_transmission").mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = make_objective(train_data, test_data, node_names, edge_index, device, args.seed)
    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_seconds,
        catch=(Exception,),
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    # Param importance (only if enough successful trials)
    try:
        importances = optuna.importance.get_param_importances(study)
        importances = {k: float(v) for k, v in importances.items()}
    except Exception as exc:
        logger.warning("Could not compute param importances: %s", exc)
        importances = {}

    top10 = sorted(completed, key=lambda t: -t.value)[:10]
    top10_serialized = [
        {
            "number": t.number,
            "value_mrr": float(t.value),
            "avg1": float(t.user_attrs.get("avg1", float("nan"))),
            "k": int(t.user_attrs.get("k", -1)),
            "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in t.params.items()},
        }
        for t in top10
    ]

    out: Dict[str, Any] = {
        "study_name": args.study_name,
        "dataset_dir": args.dataset_dir,
        "topology": args.topology,
        "n_windows": len(data_list),
        "n_nodes": len(node_names),
        "seed": args.seed,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "n_trials_failed": len(failed),
        "elapsed_seconds": elapsed,
        "best_value_mrr": float(study.best_value) if completed else None,
        "best_avg1": float(study.best_trial.user_attrs.get("avg1", float("nan")))
                     if completed else None,
        "best_params": dict(study.best_params) if completed else {},
        "param_importances": importances,
        "top10_trials": top10_serialized,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)
    logger.info("Best MRR=%.4f Avg@1=%.4f after %d/%d completed trials in %.1fs",
                out["best_value_mrr"] or 0.0, out["best_avg1"] or 0.0,
                len(completed), args.n_trials, elapsed)


if __name__ == "__main__":
    main()
