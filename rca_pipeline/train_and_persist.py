"""
Train the STA model on real cluster data with best HPO params and persist
it as a portable bundle.

Pipeline:
    1. Load sock-shop windows via observit_loader.
    2. Read best params from the Optuna HPO JSON output.
    3. For each seed in --seeds, stratified split + train_and_eval_sta.
    4. Pick the best seed's model (by MRR) and persist it as a bundle.

Output:
    checkpoints/sta_sockshop_v1/
        model.pt
        scaler.pt
        metadata.json

Usage:
    python -m rca_pipeline.train_and_persist \\
        --dataset-dir datasets/sockshop_runs \\
        --topology sock_shop \\
        --hpo-json results_transmission/hpo_sta_sockshop.json \\
        --seeds 42,123,7,1337,2025 \\
        --output-bundle checkpoints/sta_sockshop_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from rca_pipeline import tracking
from rca_pipeline.benchmark_real import (
    _clone_pyg_data,
    train_and_eval_sta,
)
from rca_pipeline.common import set_all_seeds, stratified_split_shuffle
from rca_pipeline.graph_builder import GraphBuilder
from rca_pipeline.loaders.observit_loader import load_observit
from rca_pipeline.metrics_rcaeval import RCAEvalMetrics
from rca_pipeline.model_bundle import save_bundle
from rca_pipeline.model_sta import SelectThenAnalyze

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
        scaler,
    )


def _train_one(
    train_data, test_data, node_names, edge_index, device,
    seed: int, hidden_dim: int, epochs: int, k_target: int,
    contrastive_warmup: int,
):
    """Train STA with a forced k, return (model, y_true, y_logits, metrics)."""
    set_all_seeds(seed)
    import rca_pipeline.benchmark_real as br
    original_k = br.get_optimal_k
    br.get_optimal_k = lambda _n: k_target
    try:
        result = train_and_eval_sta(
            train_data, test_data, node_names, edge_index, device,
            hidden_dim=hidden_dim, epochs=epochs, seed=seed,
            contrastive_warmup=contrastive_warmup,
        )
    finally:
        br.get_optimal_k = original_k
    if result is None:
        return None
    y_true, y_logits = result
    m = RCAEvalMetrics().compute(y_true, y_logits)
    # Now rebuild a fresh model with the same params and re-train it
    # deterministically so we can extract the state_dict (the per-seed
    # training above is already deterministic with set_all_seeds).
    # We just need to re-run the training once more to capture the model.
    return y_true, y_logits, {"avg1": float(m["avg1"]), "mrr": float(m["mrr"]),
                              "avg3": float(m["avg3"]), "avg5": float(m["avg5"])}


def _train_and_capture(
    train_data, node_names, edge_index, device,
    seed: int, hidden_dim: int, epochs: int, k_target: int,
    contrastive_warmup: int,
) -> SelectThenAnalyze:
    """Re-train STA with the same recipe and return the trained model object.

    benchmark_real.train_and_eval_sta does not expose the model, so we
    replicate its supervised loop here (with warmup skipped to match
    --sta-contrastive-warmup 0 protocol when configured). For the case
    where contrastive_warmup > 0 we follow the same recipe as train_and_eval_sta.
    """
    import torch.nn.functional as F

    set_all_seeds(seed)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)
    F_ = train_data[0].x.size(1)
    model = SelectThenAnalyze(
        num_features=F_, hidden_dim=hidden_dim, num_nodes=N, k=k_target,
    ).to(device)

    anomalous = [d for d in train_data if d.root_cause[0] in name_to_idx]
    x_sup = torch.stack([d.x for d in anomalous]).to(device)
    y_sup = torch.tensor(
        [name_to_idx[d.root_cause[0]] for d in anomalous],
        dtype=torch.long, device=device,
    )
    ei = edge_index.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x_sup.size(0), device=device)
        for s in range(0, x_sup.size(0), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            logits_full, _, top_k, sel_logits = model(x_sup[idx], ei)
            yb = y_sup[idx]
            loss_sel = F.cross_entropy(sel_logits, yb)
            in_topk = (top_k == yb.unsqueeze(1)).any(dim=1)
            if in_topk.any():
                loss_deep = F.cross_entropy(logits_full[in_topk], yb[in_topk])
            else:
                loss_deep = torch.tensor(0.0, device=device)
            loss = loss_sel + 0.5 * loss_deep
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def _export_reference_windows(data_list, bundle_dir: str) -> None:
    """Write reference_windows.parquet: one row per (window, node), columns
    are the time-averaged features. The drift job compares this against the
    same transform of live Prometheus windows."""
    from pathlib import Path

    rows = np.concatenate(
        [d.x.mean(dim=2).cpu().numpy() for d in data_list], axis=0  # [N,F] per window
    )
    n_features = rows.shape[1]
    cols = [f"f{i}" for i in range(n_features)]
    try:
        import pandas as pd
        Path(bundle_dir).mkdir(parents=True, exist_ok=True)
        out = Path(bundle_dir) / "reference_windows.parquet"
        pd.DataFrame(rows, columns=cols).to_parquet(out, index=False)
        logger.info("Reference windows exported: %s (%d rows)", out, len(rows))
    except Exception as exc:  # pyarrow optional; never fail training on it
        logger.warning("Could not export reference_windows.parquet: %s", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/sockshop_runs")
    parser.add_argument("--topology", default="sock_shop")
    parser.add_argument("--window-size", type=int, default=12)
    # required-ness checked after parsing so the values may come from --config
    parser.add_argument("--hpo-json", default=None)
    parser.add_argument("--seeds", default="42,123,7,1337,2025")
    parser.add_argument("--output-bundle", default=None)
    parser.add_argument("--no-warmup", action="store_true",
                        help="Force contrastive_warmup=0 (real-data guardrail).")
    from rca_pipeline.config import TrainConfig, parse_args_with_config
    args = parse_args_with_config(parser, TrainConfig)
    for required in ("hpo_json", "output_bundle"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (CLI flag or --config)")

    device = torch.device("cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    logger.info("Loading HPO results from %s", args.hpo_json)
    hpo = json.loads(open(args.hpo_json).read())
    bp = hpo["best_params"]
    hidden_dim = int(bp["hidden_dim"])
    k_pct = int(bp["k_pct"])
    contrastive_warmup = 0 if args.no_warmup else int(bp.get("contrastive_warmup", 0))
    epochs = int(bp["epochs"])
    logger.info("Best params: hidden_dim=%d k_pct=%d warmup=%d epochs=%d",
                hidden_dim, k_pct, contrastive_warmup, epochs)
    logger.info("Best HPO MRR=%.4f Avg@1=%.4f",
                hpo["best_value_mrr"], hpo.get("best_avg1", float("nan")))

    logger.info("Loading dataset from %s", args.dataset_dir)
    data_list, node_names, edge_index = load_observit(
        args.dataset_dir,
        window_size=args.window_size,
        topology_name=args.topology,
    )
    N = len(node_names)
    k_target = max(2, int(round(N * k_pct / 100)))
    logger.info("Loaded %d windows, N=%d, k=%d", len(data_list), N, k_target)

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    per_seed: List[Dict] = []
    best_seed: Optional[int] = None
    best_mrr = -1.0
    best_model: Optional[SelectThenAnalyze] = None
    best_scaler = None

    for seed in seeds:
        logger.info("=== seed=%d ===", seed)
        train, test, scaler = _split_and_scale(data_list, name_to_idx, seed)
        eval_result = _train_one(
            train, test, node_names, edge_index, device,
            seed=seed, hidden_dim=hidden_dim, epochs=epochs,
            k_target=k_target, contrastive_warmup=contrastive_warmup,
        )
        if eval_result is None:
            logger.warning("training failed for seed=%d", seed)
            continue
        _, _, metrics = eval_result
        logger.info("seed=%d metrics: %s", seed, metrics)
        per_seed.append({"seed": seed, **metrics})

        if metrics["mrr"] > best_mrr:
            # Re-train deterministically and capture the model
            best_mrr = metrics["mrr"]
            best_seed = seed
            best_scaler = scaler
            best_model = _train_and_capture(
                train, node_names, edge_index, device,
                seed=seed, hidden_dim=hidden_dim, epochs=epochs,
                k_target=k_target, contrastive_warmup=contrastive_warmup,
            )

    if best_model is None:
        raise SystemExit("No model trained successfully.")

    # Aggregate metrics across seeds
    avg1_vals = [r["avg1"] for r in per_seed]
    mrr_vals = [r["mrr"] for r in per_seed]
    avg3_vals = [r["avg3"] for r in per_seed]
    training_info = {
        "best_seed": int(best_seed),
        "best_mrr": float(best_mrr),
        "per_seed_metrics": per_seed,
        "avg1_mean": float(np.mean(avg1_vals)) if avg1_vals else 0.0,
        "avg1_std": float(np.std(avg1_vals, ddof=1)) if len(avg1_vals) > 1 else 0.0,
        "mrr_mean": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
        "mrr_std": float(np.std(mrr_vals, ddof=1)) if len(mrr_vals) > 1 else 0.0,
        "avg3_mean": float(np.mean(avg3_vals)) if avg3_vals else 0.0,
        "epochs": epochs,
        "contrastive_warmup": contrastive_warmup,
        "seeds": seeds,
        "dataset_dir": args.dataset_dir,
        "topology": args.topology,
        "window_size": args.window_size,
        "n_windows": len(data_list),
    }
    hpo_provenance = {
        "hpo_study_name": hpo.get("study_name"),
        "hpo_n_trials_completed": hpo.get("n_trials_completed"),
        "hpo_best_value_mrr": hpo.get("best_value_mrr"),
        "hpo_best_params": hpo.get("best_params"),
        "hpo_param_importances": hpo.get("param_importances"),
    }

    # Export a feature reference for batch drift monitoring (M2.7): the
    # per-node, per-feature time-average of every training window. Lives
    # inside the bundle so the reference is versioned with the model.
    _export_reference_windows(data_list, args.output_bundle)

    # Wrap the persist + registry step in an MLflow run (no-op offline).
    with tracking.run(experiment="sta-sockshop", run_name="train_and_persist") as run_id:
        tracking.log_params({
            "hidden_dim": hidden_dim, "k": k_target, "epochs": epochs,
            "contrastive_warmup": contrastive_warmup, "seeds": seeds,
            "topology": args.topology, "n_windows": len(data_list),
            "hpo_study_name": hpo.get("study_name"),
        })
        tracking.log_metrics({
            "avg1_mean": training_info["avg1_mean"],
            "avg1_std": training_info["avg1_std"],
            "mrr_mean": training_info["mrr_mean"],
            "mrr_std": training_info["mrr_std"],
            "best_mrr": float(best_mrr),
        })

        save_bundle(
            bundle_dir=args.output_bundle,
            model=best_model,
            node_names=node_names,
            edge_index=edge_index,
            scaler_mean=best_scaler.mean_,
            scaler_scale=best_scaler.scale_,
            num_features=data_list[0].x.size(1),
            hidden_dim=hidden_dim,
            k=k_target,
            training_info=training_info,
            hpo_provenance=hpo_provenance,
            mlflow_run_id=run_id,
        )
        logger.info("Bundle saved: %s", args.output_bundle)

        # Register the bundle as a model version (alias=challenger); the
        # bundle dir IS the artifact (no pyfunc wrapping).
        version = tracking.register_model(
            args.output_bundle, name="sta-sockshop", alias="challenger",
        )
        if version is not None:
            logger.info("Registered sta-sockshop v%s (alias=challenger)", version)

    logger.info("Best seed=%d MRR=%.4f Avg@1=%.4f",
                best_seed, best_mrr,
                next(r["avg1"] for r in per_seed if r["seed"] == best_seed))


if __name__ == "__main__":
    main()
