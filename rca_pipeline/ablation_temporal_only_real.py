#!/usr/bin/env python3
"""
Temporal-only ablation on RCAEval real datasets.

Compares:
  - STA (SelectThenAnalyze) from benchmark_real pipeline
  - Temporal-only baseline (no graph convolution)

This script keeps splits/seeds aligned between methods for fair comparison.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rca_pipeline.benchmark_ablation import TemporalOnlyModel
from rca_pipeline.benchmark_real import train_and_eval_sta
from rca_pipeline.common import set_all_seeds, stratified_split_shuffle
from rca_pipeline.loaders.rcaeval_loader import RCAEvalLoader, SUPPORTED_DATASETS
from rca_pipeline.metrics_rcaeval import RCAEvalMetrics


def train_and_eval_temporal_only(
    train_data,
    test_data,
    node_names: List[str],
    hidden_dim: int,
    epochs: int,
    seed: int,
) -> Tuple[List[int], List[np.ndarray]] | None:
    set_all_seeds(seed)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    n_nodes = len(node_names)
    num_features = train_data[0].x.size(1)

    anomalous = [d for d in train_data if d.root_cause[0] in name_to_idx]
    if len(anomalous) < 2:
        return None

    x_sup = torch.stack([d.x for d in anomalous])
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous], dtype=torch.long)

    model = TemporalOnlyModel(num_features=num_features, hidden_dim=hidden_dim, num_nodes=n_nodes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_sup.size(0))
        for start in range(0, x_sup.size(0), 64):
            idx = perm[start : start + 64]
            opt.zero_grad()
            logits = model(x_sup[idx])
            loss = F.cross_entropy(logits, y_sup[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    test_anom = [d for d in test_data if d.root_cause[0] in name_to_idx]
    y_true, y_logits = [], []
    with torch.no_grad():
        for d in test_anom:
            x = d.x.unsqueeze(0)
            logits = model(x)
            y_true.append(name_to_idx[d.root_cause[0]])
            y_logits.append(logits.squeeze(0).cpu().numpy())
    return y_true, y_logits


def run(args) -> Dict:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    loader = RCAEvalLoader(rcaeval_root=args.rcaeval_root, use_api=False, extract_multimodal=False)
    available = set(loader.available_datasets())
    datasets = [d for d in datasets if d in available]

    metrics = RCAEvalMetrics()
    out: Dict[str, Dict] = {}

    for ds in datasets:
        data_list, node_names, edge_index = loader.load(ds, max_cases=args.max_cases)
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        sta_runs = []
        temporal_runs = []

        for seed in seeds:
            train_data, test_data = stratified_split_shuffle(
                data_list,
                train_ratio=0.8,
                name_to_idx=name_to_idx,
                seed=seed,
                keep_at_least_one_in_test=True,
            )

            sta_res = train_and_eval_sta(
                train_data=train_data,
                test_data=test_data,
                node_names=node_names,
                edge_index=edge_index,
                device=torch.device("cpu"),
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                seed=seed,
                use_mm=False,
                contrastive_warmup=args.sta_contrastive_warmup,
            )
            if sta_res:
                sta_runs.append(sta_res)

            temporal_res = train_and_eval_temporal_only(
                train_data=train_data,
                test_data=test_data,
                node_names=node_names,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                seed=seed,
            )
            if temporal_res:
                temporal_runs.append(temporal_res)

        ds_blob: Dict[str, object] = {
            "config": {
                "dataset": ds,
                "seeds": seeds,
                "epochs": int(args.epochs),
                "hidden_dim": int(args.hidden_dim),
                "max_cases": None if args.max_cases is None else int(args.max_cases),
                "n_cases_loaded": int(len(data_list)),
                "n_nodes": int(len(node_names)),
                "sta_contrastive_warmup": int(args.sta_contrastive_warmup),
            }
        }
        if sta_runs:
            ds_blob["STA"] = metrics.compute_with_std(sta_runs)
        if temporal_runs:
            ds_blob["TemporalOnly"] = metrics.compute_with_std(temporal_runs)
        if "STA" in ds_blob and "TemporalOnly" in ds_blob:
            sta = ds_blob["STA"]
            temporal = ds_blob["TemporalOnly"]
            ds_blob["delta_sta_minus_temporal"] = {
                "avg1": float(sta["avg1"] - temporal["avg1"]),
                "mrr": float(sta["mrr"] - temporal["mrr"]),
                "ndcg3": float(sta["ndcg3"] - temporal["ndcg3"]),
            }
        out[ds] = ds_blob

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-only ablation on RCAEval real datasets")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="42,123,7,1337,2025")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sta-contrastive-warmup", type=int, default=0)
    parser.add_argument("--rcaeval-root", type=str, default="rcaeval")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-json", type=str, default="rca_pipeline/validation_runs/ablation_temporal_real.json")
    args = parser.parse_args()

    results = run(args)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved temporal ablation results to {args.output_json}")


if __name__ == "__main__":
    main()
