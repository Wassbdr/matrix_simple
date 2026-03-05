import argparse
import torch
import numpy as np
import warnings
from collections import Counter
from baselines import train_pure_gcn, train_random_forest, train_mlp_baseline, get_ranking_metrics
from model import SpatioTemporalRCA
from common import NODE_NAMES, NAME_TO_IDX, stratified_split
import os

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1024]


def print_benchmark_table(metrics_dict, show_std: bool = False):
    if show_std:
        print("\n" + "=" * 95)
        print(f"{'MODEL':<30} | {'TOP-1 ACC':<16} | {'TOP-2 ACC':<16} | {'MRR':<16}")
        print("-" * 95)
        for model_name, (mean, std) in metrics_dict.items():
            t1m, t2m, mm = mean
            t1s, t2s, ms = std
            print(f"{model_name:<30} | {t1m:.4f} ± {t1s:.4f} | {t2m:.4f} ± {t2s:.4f} | {mm:.4f} ± {ms:.4f}")
        print("=" * 95 + "\n")
    else:
        print("\n" + "=" * 80)
        print(f"{'MODEL':<30} | {'TOP-1 ACC':<12} | {'TOP-2 ACC':<12} | {'MRR':<12}")
        print("-" * 80)
        for model_name, metrics in metrics_dict.items():
            top1, top2, mrr = metrics
            print(f"{model_name:<30} | {top1:<12.4f} | {top2:<12.4f} | {mrr:<12.4f}")
        print("=" * 80 + "\n")


def eval_stgcn_checkpoint(checkpoint_path, model_name, dataset, test_anomalies, results):
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping {model_name} evaluation.")
        results[model_name] = (0, 0, 0)
        return
        
    print(f"\n--- Evaluating {model_name} ---")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sample = dataset[0]
    nodes = sample.x.size(0)
    features = sample.x.size(1)
    
    model = SpatioTemporalRCA(
        num_nodes=nodes,
        num_features=features,
        hidden_dim=checkpoint.get('hidden_dim', 32),
        temporal_kernel=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    y_true, y_logits = [], []
    
    use_gib = checkpoint.get('use_gib', True)
    
    print(f"  Inference: {'argmax(mask_probs)' if use_gib else 'node_scorer(h_spatial)'}")
    
    with torch.no_grad():
        for data in test_anomalies:
            x = data.x.unsqueeze(0)
            edge_index = data.edge_index
            
            z_graph, probs, h_spatial = model(x, edge_index, use_gib=use_gib)
            
            if use_gib:
                # GIB mask is the sole predictor
                node_scores = probs.squeeze(-1).squeeze(0).cpu().numpy()  # [num_nodes]
            else:
                # Per-node scorer on spatial embeddings
                rc_logits = model.node_scorer(h_spatial)  # [1, num_nodes]
                node_scores = rc_logits.squeeze(0).cpu().numpy()
            
            y_true.append(NAME_TO_IDX[data.root_cause[0]])
            y_logits.append(node_scores)
            
    if len(y_true) > 0:
        top1, top2, mrr = get_ranking_metrics(y_true, y_logits)
        results[model_name] = (top1, top2, mrr)
    else:
        results[model_name] = (0, 0, 0)


def run_benchmark(dataset_path: str, num_seeds: int = 5):
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    # Stratified train/test split
    train_data, test_data = stratified_split(dataset, train_ratio=0.8)

    # Count anomalies in each split
    train_anomalies = [d for d in train_data if d.y is not None and d.y.item() == 1
                       and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                       and d.root_cause[0] in NAME_TO_IDX]
    test_anomalies = [d for d in test_data if d.y is not None and d.y.item() == 1
                      and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                      and d.root_cause[0] in NAME_TO_IDX]

    print(f"Dataset split: {len(train_data)} train, {len(test_data)} test")
    print(f"  Train anomalies: {len(train_anomalies)}, Test anomalies: {len(test_anomalies)}")

    train_rc = Counter([d.root_cause[0] for d in train_anomalies])
    test_rc = Counter([d.root_cause[0] for d in test_anomalies])
    print(f"  Train RC dist: {dict(train_rc)}")
    print(f"  Test RC dist:  {dict(test_rc)}")

    # -- Single-run results for STGCN checkpoints (deterministic) --
    single_results = {}
    eval_stgcn_checkpoint("checkpoints/stgcn_rca_final.pt", "STGCN + GIB (Semi-Sup)", dataset, test_anomalies, single_results)
    eval_stgcn_checkpoint("checkpoints/stgcn_supervised_only.pt", "STGCN + GIB (Sup Only)", dataset, test_anomalies, single_results)
    eval_stgcn_checkpoint("checkpoints/stgcn_no_gib.pt", "STGCN (No GIB)", dataset, test_anomalies, single_results)

    # -- Multi-seed results for baselines --
    seeds = SEEDS[:num_seeds]
    print(f"\n--- Multi-seed evaluation ({num_seeds} seeds: {seeds}) ---")

    baseline_configs = [
        ("Pure GCN", lambda s: train_pure_gcn(train_data, test_data, epochs=150, seed=s)),
        ("MLP Baseline", lambda s: train_mlp_baseline(train_data, test_data, epochs=150, seed=s)),
        ("Random Forest", lambda s: train_random_forest(train_data, test_data, seed=s)),
    ]

    multi_results = {}
    for name, train_fn in baseline_configs:
        print(f"\n--- Training & Evaluating {name} ({num_seeds} seeds) ---")
        all_runs = []
        for seed in seeds:
            t1, t2, mrr = train_fn(seed)
            all_runs.append((t1, t2, mrr))
            print(f"  Seed {seed}: Top-1={t1:.4f}, Top-2={t2:.4f}, MRR={mrr:.4f}")
        arr = np.array(all_runs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        multi_results[name] = (tuple(mean), tuple(std))

    # -- Print combined table --
    # Convert STGCN single-run to same format (std=0)
    combined = {}
    for name, (t1, t2, mrr) in single_results.items():
        combined[name] = ((t1, t2, mrr), (0.0, 0.0, 0.0))
    for name, (mean, std) in multi_results.items():
        combined[name] = (mean, std)

    print_benchmark_table(combined, show_std=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RCA Models")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset (.pt)")
    
    args = parser.parse_args()
    run_benchmark(args.dataset)
