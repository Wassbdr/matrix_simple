import argparse
import torch
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, recall_score, f1_score
from model import SpatioTemporalRCA
from baselines import train_pure_gcn, train_random_forest
import os

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}


def stratified_split(dataset, train_ratio=0.8):
    """
    Splits dataset ensuring all root cause types appear in both train and test.
    Healthy windows are split chronologically.
    Anomalous windows of each root cause type are split by ratio.
    """
    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in NAME_TO_IDX]

    # Split healthy chronologically
    h_split = int(len(healthy) * train_ratio)
    train_data = healthy[:h_split]
    test_data = healthy[h_split:]

    # Group anomalous by root cause and split each group
    by_rc = {}
    for d in anomalous:
        rc = d.root_cause[0]
        by_rc.setdefault(rc, []).append(d)

    for rc, items in by_rc.items():
        split = max(1, int(len(items) * train_ratio))
        train_data.extend(items[:split])
        test_data.extend(items[split:])

    return train_data, test_data


def print_benchmark_table(metrics_dict):
    print("\n" + "=" * 80)
    print(f"{'MODEL':<20} | {'PRECISION':<15} | {'RECALL':<15} | {'F1-SCORE':<15}")
    print("-" * 80)
    
    for model_name, metrics in metrics_dict.items():
        p, r, f1 = metrics
        print(f"{model_name:<20} | {p:<15.4f} | {r:<15.4f} | {f1:<15.4f}")
    
    print("=" * 80 + "\n")


def run_benchmark(dataset_path: str, stgcn_checkpoint: str):
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
    
    # Show root cause distribution
    from collections import Counter
    train_rc = Counter([d.root_cause[0] for d in train_anomalies])
    test_rc = Counter([d.root_cause[0] for d in test_anomalies])
    print(f"  Train RC dist: {dict(train_rc)}")
    print(f"  Test RC dist:  {dict(test_rc)}")
    
    results = {}
    
    # -------------------------------------------------------------
    # 1. Evaluate SpatioTemporalRCA + GIB (Pre-trained)
    # -------------------------------------------------------------
    print("\n--- Evaluating SpatioTemporalRCA + GIB ---")
    if os.path.exists(stgcn_checkpoint):
        checkpoint = torch.load(stgcn_checkpoint, map_location="cpu", weights_only=False)
        sample = dataset[0]
        nodes = sample.x.size(0)
        features = sample.x.size(1)
        
        # Load model
        model = SpatioTemporalRCA(
            num_nodes=nodes,
            num_features=features,
            hidden_dim=checkpoint.get('hidden_dim',
                checkpoint['model_state_dict']['temporal_block.conv1.weight'].size(0)),
            temporal_kernel=3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data in test_anomalies:
                x = data.x.unsqueeze(0)
                edge_index = data.edge_index
                
                z, probs = model(x, edge_index)
                
                # Use the root cause classifier head
                rc_logits = model.classify_root_cause(z)  # [1, num_nodes]
                pred_idx = rc_logits.argmax(dim=1).item()
                
                y_true.append(NAME_TO_IDX[data.root_cause[0]])
                y_pred.append(pred_idx)
                
        if len(y_true) > 0:
            p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            results['STGCN + GIB'] = (p, r, f1)
            print(f"  y_true: {y_true}")
            print(f"  y_pred: {y_pred}")
        else:
            results['STGCN + GIB'] = (0, 0, 0)
    else:
        print(f"Warning: Checkpoint {stgcn_checkpoint} not found. Skipping STGCN evaluation.")
        results['STGCN + GIB'] = (0, 0, 0)

    # -------------------------------------------------------------
    # 2. Evaluate Pure GCN (Supervised)
    # -------------------------------------------------------------
    print("\n--- Training & Evaluating Pure GCN ---")
    p, r, f1 = train_pure_gcn(train_data, test_data, epochs=100)
    results['Pure GCN (Supervised)'] = (p, r, f1)
    
    # -------------------------------------------------------------
    # 3. Evaluate Random Forest
    # -------------------------------------------------------------
    print("\n--- Training & Evaluating Random Forest ---")
    p, r, f1 = train_random_forest(train_data, test_data)
    results['Random Forest'] = (p, r, f1)
    
    # Print table
    print_benchmark_table(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RCA Models")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset (.pt)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stgcn_rca_final.pt", help="Path to STGCN checkpoint")
    
    args = parser.parse_args()
    run_benchmark(args.dataset, args.checkpoint)
