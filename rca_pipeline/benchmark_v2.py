"""
Benchmark v2: Compare legacy STGCN (5-node) vs SelectThenAnalyze (20-node).

Evaluations:
    A) Legacy models on 5-node dataset (unchanged from benchmark.py)
    B) SelectThenAnalyze on 20-node dataset: Semi-Sup and Sup-Only variants
    C) Baselines on 20-node dataset: Pure GCN, MLP, RF (multi-seed)
    
Measures: Top-1 Accuracy, Top-2 Accuracy, Top-5 Accuracy, MRR
         + GIB selection rate (% of samples where root cause ∈ top-k)
"""

import argparse
import torch
import numpy as np
import warnings
import time
from collections import Counter
from typing import List, Tuple, Dict
from torch_geometric.data import Data


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1024]


def get_ranking_metrics(y_true_list, y_pred_logits_list, top_ns=(1, 2, 5)):
    """Extended ranking metrics with configurable top-N."""
    tops = {n: 0.0 for n in top_ns}
    mrr = 0.0
    for y_true, logits in zip(y_true_list, y_pred_logits_list):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        ranked = np.argsort(logits)[::-1]
        rank = int(np.where(ranked == y_true)[0][0]) + 1
        for n in top_ns:
            if rank <= n:
                tops[n] += 1
        mrr += 1.0 / rank
    
    count = len(y_true_list)
    if count == 0:
        return {n: 0.0 for n in top_ns}, 0.0
    return {n: tops[n] / count for n in top_ns}, mrr / count


def stratified_split_generic(dataset, node_names, train_ratio=0.8):
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in name_to_idx]
    
    h_split = int(len(healthy) * train_ratio)
    train_data = healthy[:h_split]
    test_data = healthy[h_split:]
    
    by_rc: Dict[str, list] = {}
    for d in anomalous:
        by_rc.setdefault(d.root_cause[0], []).append(d)
    for rc, items in by_rc.items():
        split = max(1, int(len(items) * train_ratio))
        train_data.extend(items[:split])
        test_data.extend(items[split:])
    
    return train_data, test_data


def get_anomalies(data_list, node_names):
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    return [d for d in data_list
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and len(d.root_cause) > 0
            and d.root_cause[0] in name_to_idx]


# ── SelectThenAnalyze evaluation ────────────────────────────────────

def eval_select_then_analyze(checkpoint_path, dataset, node_names):
    """Evaluate a SelectThenAnalyze checkpoint."""
    import os
    from model_v2 import SelectThenAnalyze
    
    if not os.path.exists(checkpoint_path):
        print(f"  Warning: {checkpoint_path} not found.")
        return None
    
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = SelectThenAnalyze(
        num_features=checkpoint['num_features'],
        hidden_dim=checkpoint['hidden_dim'],
        num_nodes=checkpoint['num_nodes'],
        k=checkpoint['k'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    _, test_data = stratified_split_generic(dataset, node_names)
    test_anomalies = get_anomalies(test_data, node_names)
    
    if not test_anomalies:
        return None
    
    y_true, y_logits = [], []
    selection_hits = 0
    
    # Warm-up pass (exclude from timing)
    with torch.no_grad():
        _x = test_anomalies[0].x.unsqueeze(0)
        _ei = test_anomalies[0].edge_index
        _ = model(_x, _ei)
    
    # Timed inference
    latencies = []
    with torch.no_grad():
        for data in test_anomalies:
            x = data.x.unsqueeze(0)
            edge_index = data.edge_index
            
            t0 = time.perf_counter()
            logits_full, probs, top_k, gib_logits = model(x, edge_index)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            
            true_idx = name_to_idx[data.root_cause[0]]
            y_true.append(true_idx)
            
            # Ensemble: GIB probs + scorer logits (mapped to full graph)
            gib_scores = probs.squeeze(0)
            scorer_scores = torch.softmax(logits_full.squeeze(0), dim=-1)
            combined = 0.5 * gib_scores + 0.5 * scorer_scores
            scores = combined.cpu().numpy()
            y_logits.append(scores)
            
            if true_idx in top_k[0].tolist():
                selection_hits += 1
    
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    selection_rate = selection_hits / len(y_true)
    total_params, _ = count_parameters(model)
    
    return {
        'top1': tops[1], 'top2': tops[2], 'top5': tops[5],
        'mrr': mrr, 'selection_rate': selection_rate,
        'n_test': len(y_true),
        'mean_latency_ms': np.mean(latencies) * 1000,
        'std_latency_ms': np.std(latencies) * 1000,
        'params': total_params,
        'deep_nodes': checkpoint['k'],
        'total_nodes': checkpoint['num_nodes'],
    }


# ── Legacy STGCN evaluation (5-node) ───────────────────────────────

def eval_legacy_stgcn(checkpoint_path, dataset, node_names):
    """Evaluate a legacy SpatioTemporalRCA checkpoint."""
    import os
    from model import SpatioTemporalRCA
    
    if not os.path.exists(checkpoint_path):
        print(f"  Warning: {checkpoint_path} not found.")
        return None

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = SpatioTemporalRCA(
        num_features=checkpoint.get('num_features', 5),
        hidden_dim=checkpoint.get('hidden_dim', 32),
        num_nodes=len(node_names),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    _, test_data = stratified_split_generic(dataset, node_names)
    test_anomalies = get_anomalies(test_data, node_names)
    
    use_gib = checkpoint.get('use_gib', True)
    y_true, y_logits = [], []
    
    with torch.no_grad():
        for data in test_anomalies:
            x = data.x.unsqueeze(0)
            edge_index = data.edge_index
            _, probs, h_spatial = model(x, edge_index, use_gib=use_gib)
            
            if use_gib:
                scores = probs.squeeze(-1).squeeze(0).cpu().numpy()
            else:
                scores = model.node_scorer(h_spatial).squeeze(0).cpu().numpy()
            
            y_true.append(name_to_idx[data.root_cause[0]])
            y_logits.append(scores)
    
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {
        'top1': tops[1], 'top2': tops[2], 'top5': tops[5],
        'mrr': mrr, 'n_test': len(y_true),
    }


# ── Baselines on 20-node dataset ───────────────────────────────────

def train_pure_gcn_large(train_data, test_data, node_names, epochs=150, hidden_dim=32, batch_size=256, seed=42):
    """Pure GCN baseline adapted for any graph size."""
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    torch.manual_seed(seed)
    
    train_anom = get_anomalies(train_data, node_names)
    test_anom = get_anomalies(test_data, node_names)
    if not train_anom or not test_anom:
        return None
    
    sample = train_anom[0]
    N, F_ = sample.x.size(0), sample.x.size(1)
    edge_index = sample.edge_index
    
    class PureGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.tc1 = nn.Conv1d(F_, hidden_dim, 3, padding=1)
            self.tc2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
            self.tn = nn.BatchNorm1d(hidden_dim)
            self.ta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
            self.g1 = GCNConv(hidden_dim, hidden_dim)
            self.g2 = GCNConv(hidden_dim, hidden_dim)
            self.sn = nn.LayerNorm(hidden_dim)
            self.sc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))
        
        def forward(self, x, ei):
            sq = x.dim() == 3
            if sq: x = x.unsqueeze(0)
            B, V, Ff, W = x.shape
            h = x.view(B*V, Ff, W)
            h = torch.nn.functional.relu(self.tc1(h))
            h = torch.nn.functional.relu(self.tc2(h))
            ht = h.permute(0,2,1)
            a = torch.nn.functional.softmax(self.ta(ht).squeeze(-1), dim=-1)
            h = (ht * a.unsqueeze(-1)).sum(1)
            h = self.tn(h)
            if B > 1:
                off = torch.arange(B, device=h.device) * V
                bei = ei.unsqueeze(0).expand(B,-1,-1) + off.view(-1,1,1)
                bei = bei.reshape(2,-1)
            else:
                bei = ei
            r = h
            h = torch.nn.functional.relu(self.g1(h, bei))
            h = self.g2(h, bei)
            h = self.sn(h + r)
            h = h.view(B, V, -1)
            lo = self.sc(h).squeeze(-1)
            return lo.squeeze(0) if sq else lo
    
    model = PureGCN()
    X_train = torch.stack([d.x for d in train_anom])
    y_train = torch.tensor([name_to_idx[d.root_cause[0]] for d in train_anom], dtype=torch.long)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    n = len(train_anom)
    
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            opt.zero_grad()
            lo = model(X_train[idx], edge_index)
            loss = crit(lo, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    
    model.eval()
    total_params, _ = count_parameters(model)
    N = len(node_names)
    y_true, y_logits = [], []
    X_test = torch.stack([d.x for d in test_anom])
    y_test_labels = [name_to_idx[d.root_cause[0]] for d in test_anom]
    
    # Warm-up
    with torch.no_grad():
        _ = model(X_test[:1], edge_index)
    
    # Timed inference (single-sample to match STA measurement)
    latencies = []
    with torch.no_grad():
        for i in range(len(test_anom)):
            t0 = time.perf_counter()
            lo = model(X_test[i:i+1], edge_index)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            if lo.dim() == 1:
                lo = lo.unsqueeze(0)
            y_true.append(y_test_labels[i])
            y_logits.append(lo[0].cpu().numpy())
    
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {
        'top1': tops[1], 'top2': tops[2], 'top5': tops[5], 'mrr': mrr,
        'mean_latency_ms': np.mean(latencies) * 1000,
        'std_latency_ms': np.std(latencies) * 1000,
        'params': total_params,
        'deep_nodes': N,
        'total_nodes': N,
    }


def train_rf_large(train_data, test_data, node_names, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    
    train_anom = get_anomalies(train_data, node_names)
    test_anom = get_anomalies(test_data, node_names)
    if not train_anom or not test_anom:
        return None
    
    N = len(node_names)
    X_train = np.array([d.x.numpy().flatten() for d in train_anom])
    y_train = np.array([name_to_idx[d.root_cause[0]] for d in train_anom])
    X_test = np.array([d.x.numpy().flatten() for d in test_anom])
    y_test = np.array([name_to_idx[d.root_cause[0]] for d in test_anom])
    
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced",
                                 max_depth=20, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    
    y_pred_probs = clf.predict_proba(X_test)
    logits = np.zeros((len(X_test), N))
    for i, c in enumerate(clf.classes_):
        logits[:, c] = y_pred_probs[:, i]
    
    tops, mrr = get_ranking_metrics(y_test, logits, top_ns=(1, 2, 5))
    return {'top1': tops[1], 'top2': tops[2], 'top5': tops[5], 'mrr': mrr}


def train_mlp_large(train_data, test_data, node_names, epochs=150, hidden_dim=128, batch_size=256, seed=42):
    import torch.nn as nn
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    torch.manual_seed(seed)
    
    train_anom = get_anomalies(train_data, node_names)
    test_anom = get_anomalies(test_data, node_names)
    if not train_anom or not test_anom:
        return None
    
    N = len(node_names)
    X_train = torch.stack([d.x.flatten() for d in train_anom])
    y_train = torch.tensor([name_to_idx[d.root_cause[0]] for d in train_anom], dtype=torch.long)
    
    input_dim = X_train.size(1)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, N)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    n = len(train_anom)
    
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            opt.zero_grad()
            loss = crit(model(X_train[idx]), y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    
    model.eval()
    X_test = torch.stack([d.x.flatten() for d in test_anom])
    y_test_labels = [name_to_idx[d.root_cause[0]] for d in test_anom]
    y_true, y_logits = [], []
    with torch.no_grad():
        for s in range(0, len(test_anom), batch_size):
            lo = model(X_test[s:s+batch_size])
            for i in range(lo.size(0)):
                y_true.append(y_test_labels[s+i])
                y_logits.append(lo[i].cpu().numpy())
    
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {'top1': tops[1], 'top2': tops[2], 'top5': tops[5], 'mrr': mrr}


# ── Printing ─────────────────────────────────────────────────────────

def print_table(title, results):
    """Print a formatted results table."""
    has_selection = any('selection_rate' in v for v in results.values() if isinstance(v, dict))
    
    print(f"\n{'=' * 105}")
    print(f"  {title}")
    print(f"{'=' * 105}")
    if has_selection:
        print(f"  {'MODEL':<35} | {'TOP-1':>8} | {'TOP-2':>8} | {'TOP-5':>8} | {'MRR':>8} | {'SELECT':>8}")
        print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    else:
        print(f"  {'MODEL':<35} | {'TOP-1':>8} | {'TOP-2':>8} | {'TOP-5':>8} | {'MRR':>8}")
        print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for name, metrics in results.items():
        if metrics is None:
            print(f"  {name:<35} | {'N/A':>8}")
            continue
        
        if isinstance(metrics, tuple):
            # Multi-seed: (mean_dict, std_dict)
            m, s = metrics
            t1 = f"{m['top1']:.3f}±{s['top1']:.3f}"
            t2 = f"{m['top2']:.3f}±{s['top2']:.3f}"
            t5 = f"{m['top5']:.3f}±{s['top5']:.3f}"
            mr = f"{m['mrr']:.3f}±{s['mrr']:.3f}"
            if has_selection and 'selection_rate' in m:
                sel = f"{m['selection_rate']:.3f}"
                print(f"  {name:<35} | {t1:>8} | {t2:>8} | {t5:>8} | {mr:>8} | {sel:>8}")
            else:
                print(f"  {name:<35} | {t1:>8} | {t2:>8} | {t5:>8} | {mr:>8}")
        else:
            t1 = f"{metrics['top1']:.4f}"
            t2 = f"{metrics['top2']:.4f}"
            t5 = f"{metrics['top5']:.4f}"
            mr = f"{metrics['mrr']:.4f}"
            if has_selection and 'selection_rate' in metrics:
                sel = f"{metrics['selection_rate']:.4f}"
                print(f"  {name:<35} | {t1:>8} | {t2:>8} | {t5:>8} | {mr:>8} | {sel:>8}")
            else:
                print(f"  {name:<35} | {t1:>8} | {t2:>8} | {t5:>8} | {mr:>8}")
    
    print(f"{'=' * 105}\n")


def print_compute_table(results):
    """Print a computation comparison table: latency, params, deep-processed nodes."""
    # Collect entries that have timing info
    entries = []
    for name, metrics in results.items():
        if metrics is None:
            continue
        m = metrics[0] if isinstance(metrics, tuple) else metrics
        s = metrics[1] if isinstance(metrics, tuple) else {}
        if 'mean_latency_ms' in m:
            entries.append((name, m, s))
    
    if not entries:
        return
    
    print(f"\n{'=' * 110}")
    print(f"  COMPUTATION COMPARISON")
    print(f"{'=' * 110}")
    print(f"  {'MODEL':<35} | {'LATENCY (ms)':>14} | {'PARAMS':>10} | {'DEEP NODES':>12} | {'NODE REDUCTION':>14}")
    print(f"  {'-'*35}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}")
    
    # Find the full-graph reference (Pure GCN) for speedup comparison
    ref_latency = None
    ref_name = None
    for name, m, _ in entries:
        dn = m.get('deep_nodes', 0)
        tn = m.get('total_nodes', 0)
        if isinstance(dn, float):
            dn = int(round(dn))
        if isinstance(tn, float):
            tn = int(round(tn))
        if dn == tn and tn > 0:
            ref_latency = m['mean_latency_ms']
            ref_name = name
            break
    
    for name, m, s in entries:
        lat_mean = m['mean_latency_ms']
        lat_std = s.get('mean_latency_ms', m.get('std_latency_ms', 0))
        lat = f"{lat_mean:.2f}±{lat_std:.2f}"
        
        params = int(round(m['params']))
        params_str = f"{params:,}"
        
        deep = m.get('deep_nodes', 0)
        total = m.get('total_nodes', 0)
        if isinstance(deep, float):
            deep = int(round(deep))
        if isinstance(total, float):
            total = int(round(total))
        deep_str = f"{deep}/{total}"
        
        if deep != total and total > 0:
            reduction = f"{(1 - deep / total) * 100:.0f}% fewer"
        else:
            reduction = "0% (full graph)"
        
        print(f"  {name:<35} | {lat:>14} | {params_str:>10} | {deep_str:>12} | {reduction:>14}")
    
    # Latency ratio summary
    if ref_latency and len(entries) > 1:
        print(f"  {'-'*35}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}")
        for name, m, _ in entries:
            dn = m.get('deep_nodes', 0)
            tn = m.get('total_nodes', 0)
            if isinstance(dn, float):
                dn = int(round(dn))
            if isinstance(tn, float):
                tn = int(round(tn))
            if dn != tn:
                ratio = m['mean_latency_ms'] / ref_latency
                label = f"  {name} vs {ref_name}"
                if ratio > 1:
                    print(f"  {label:<35} | {'RATIO':>14} | {'':>10} | {'':>12} | {ratio:>10.2f}x lat")
                else:
                    print(f"  {label:<35} | {'SPEEDUP':>14} | {'':>10} | {'':>12} | {1/ratio:>10.2f}x fast")
    
    print(f"{'=' * 110}\n")


# ── Main ─────────────────────────────────────────────────────────────

def run_benchmark(
    large_dataset_path: str = None,
    small_dataset_path: str = None,
    num_seeds: int = 5,
    skip_part_a: bool = False,
):
    seeds = SEEDS[:num_seeds]
    
    # ── Part A: Legacy 5-node evaluation ──
    if small_dataset_path and not skip_part_a:
        from common import NODE_NAMES as NODE_NAMES_SMALL
        print("\n" + "▓" * 60)
        print("  PART A: Legacy STGCN (5-node dataset)")
        print("▓" * 60)
        
        dataset_small = torch.load(small_dataset_path, weights_only=False)
        results_small = {}
        
        for ckpt, name in [
            ("checkpoints/stgcn_rca_final.pt", "STGCN+GIB Semi-Sup (v1)"),
            ("checkpoints/stgcn_supervised_only.pt", "STGCN+GIB Sup-Only (v1)"),
            ("checkpoints/stgcn_no_gib.pt", "STGCN No-GIB (v1)"),
        ]:
            print(f"\n  Evaluating {name}...")
            r = eval_legacy_stgcn(ckpt, dataset_small, NODE_NAMES_SMALL)
            results_small[name] = r
        
        # Baselines on 5-node (multi-seed)
        _, test_small = stratified_split_generic(dataset_small, NODE_NAMES_SMALL)
        train_small, _ = stratified_split_generic(dataset_small, NODE_NAMES_SMALL)
        
        from baselines import train_pure_gcn, train_random_forest, train_mlp_baseline
        for bname, fn in [
            ("Pure GCN (5-node)", lambda s: train_pure_gcn(train_small, test_small, epochs=150, seed=s)),
            ("MLP (5-node)", lambda s: train_mlp_baseline(train_small, test_small, epochs=150, seed=s)),
            ("RF (5-node)", lambda s: train_random_forest(train_small, test_small, seed=s)),
        ]:
            print(f"\n  {bname} ({num_seeds} seeds)...")
            runs = []
            for s in seeds:
                t1, t2, mrr = fn(s)
                runs.append({'top1': t1, 'top2': t2, 'top5': 1.0, 'mrr': mrr})
            arr_t1 = np.array([r['top1'] for r in runs])
            arr_t2 = np.array([r['top2'] for r in runs])
            arr_mr = np.array([r['mrr'] for r in runs])
            results_small[bname] = (
                {'top1': arr_t1.mean(), 'top2': arr_t2.mean(), 'top5': 1.0, 'mrr': arr_mr.mean()},
                {'top1': arr_t1.std(), 'top2': arr_t2.std(), 'top5': 0.0, 'mrr': arr_mr.std()},
            )
        
        print_table("LEGACY 5-NODE RESULTS", results_small)
    
    # ── Part B: SelectThenAnalyze 20-node evaluation ──
    if large_dataset_path:
        from simulated_data_large import NODE_NAMES_LARGE
        print("\n" + "▓" * 60)
        print("  PART B: SelectThenAnalyze (20-node dataset)")
        print("▓" * 60)
        
        dataset_large = torch.load(large_dataset_path, weights_only=False)
        results_large = {}
        
        for ckpt, name in [
            ("checkpoints/sta_semi_sup.pt", "SelectThenAnalyze Semi-Sup"),
            ("checkpoints/sta_sup_only.pt", "SelectThenAnalyze Sup-Only"),
        ]:
            print(f"\n  Evaluating {name}...")
            r = eval_select_then_analyze(ckpt, dataset_large, NODE_NAMES_LARGE)
            results_large[name] = r
        
        # Baselines on 20-node (multi-seed)
        train_large, test_large = stratified_split_generic(dataset_large, NODE_NAMES_LARGE)
        
        for bname, fn in [
            ("Pure GCN (20-node)", lambda s: train_pure_gcn_large(train_large, test_large, NODE_NAMES_LARGE, epochs=150, seed=s)),
            ("MLP (20-node)", lambda s: train_mlp_large(train_large, test_large, NODE_NAMES_LARGE, epochs=150, seed=s)),
            ("RF (20-node)", lambda s: train_rf_large(train_large, test_large, NODE_NAMES_LARGE, seed=s)),
        ]:
            print(f"\n  {bname} ({num_seeds} seeds)...")
            runs = []
            for s in seeds:
                r = fn(s)
                if r:
                    runs.append(r)
            if runs:
                mean_d = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
                std_d = {k: np.std([r[k] for r in runs]) for k in runs[0]}
                results_large[bname] = (mean_d, std_d)
            else:
                results_large[bname] = None
        
        print_table("20-NODE RESULTS (SelectThenAnalyze vs Baselines)", results_large)
        print_compute_table(results_large)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark v2: Legacy vs SelectThenAnalyze")
    parser.add_argument("--large-dataset", type=str, default=None,
                        help="Path to 20-node dataset .pt")
    parser.add_argument("--small-dataset", type=str, default=None,
                        help="Path to 5-node dataset .pt (legacy)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--skip-part-a", action="store_true",
                        help="Skip Part A (legacy 5-node evaluation)")
    args = parser.parse_args()
    
    run_benchmark(
        large_dataset_path=args.large_dataset,
        small_dataset_path=args.small_dataset,
        num_seeds=args.seeds,
        skip_part_a=args.skip_part_a,
    )
