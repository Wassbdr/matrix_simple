#!/usr/bin/env python3
"""
End-to-end validation & benchmark comparing two RCA approaches:

    - Late Bottleneck (STGCN+GIB): STGCN + GIB applied after full-graph processing.
      Processes all N nodes through the full pipeline, then applies the GIB
      mask to isolate the root cause. Complexity: O(N²).

    - STA (Select-then-Analyze): Early subgraph selection via STE selector,
      followed by deep analysis on the k selected nodes only.
      Complexity: O(N + k²), advantageous when k ≪ N.

Both approaches are valid and complementary:
    - Late Bottleneck excels on small/medium graphs where full processing is cheap.
    - STA is designed for large-scale topologies where O(N²) becomes
      prohibitive and early selection provides computational savings.

Pipeline:
    1. Generate simulated BA(N, m) datasets
    2. Train Late Bottleneck and STA on each (multi-seed)
    3. Evaluate: Top-1/2/5 accuracy, MRR, selection rate, latency
    4. Compare against baselines: full-graph GCN, MLP, Random Forest
    5. Latency scaling analysis: O(k²) vs O(N²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import os
import sys
import warnings
from collections import Counter
from typing import List, Dict, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

# Local imports
from model_sta import SelectThenAnalyze
from model import SpatioTemporalRCA
from losses import info_nce_loss, asymmetric_gib_loss
from augment import augment_positive

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =====================================================================
# Utilities
# =====================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_ranking_metrics(y_true_list, y_pred_logits_list, top_ns=(1, 2, 5)):
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


def stratified_split(dataset, name_to_idx, train_ratio=0.8):
    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and d.root_cause[0] in name_to_idx]
    h_split = int(len(healthy) * train_ratio)
    train = healthy[:h_split]
    test = healthy[h_split:]
    by_rc: Dict[str, list] = {}
    for d in anomalous:
        by_rc.setdefault(d.root_cause[0], []).append(d)
    for rc, items in by_rc.items():
        s = max(1, int(len(items) * train_ratio))
        train.extend(items[:s])
        test.extend(items[s:])
    return train, test


def get_anomalies(data_list, name_to_idx):
    return [d for d in data_list
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and d.root_cause[0] in name_to_idx]


# =====================================================================
# Normalization
# =====================================================================

def compute_norm_stats(data_list):
    """Compute per-feature mean/std across all nodes and samples.
    x shape per sample: [N, F, W]. Returns (mean, std) each [1, F, 1]."""
    all_x = torch.cat([d.x for d in data_list], dim=0)  # [total_nodes, F, W]
    mean = all_x.mean(dim=(0, 2), keepdim=True)  # [1, F, 1]
    std = all_x.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
    return mean, std


def normalize_dataset(data_list, mean, std):
    """In-place z-score normalization on x tensors."""
    for d in data_list:
        d.x = (d.x - mean) / std
    return data_list


# =====================================================================
# Early-stopping helper
# =====================================================================

class EarlyStopping:
    """Patience-based early stopping. Call .step(val_loss) each epoch."""
    def __init__(self, patience=15):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =====================================================================
# Baselines
# =====================================================================

class FullGraphGCN(nn.Module):
    """Full-graph GCN baseline (no subgraph selection) -- O(N^2)."""
    def __init__(self, num_features, hidden_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.tc1 = nn.Conv1d(num_features, hidden_dim, 3, padding=1)
        self.tc2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.tn = nn.BatchNorm1d(hidden_dim)
        self.ta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                                nn.Linear(hidden_dim // 2, 1))
        self.g1 = GCNConv(hidden_dim, hidden_dim)
        self.g2 = GCNConv(hidden_dim, hidden_dim)
        self.sn = nn.LayerNorm(hidden_dim)
        self.sc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x, ei):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, V, Ff, W = x.shape
        h = x.view(B * V, Ff, W)
        h = F.relu(self.tc1(h))
        h = F.relu(self.tc2(h))
        ht = h.permute(0, 2, 1)
        a = F.softmax(self.ta(ht).squeeze(-1), dim=-1)
        h = (ht * a.unsqueeze(-1)).sum(1)
        h = self.tn(h)
        if B > 1:
            off = torch.arange(B, device=h.device) * V
            bei = ei.unsqueeze(0).expand(B, -1, -1) + off.view(-1, 1, 1)
            bei = bei.reshape(2, -1)
        else:
            bei = ei
        r = h
        h = F.relu(self.g1(h, bei))
        h = self.g2(h, bei)
        h = self.sn(h + r)
        h = h.view(B, V, -1)
        return self.sc(h).squeeze(-1)


def train_baseline_gcn(train_anom, edge_index, name_to_idx, N, F_, hidden_dim=32,
                       epochs=150, batch_size=256, seed=42):
    torch.manual_seed(seed)
    model = FullGraphGCN(F_, hidden_dim, N)
    X = torch.stack([d.x for d in train_anom])
    y = torch.tensor([name_to_idx[d.root_cause[0]] for d in train_anom], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(train_anom))
        for s in range(0, len(train_anom), batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            lo = model(X[idx], edge_index)
            loss = F.cross_entropy(lo, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def train_baseline_mlp(train_anom, name_to_idx, N, hidden_dim=64, epochs=150,
                       batch_size=256, seed=42):
    torch.manual_seed(seed)
    # Per-node temporal statistics (mean, std, max) instead of raw flatten
    # Reduces input from N*F*W to N*F*3 — much less overfitting
    def extract_features(data_list):
        feats = []
        for d in data_list:
            x = d.x  # [N, F, W]
            stats = torch.cat([x.mean(dim=-1), x.std(dim=-1), x.max(dim=-1).values], dim=-1)  # [N, F*3]
            feats.append(stats.flatten())
        return torch.stack(feats)

    X = extract_features(train_anom)
    y = torch.tensor([name_to_idx[d.root_cause[0]] for d in train_anom], dtype=torch.long)
    input_dim = X.size(1)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.4), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(hidden_dim // 2, N))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(train_anom))
        for s in range(0, len(train_anom), batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def train_baseline_rf(train_anom, name_to_idx, N, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([d.x.numpy().flatten() for d in train_anom])
    y = np.array([name_to_idx[d.root_cause[0]] for d in train_anom])
    clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 class_weight="balanced", max_depth=20,
                                 min_samples_leaf=5)
    clf.fit(X, y)
    return clf


# =====================================================================
# Late Bottleneck Training & Eval (STGCN + GIB)
# Parallel approach to STA: processes full graph then masks via GIB.
# Preferred for small/medium topologies where O(N²) is acceptable.
# =====================================================================

def train_v1_compact(
    train_data, node_names, edge_index, device,
    hidden_dim=32,
    contrastive_epochs=80, supervised_epochs=150,
    lr=1e-3, batch_size=256,
    nce_temperature=0.07, beta_max=0.5, warmup_epochs=40,
):
    """
    Train V1 (SpatioTemporalRCA) with the same two-phase approach as train.py:
    Phase 1: Contrastive + asymmetric GIB (target_sparsity = 1/N)
    Phase 2: Supervised CE on raw GIB logits (not BCE on sigmoid probs)
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)

    healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
    anomalous = get_anomalies(train_data, name_to_idx)

    if len(healthy) < 2 or len(anomalous) < 2:
        raise ValueError(f"Not enough data: {len(healthy)} healthy, {len(anomalous)} anomalous")

    x_anchor = torch.stack([healthy[i].x for i in range(len(healthy) - 1)]).to(device)
    x_positive = torch.stack([healthy[i + 1].x for i in range(len(healthy) - 1)]).to(device)
    x_negative = torch.stack([d.x for d in anomalous]).to(device)
    ei = edge_index.to(device)
    num_features = x_anchor.size(2)

    # Dynamic target_sparsity = 1/N (not hardcoded 0.2 which was tuned for N=5)
    target_sparsity = 1.0 / N

    model = SpatioTemporalRCA(
        num_features=num_features, hidden_dim=hidden_dim,
        num_nodes=N, gib_temperature=1.0,
    ).to(device)

    # ---- Phase 1: Contrastive + Asymmetric GIB ----
    if contrastive_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=contrastive_epochs, eta_min=1e-5)
        n_h, n_n = x_anchor.size(0), x_negative.size(0)
        temp_start, temp_end = 1.0, 0.3

        for epoch in range(contrastive_epochs):
            model.train()
            beta = min(1.0, epoch / max(warmup_epochs, 1)) * beta_max
            t_frac = epoch / max(contrastive_epochs - 1, 1)
            model.gib.set_temperature(temp_start + t_frac * (temp_end - temp_start))

            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]

                optimizer.zero_grad()
                z_a, probs_a, _ = model(x_anchor[idx_h], ei, use_gib=True)
                x_pos_aug = augment_positive(x_positive[idx_h], noise_sigma=0.05,
                                             jitter_shift=1, feature_mask_ratio=0.1)
                z_p, probs_p, _ = model(x_pos_aug, ei, use_gib=True)
                z_n, probs_n, _ = model(x_negative[idx_n], ei, use_gib=True)

                l_nce = info_nce_loss(z_a, z_p, z_n, nce_temperature)
                probs_healthy = torch.cat([probs_a, probs_p], dim=0)
                l_gib = asymmetric_gib_loss(probs_healthy, probs_n, target_sparsity)
                loss = l_nce + beta * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

        print(f"    [V1-Phase1] {contrastive_epochs} contrastive epochs done | last_loss={loss.item():.4f}")

    # ---- Phase 2: Supervised CE on raw GIB logits ----
    # KEY FIX: BCE on sigmoid probs collapses at N>5 because the 19:1
    # negative-to-positive gradient ratio pushes all probs → 0.
    # CE on raw logits normalizes via softmax and works at any N.
    x_sup = x_negative
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous], dtype=torch.long, device=device)
    n_sup = x_sup.size(0)

    # Hold out 5% for early stopping validation
    n_val = max(1, int(n_sup * 0.05))
    perm_split = torch.randperm(n_sup, device=device)
    val_idx = perm_split[:n_val]
    train_idx_sup = perm_split[n_val:]
    x_val, y_val = x_sup[val_idx], y_sup[val_idx]
    x_train_sup, y_train_sup = x_sup[train_idx_sup], y_sup[train_idx_sup]
    n_train = x_train_sup.size(0)

    # Full LR for all params -- backbone needs to adapt for N-way discrimination
    opt_sup = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch_sup = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sup, T_max=supervised_epochs, eta_min=1e-5)
    es = EarlyStopping(patience=30)

    for epoch in range(supervised_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss = ep_correct = 0
        nb = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x_b, y_b = x_train_sup[idx], y_train_sup[idx]
            B = x_b.size(0)
            opt_sup.zero_grad()

            # Forward through temporal + spatial blocks
            x_flat = x_b.view(B * N, num_features, -1)
            h_temporal = model.temporal_block(x_flat)
            if B > 1:
                offsets = torch.arange(B, device=device) * N
                batch_ei = ei.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
                batch_ei = batch_ei.reshape(2, -1)
            else:
                batch_ei = ei
            h_spatial = model.spatial_block(h_temporal, batch_ei)

            # Raw GIB logits (before sigmoid) — proper N-way classification
            gib_logits = model.gib.mlp(h_spatial).squeeze(-1).view(B, N)
            loss = F.cross_entropy(gib_logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_sup.step()

            preds = gib_logits.argmax(dim=1)
            ep_loss += loss.item()
            ep_correct += (preds == y_b).sum().item()
            nb += 1

        sch_sup.step()
        if epoch % 30 == 0 or epoch == supervised_epochs - 1:
            print(f"    [V1-Phase2] Ep {epoch:03d}/{supervised_epochs} | loss={ep_loss/nb:.4f} | "
                  f"acc={ep_correct/n_train:.4f}")

        # Early stopping on validation set
        model.eval()
        with torch.no_grad():
            B_v = x_val.size(0)
            x_flat_v = x_val.view(B_v * N, num_features, -1)
            h_t_v = model.temporal_block(x_flat_v)
            if B_v > 1:
                offsets_v = torch.arange(B_v, device=device) * N
                batch_ei_v = ei.unsqueeze(0).expand(B_v, -1, -1) + offsets_v.view(-1, 1, 1)
                batch_ei_v = batch_ei_v.reshape(2, -1)
            else:
                batch_ei_v = ei
            h_s_v = model.spatial_block(h_t_v, batch_ei_v)
            gib_logits_v = model.gib.mlp(h_s_v).squeeze(-1).view(B_v, N)
            val_loss = F.cross_entropy(gib_logits_v, y_val).item()
        if es.step(val_loss, model):
            print(f"    [V1-Phase2] Early stop at epoch {epoch} (patience={es.patience})")
            break

    es.restore(model)
    model.eval()
    return model


def eval_v1(model, test_anom, edge_index, name_to_idx, device):
    """Evaluate V1 (STGCN+GIB) -- use raw GIB logits (not sigmoid probs) for ranking."""
    N = model.num_nodes
    y_true, y_logits = [], []
    ei = edge_index.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(test_anom[0].x.unsqueeze(0).to(device), ei, use_gib=True)

    latencies = []
    with torch.no_grad():
        for data in test_anom:
            x = data.x.unsqueeze(0).to(device)
            true_idx = name_to_idx[data.root_cause[0]]

            t0 = time.perf_counter()
            # Forward through sub-modules to get raw GIB logits
            x_flat = x.view(N, x.size(2), x.size(3))
            h_temporal = model.temporal_block(x_flat)
            h_spatial = model.spatial_block(h_temporal, ei)
            gib_logits = model.gib.mlp(h_spatial).squeeze(-1)  # [N]
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            y_true.append(true_idx)
            y_logits.append(gib_logits.cpu().numpy())

    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model), "k": N, "N": N,
    }


# =====================================================================
# STA Training (compact -- both phases)
# =====================================================================

def train_sta_compact(
    train_data, node_names, edge_index, device,
    hidden_dim=32, k=5,
    warmup_epochs=20, supervised_epochs=150,
    lr=1e-3, batch_size=256,
    nce_temperature=0.07,
    k_anneal_epochs=60,
):
    """
    Train STA with k-annealing curriculum.

    Phase 1 (short): Contrastive warmup for the light encoder only (20 epochs).
    Phase 2 (main):  Supervised with k-annealing.
        - k starts at min(N//2, N-1) and linearly anneals to k_target over
          k_anneal_epochs.  This gives the scorer a ~50% chance of seeing the
          root cause from the start, breaking the chicken-and-egg deadlock.
        - Primary loss: CE on selector_logits (all N nodes; teaches WHICH node).
        - Secondary loss: CE on scorer (only when RC in top-k; refines ranking).
        - No L1 sparsity (it fights the CE objective).
        - Temperature annealing: 1.0 -> 0.3 over the full phase.
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)

    healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
    anomalous = get_anomalies(train_data, name_to_idx)

    if len(healthy) < 2 or len(anomalous) < 2:
        raise ValueError(f"Not enough data: {len(healthy)} healthy, {len(anomalous)} anomalous")

    x_anchor = torch.stack([healthy[i].x for i in range(len(healthy) - 1)]).to(device)
    x_positive = torch.stack([healthy[i + 1].x for i in range(len(healthy) - 1)]).to(device)
    x_negative = torch.stack([d.x for d in anomalous]).to(device)
    ei = edge_index.to(device)
    num_features = x_anchor.size(2)

    model = SelectThenAnalyze(
        num_features=num_features, hidden_dim=hidden_dim,
        num_nodes=N, k=k, gib_temperature=1.0,
    ).to(device)

    # ---- Phase 1: Short contrastive warmup (encoder only) ----
    if warmup_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        n_h, n_n = x_anchor.size(0), x_negative.size(0)

        for epoch in range(warmup_epochs):
            model.train()
            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                B_h, B_n = idx_h.size(0), idx_n.size(0)

                optimizer.zero_grad()
                bei_h = model._batch_edge_index(ei, B_h)
                bei_n = model._batch_edge_index(ei, B_n)
                z_a = model.light_encoder(x_anchor[idx_h].view(B_h * N, num_features, -1), bei_h).view(B_h, N, -1).mean(1)
                x_pos_aug = augment_positive(x_positive[idx_h], noise_sigma=0.05,
                                             jitter_shift=1, feature_mask_ratio=0.1)
                z_p = model.light_encoder(x_pos_aug.view(B_h * N, num_features, -1), bei_h).view(B_h, N, -1).mean(1)
                z_n = model.light_encoder(x_negative[idx_n].view(B_n * N, num_features, -1), bei_n).view(B_n, N, -1).mean(1)

                loss = info_nce_loss(z_a, z_p, z_n, nce_temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        print(f"    [Phase1] {warmup_epochs} warmup epochs done | last_nce={loss.item():.4f}")

    # ---- Phase 2: Supervised with k-annealing ----
    x_sup = x_negative
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous], dtype=torch.long, device=device)
    n_sup = x_sup.size(0)

    # Hold out 5% for early stopping validation
    n_val = max(1, int(n_sup * 0.05))
    perm_split = torch.randperm(n_sup, device=device)
    val_idx = perm_split[:n_val]
    train_idx_sta = perm_split[n_val:]
    x_val_sta, y_val_sta = x_sup[val_idx], y_sup[val_idx]
    x_train_sta, y_train_sta = x_sup[train_idx_sta], y_sup[train_idx_sta]
    n_train_sta = x_train_sta.size(0)

    # Full LR for all parameters (encoder needs to adapt fully for node selection)
    opt_sup = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch_sup = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sup, T_max=supervised_epochs, eta_min=1e-5)
    es = EarlyStopping(patience=30)

    k_warm = min(N // 2, N - 1)  # start with large k
    k_target = k

    for epoch in range(supervised_epochs):
        model.train()

        # k-annealing: linearly decrease from k_warm to k_target
        if epoch < k_anneal_epochs:
            k_cur = k_warm - int((k_warm - k_target) * epoch / k_anneal_epochs)
        else:
            k_cur = k_target
        model.set_k(k_cur)

        # Temperature annealing: 1.0 -> 0.3
        t_frac = epoch / max(supervised_epochs - 1, 1)
        model.set_temperature(1.0 + t_frac * (0.3 - 1.0))

        perm = torch.randperm(n_train_sta, device=device)
        ep_loss = ep_correct = ep_selected = 0
        nb = 0

        for start in range(0, n_train_sta, batch_size):
            idx = perm[start:start + batch_size]
            x_b, y_b = x_train_sta[idx], y_train_sta[idx]
            opt_sup.zero_grad()

            logits_full, M_soft, top_k, sel_logits = model(x_b, ei)

            # Primary: CE on selector logits (covers all N — main learning signal)
            loss_sel = F.cross_entropy(sel_logits, y_b)

            # Secondary: CE on scorer output (only when RC in top-k)
            in_topk = (top_k == y_b.unsqueeze(1)).any(dim=1)
            if in_topk.any():
                loss_scorer = F.cross_entropy(logits_full[in_topk], y_b[in_topk])
            else:
                loss_scorer = torch.tensor(0.0, device=device)

            loss = loss_sel + loss_scorer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_sup.step()

            preds = sel_logits.argmax(dim=1)
            ep_loss += loss.item()
            ep_correct += (preds == y_b).sum().item()
            ep_selected += in_topk.sum().item()
            nb += 1

        sch_sup.step()
        if epoch % 30 == 0 or epoch == supervised_epochs - 1:
            print(f"    [Phase2] Ep {epoch:03d}/{supervised_epochs} | loss={ep_loss/nb:.4f} | "
                  f"acc={ep_correct/n_train_sta:.4f} | RC-in-top-{k_cur}={ep_selected/n_train_sta:.4f} | k={k_cur}")

        # Early stopping on validation set (only after k-annealing completes)
        if epoch >= k_anneal_epochs:
            model.eval()
            with torch.no_grad():
                vl_full, _, _, vl_sel = model(x_val_sta, ei)
                val_loss = F.cross_entropy(vl_sel, y_val_sta).item()
            if es.step(val_loss, model):
                print(f"    [Phase2] Early stop at epoch {epoch} (patience={es.patience})")
                break

    # Reset to target k for inference
    es.restore(model)
    model.set_k(None)
    model.eval()
    return model


# =====================================================================
# Evaluation
# =====================================================================

def eval_sta(model, test_anom, edge_index, name_to_idx, device):
    """Evaluate STA model -- returns metrics dict."""
    N = model.num_nodes
    k = model.k
    y_true = []
    y_logits_combined = []  # selector + scorer ensemble
    y_logits_selector = []  # selector only (for ablation)
    selection_hits = 0
    ei = edge_index.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(test_anom[0].x.unsqueeze(0).to(device), ei)

    latencies = []
    with torch.no_grad():
        for data in test_anom:
            x = data.x.unsqueeze(0).to(device)
            true_idx = name_to_idx[data.root_cause[0]]

            t0 = time.perf_counter()
            logits_full, M_soft, top_k, sel_logits = model(x, ei)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            # Primary signal: selector logits (trained as N-way classifier)
            sel = sel_logits.squeeze(0)  # [N]

            # Boost: for the k selected nodes, add the scorer's opinion
            # scorer logits are -inf for non-selected, finite for selected
            scorer = logits_full.squeeze(0)  # [N]
            # Replace -inf with 0 for safe addition, then scale
            scorer_safe = scorer.clone()
            scorer_safe[scorer == float('-inf')] = 0.0
            combined = sel + 0.5 * scorer_safe

            y_true.append(true_idx)
            y_logits_combined.append(combined.cpu().numpy())
            y_logits_selector.append(sel.cpu().numpy())

            if true_idx in top_k[0].tolist():
                selection_hits += 1

    tops, mrr = get_ranking_metrics(y_true, y_logits_combined, top_ns=(1, 2, 5))
    tops_sel, mrr_sel = get_ranking_metrics(y_true, y_logits_selector, top_ns=(1, 2, 5))
    return {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "sel_top1": tops_sel[1], "sel_top2": tops_sel[2], "sel_top5": tops_sel[5],
        "sel_mrr": mrr_sel,
        "selection_rate": selection_hits / len(y_true),
        "n_test": len(y_true),
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model),
        "k": k, "N": N,
    }


def eval_gcn_baseline(model, test_anom, edge_index, name_to_idx, N):
    y_true, y_logits = [], []
    X_test = torch.stack([d.x for d in test_anom])
    # Warmup
    with torch.no_grad():
        _ = model(X_test[:1], edge_index)
    latencies = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_anom):
            x = data.x.unsqueeze(0)
            t0 = time.perf_counter()
            lo = model(x, edge_index)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            if lo.dim() == 1:
                lo = lo.unsqueeze(0)
            y_true.append(name_to_idx[data.root_cause[0]])
            y_logits.append(lo[0].cpu().numpy())
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model), "k": N, "N": N,
    }


def eval_mlp_baseline(model, test_anom, name_to_idx, N):
    def extract_features(data_list):
        feats = []
        for d in data_list:
            x = d.x
            stats = torch.cat([x.mean(dim=-1), x.std(dim=-1), x.max(dim=-1).values], dim=-1)
            feats.append(stats.flatten())
        return torch.stack(feats)

    y_true, y_logits = [], []
    model.eval()
    with torch.no_grad():
        for data in test_anom:
            x = data.x
            stats = torch.cat([x.mean(dim=-1), x.std(dim=-1), x.max(dim=-1).values], dim=-1)
            lo = model(stats.flatten().unsqueeze(0))
            y_true.append(name_to_idx[data.root_cause[0]])
            y_logits.append(lo[0].cpu().numpy())
    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    return {"top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr, "params": count_parameters(model)}


def eval_rf_baseline(clf, test_anom, name_to_idx, N):
    X_test = np.array([d.x.numpy().flatten() for d in test_anom])
    y_test = [name_to_idx[d.root_cause[0]] for d in test_anom]
    probs = clf.predict_proba(X_test)
    logits = np.zeros((len(X_test), N))
    for i, c in enumerate(clf.classes_):
        logits[:, c] = probs[:, i]
    tops, mrr = get_ranking_metrics(y_test, logits, top_ns=(1, 2, 5))
    return {"top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr}


# =====================================================================
# Printing
# =====================================================================

def print_header(title, char="=", width=110):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_results_table(results: Dict[str, dict], title: str):
    print_header(title)
    has_lat = any("mean_latency_ms" in v for v in results.values() if v)
    has_sel = any("selection_rate" in v for v in results.values() if v)
    has_std = any("top1_std" in v for v in results.values() if v)

    hdr = f"  {'MODEL':<38} | {'TOP-1':>7} | {'TOP-2':>7} | {'TOP-5':>7} | {'MRR':>7}"
    sep = f"  {'-'*38}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}"
    if has_sel:
        hdr += f" | {'SEL%':>7}"
        sep += f"-+-{'-'*7}"
    if has_lat:
        hdr += f" | {'LAT(ms)':>9} | {'PARAMS':>9}"
        sep += f"-+-{'-'*9}-+-{'-'*9}"
    print(hdr)
    print(sep)

    for name, m in results.items():
        if m is None:
            print(f"  {name:<38} | {'N/A':>7}")
            continue
        if has_std and "top1_std" in m:
            row = (f"  {name:<38} | {m['top1']:.4f}" + (f"\u00b1{m['top1_std']:.3f}" if m.get('top1_std') else "") +
                   f" | {m['top2']:.4f}" + (f"\u00b1{m['top2_std']:.3f}" if m.get('top2_std') else "") +
                   f" | {m['top5']:.4f}" + (f"\u00b1{m['top5_std']:.3f}" if m.get('top5_std') else "") +
                   f" | {m['mrr']:.4f}" + (f"\u00b1{m['mrr_std']:.3f}" if m.get('mrr_std') else ""))
        else:
            row = f"  {name:<38} | {m['top1']:>7.4f} | {m['top2']:>7.4f} | {m['top5']:>7.4f} | {m['mrr']:>7.4f}"
        if has_sel:
            sel = m.get("selection_rate", 0)
            row += f" | {sel:>7.4f}"
        if has_lat:
            lat = m.get("mean_latency_ms", 0)
            par = m.get("params", 0)
            row += f" | {lat:>9.2f} | {par:>9,}"
        print(row)
    print(f"  {'-'*38}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")


def print_latency_scaling(scaling_results):
    print_header("LATENCY SCALING ANALYSIS: O(N^2) vs O(k^2)")
    print(f"  {'N':>5} | {'k':>5} | {'STA (ms)':>10} | {'FullGCN (ms)':>12} | {'Speedup':>8} | {'Node reduction':>15}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*15}")
    for r in scaling_results:
        reduction = f"{(1 - r['k']/r['N'])*100:.0f}%"
        speedup = r['gcn_lat'] / r['v3_lat'] if r['v3_lat'] > 0 else 0
        print(f"  {r['N']:>5} | {r['k']:>5} | {r['v3_lat']:>10.2f} | {r['gcn_lat']:>12.2f} | {speedup:>7.2f}x | {reduction:>15}")
    print()


# =====================================================================
# Multi-seed aggregation
# =====================================================================

def aggregate_seeds(runs: List[dict]) -> dict:
    """Aggregate metrics over multiple seed runs into mean ± std."""
    if len(runs) == 1:
        return runs[0]
    keys = ["top1", "top2", "top5", "mrr", "selection_rate", "mean_latency_ms"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    # Copy non-metric fields from first run
    for k in ["params", "k", "N", "n_test"]:
        if k in runs[0]:
            agg[k] = runs[0][k]
    return agg


def parse_int_list(raw: str) -> List[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Empty list is not allowed")
    return [int(v) for v in vals]


def parse_int_map(raw: str) -> Dict[int, int]:
    """Parse 'key:value,key:value' into dict[int, int]."""
    out: Dict[int, int] = {}
    items = [i.strip() for i in raw.split(",") if i.strip()]
    if not items:
        raise ValueError("Empty map is not allowed")
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid map item '{item}'. Expected key:value")
        k_s, v_s = item.split(":", 1)
        out[int(k_s.strip())] = int(v_s.strip())
    return out


def build_configs(
    sizes: List[int],
    ba_m_map: Dict[int, int],
    k_map: Dict[int, int],
    num_samples: int,
    hidden_dim: int,
) -> List[Dict[str, int]]:
    cfgs: List[Dict[str, int]] = []
    for n in sizes:
        if n not in ba_m_map:
            raise ValueError(f"Missing ba_m for N={n}. Provide it in --ba-m-map")
        if n not in k_map:
            raise ValueError(f"Missing k for N={n}. Provide it in --k-map")
        cfgs.append({
            "N": n,
            "ba_m": ba_m_map[n],
            "num_samples": num_samples,
            "k": k_map[n],
            "hidden": hidden_dim,
        })
    return cfgs


# =====================================================================
# Main
# =====================================================================

def main(args):
    from simulated_data_scalable import generate_scalable_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    SEEDS = parse_int_list(args.seeds)
    NUM_SEEDS = len(SEEDS)

    configs = build_configs(
        sizes=parse_int_list(args.sizes),
        ba_m_map=parse_int_map(args.ba_m_map),
        k_map=parse_int_map(args.k_map),
        num_samples=args.num_samples,
        hidden_dim=args.hidden_dim,
    )

    all_scaling = []

    for cfg in configs:
        N = cfg["N"]
        k = cfg["k"]
        hidden = cfg["hidden"]

        print_header(f"GRAPH SIZE N={N}, k={k}, samples={cfg['num_samples']}, seeds={NUM_SEEDS}", char="#")

        # ---- 1. Generate dataset (fixed seed for fair comparison) ----
        print(f"\n  Generating BA({N}, m={cfg['ba_m']}) dataset...")
        t0 = time.time()
        dataset, node_names, edge_index = generate_scalable_dataset(
            num_nodes=N, ba_m=cfg["ba_m"], num_samples=cfg["num_samples"],
            window_size=args.window_size,
            num_features=args.num_features,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.dataset_seed,
        )
        print(f"  Generated {len(dataset)} samples in {time.time()-t0:.1f}s")

        name_to_idx = {n: i for i, n in enumerate(node_names)}
        train_data, test_data = stratified_split(dataset, name_to_idx)

        # NOTE: Normalization disabled — simulated data already has controlled scale.

        train_anom = get_anomalies(train_data, name_to_idx)
        test_anom = get_anomalies(test_data, name_to_idx)
        F_ = dataset[0].x.size(1)

        rc_dist = Counter(d.root_cause[0] for d in train_anom)
        print(f"  Train: {len(train_data)} ({len(train_anom)} anomalies, {len(rc_dist)} RC classes)")
        print(f"  Test:  {len(test_data)} ({len(test_anom)} anomalies)")

        # Collect multi-seed results
        sta_runs, late_runs, gcn_runs, mlp_runs, rf_runs = [], [], [], [], []

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  === Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed}) ===")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # ---- STA ----
            print(f"\n  -- Training SelectThenAnalyze (k={k}) --")
            t0 = time.time()
            sta_model = train_sta_compact(
                train_data, node_names, edge_index, device,
                hidden_dim=hidden, k=k,
                warmup_epochs=args.sta_warmup_epochs,
                supervised_epochs=args.supervised_epochs,
                batch_size=args.batch_size,
                k_anneal_epochs=args.sta_k_anneal_epochs,
            )
            print(f"  STA training completed in {time.time()-t0:.1f}s")
            sta_runs.append(eval_sta(sta_model, test_anom, edge_index, name_to_idx, device))

            # ---- Late Bottleneck (parallel approach) ----
            print(f"\n  -- Training STGCN+GIB (Late Bottleneck) --")
            t0 = time.time()
            late_model = train_v1_compact(
                train_data, node_names, edge_index, device,
                hidden_dim=hidden,
                contrastive_epochs=args.late_contrastive_epochs,
                supervised_epochs=args.supervised_epochs,
                batch_size=args.batch_size,
            )
            print(f"  Late Bottleneck training completed in {time.time()-t0:.1f}s")
            late_runs.append(eval_v1(late_model, test_anom, edge_index, name_to_idx, device))

            # ---- Baselines ----
            print(f"\n  -- Training Full-Graph GCN baseline --")
            gcn_model = train_baseline_gcn(train_anom, edge_index, name_to_idx, N, F_,
                                           hidden_dim=hidden,
                                           epochs=args.baseline_epochs,
                                           batch_size=args.batch_size,
                                           seed=seed)
            gcn_model.eval()
            gcn_runs.append(eval_gcn_baseline(gcn_model, test_anom, edge_index, name_to_idx, N))

            print(f"  -- Training MLP baseline --")
            mlp_model = train_baseline_mlp(train_anom, name_to_idx, N, hidden_dim=64,
                                           epochs=args.baseline_epochs,
                                           batch_size=args.batch_size,
                                           seed=seed)
            mlp_runs.append(eval_mlp_baseline(mlp_model, test_anom, name_to_idx, N))

            print(f"  -- Training RF baseline --")
            rf_model = train_baseline_rf(train_anom, name_to_idx, N, seed=seed)
            rf_runs.append(eval_rf_baseline(rf_model, test_anom, name_to_idx, N))

        # ---- Aggregate multi-seed results ----
        results = {}
        r_sta = aggregate_seeds(sta_runs)
        results[f"STA Select-then-Analyze (k={k})"] = r_sta

        r_sta_sel = aggregate_seeds([{
            "top1": r.get("sel_top1", 0), "top2": r.get("sel_top2", 0),
            "top5": r.get("sel_top5", 0), "mrr": r.get("sel_mrr", 0),
        } for r in sta_runs])
        results[f"STA Selector-only (k={k})"] = r_sta_sel

        r_late = aggregate_seeds(late_runs)
        results[f"Late Bottleneck STGCN+GIB (N={N})"] = r_late

        r_gcn = aggregate_seeds(gcn_runs)
        results[f"Full-Graph GCN (N={N})"] = r_gcn

        r_mlp = aggregate_seeds(mlp_runs)
        results[f"MLP Baseline"] = r_mlp

        r_rf = aggregate_seeds(rf_runs)
        results[f"Random Forest"] = r_rf

        # ---- Print results ----
        print_results_table(results, f"RESULTS FOR N={N} (mean over {NUM_SEEDS} seeds)")

        # ---- Collect latency for scaling plot ----
        sta_lat = r_sta.get("mean_latency_ms", 0)
        gcn_lat = r_gcn.get("mean_latency_ms", 0)
        all_scaling.append({"N": N, "k": k, "v3_lat": sta_lat, "gcn_lat": gcn_lat,
                            "v3_params": r_sta.get("params", 0), "gcn_params": r_gcn.get("params", 0)})

    # ---- Latency scaling summary ----
    if len(all_scaling) > 1:
        print_latency_scaling(all_scaling)

    print_header("BENCHMARK COMPLETE", char="*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCA multi-scale benchmark")

    # Dataset generation
    parser.add_argument("--sizes", type=str, default="20,50,100",
                        help="Comma-separated graph sizes N")
    parser.add_argument("--ba-m-map", type=str, default="20:2,50:3,100:3",
                        help="BA m per N as N:m,N:m")
    parser.add_argument("--k-map", type=str, default="20:5,50:8,100:10",
                        help="STA k per N as N:k,N:k")
    parser.add_argument("--num-samples", type=int, default=3000,
                        help="Number of windows per generated dataset")
    parser.add_argument("--window-size", type=int, default=12,
                        help="Temporal window size")
    parser.add_argument("--num-features", type=int, default=5,
                        help="Number of node features")
    parser.add_argument("--anomaly-ratio", type=float, default=0.4,
                        help="Anomaly ratio in generated dataset")
    parser.add_argument("--dataset-seed", type=int, default=42,
                        help="Seed used for dataset generation")

    # Training/eval settings
    parser.add_argument("--seeds", type=str, default="42,123,7",
                        help="Comma-separated training seeds")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden dimension for graph models")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--supervised-epochs", type=int, default=150,
                        help="Supervised epochs for STA and Late BN")
    parser.add_argument("--late-contrastive-epochs", type=int, default=80,
                        help="Contrastive epochs for Late BN")
    parser.add_argument("--sta-warmup-epochs", type=int, default=20,
                        help="Warmup contrastive epochs for STA")
    parser.add_argument("--sta-k-anneal-epochs", type=int, default=60,
                        help="k-annealing epochs for STA")
    parser.add_argument("--baseline-epochs", type=int, default=150,
                        help="Training epochs for GCN/MLP baselines")

    main(parser.parse_args())
