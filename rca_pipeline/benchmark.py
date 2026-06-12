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

import argparse
import json
import time
import warnings
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

# Local imports (package-safe for `python -m rca_pipeline.benchmark`)
import logging

from rca_pipeline.augment import augment_positive
from rca_pipeline.cascade_aware_loss import CausalCascadeAwareLoss
from rca_pipeline.causal_graph_builder import CausalGraphBuilder
from rca_pipeline.common import (
    runtime_env_info,
    set_all_seeds,
    set_torch_deterministic,
    stratified_split_shuffle,
)
from rca_pipeline.drift_detector import make_k8s_drift_detector
from rca_pipeline.failure_type_utils import split_logits_by_failure_type
from rca_pipeline.k_selection import adaptive_gib_beta, get_optimal_k
from rca_pipeline.losses import asymmetric_gib_loss, info_nce_loss, variational_gib_loss
from rca_pipeline.model import SpatioTemporalRCA
from rca_pipeline.model_late_gat import SpatioTemporalRCA_GAT
from rca_pipeline.model_sta import SelectThenAnalyze
from rca_pipeline.sta_fusion_utils import gated_selector_scorer_fusion
from rca_pipeline.training_utils import EarlyStopping

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from rca_pipeline.metrics_rcaeval import RCAEvalMetrics

# =====================================================================
# Reference Top-1 (multi-seed means) for regression alerts
# =====================================================================

# Calibrated from benchmark_full.json (3 seeds: 42, 123, 7) — honest multi-seed means.
# Previous values (0.93–0.98) were from non-reproducible single-run conditions.
REFERENCE_TOP1 = {
    "FullGraphGCN": {20: 0.731, 50: 0.260, 100: 0.253},
    "LateBN_GIB":   {20: 0.702, 50: 0.273, 100: 0.229},
    "STA":          {20: 0.300, 50: 0.183, 100: 0.067},
    "LateBN_GAT":   {20: 0.676, 50: 0.083, 100: 0.188},
}

REFERENCE_TOP1_STD = {
    "FullGraphGCN": {20: 0.030, 50: 0.036, 100: 0.033},
    "LateBN_GIB":   {20: 0.033, 50: 0.029, 100: 0.070},
    "STA":          {20: 0.122, 50: 0.085, 100: 0.031},
    "LateBN_GAT":   {20: 0.051, 50: 0.022, 100: 0.044},
}


def _regression_flag(model_key: str, n: int, top1: float, threshold: float = 0.005) -> bool:
    ref = REFERENCE_TOP1.get(model_key, {}).get(n)
    if ref is None:
        return False
    std = REFERENCE_TOP1_STD.get(model_key, {}).get(n, 0.05)
    return (ref - top1) > max(threshold, 2.0 * std)


def _attach_rcaeval(y_true: List[int], y_logits_list) -> Dict[str, float]:
    m = RCAEvalMetrics().compute(y_true, y_logits_list)
    return {f"rcaeval_{k}": v for k, v in m.items()}


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


def get_anomalies(data_list, name_to_idx):
    return [d for d in data_list
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and d.root_cause[0] in name_to_idx]


def _sample_to_metric_dict(x_tensor):
    """
    Convert sample tensor x [N,F,W] to 6 canonical metrics for drift detector.
    """
    x_np = x_tensor.detach().cpu().numpy()
    agg = x_np.mean(axis=(0, 2)) if x_np.size else np.zeros((0,), dtype=np.float32)
    names = ["cpu_usage", "memory_bytes", "network_rx", "network_tx", "latency_p99", "error_rate"]
    out = {}
    for i, n in enumerate(names):
        out[n] = float(agg[i]) if i < len(agg) else 0.0
    return out


def _apply_drift_filter(data_list, enable_drift=False, delta=0.002, cooldown_steps=30):
    """
    Sequentially updates ADWIN detector and drops samples during cooldown.
    """
    if not enable_drift:
        return data_list
    detector = make_k8s_drift_detector(delta=delta, cooldown_steps=cooldown_steps)
    kept = []
    for d in data_list:
        detector.update_batch(_sample_to_metric_dict(d.x))
        if detector.is_in_cooldown():
            continue
        kept.append(d)
    return kept


# =====================================================================
# Normalization
# =====================================================================

def compute_norm_stats(train_data_list):
    """Compute per-feature mean/std on the TRAINING SET ONLY (no data leakage).
    x shape per sample: [N, F, W]. Returns (mean, std) each [1, F, 1].

    IMPORTANT: only pass train_data here, never the full dataset.
    Apply the returned stats to both train and test via normalize_dataset().
    """
    all_x = torch.cat([d.x for d in train_data_list], dim=0)  # [total_nodes, F, W]
    mean = all_x.mean(dim=(0, 2), keepdim=True)  # [1, F, 1]
    std = all_x.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
    return mean, std


def normalize_dataset(data_list, mean, std):
    """In-place z-score normalization on x tensors (apply train stats to both splits)."""
    for d in data_list:
        d.x = (d.x - mean) / std
    return data_list



# EarlyStopping imported from training_utils


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
    set_all_seeds(seed)
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
    set_all_seeds(seed)
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

def _compute_val_mrr(model, x_val, y_val, ei, device, N, num_features):
    """Compute MRR on validation set for early stopping (higher = better)."""
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
        logits_v = model.gib.mlp(h_s_v).squeeze(-1).view(B_v, N)  # [B_v, N]

    mrr = 0.0
    logits_np = logits_v.cpu().numpy()
    y_np = y_val.cpu().numpy()
    for i in range(B_v):
        ranked = np.argsort(logits_np[i])[::-1]
        rank = int(np.where(ranked == y_np[i])[0][0]) + 1
        mrr += 1.0 / rank
    return mrr / B_v


def _compute_val_mrr_gat(model, x_val, y_val, ei, device):
    """Validation MRR for Late BN GAT (logits [B, N])."""
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x_val, ei)
        logits_np = logits.cpu().numpy()
        y_np = y_val.cpu().numpy()
    mrr = 0.0
    for i in range(len(y_np)):
        ranked = np.argsort(logits_np[i])[::-1]
        rank = int(np.where(ranked == y_np[i])[0][0]) + 1
        mrr += 1.0 / rank
    return mrr / len(y_np)


def train_v1_compact(
    train_data, node_names, edge_index, device,
    hidden_dim=32,
    contrastive_epochs=80, supervised_epochs=150,
    lr=1e-3, batch_size=256,
    nce_temperature=0.07, beta_max=0.5, warmup_epochs=40,
    patience=75,
    restart_seed_offset=1000,
    edge_weights=None,
    cascade_loss_weight=0.0,
):
    """
    Train V1 (SpatioTemporalRCA) with the same two-phase approach as train.py:
    Phase 1: Contrastive + asymmetric GIB (target_sparsity = 1/N)
    Phase 2: Supervised CE on raw GIB logits (not BCE on sigmoid probs)

    Fixes vs original:
    - Adaptive beta: beta_eff = beta_max * sqrt(N_ref / N) prevents excessive
      sparsity at large N (critical for N=100 variance reduction).
    - EarlyStopping monitored on val MRR (patience=75, not 30 on loss).
    - Degenerate convergence detection at epoch 30 with auto-restart.
    - Full seeding via set_all_seeds() at entry.
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)

    # Adaptive GIB beta: ~4.5× smaller at N=100 vs N=20 (see k_selection.adaptive_gib_beta)
    beta_eff = adaptive_gib_beta(beta_max, N)

    def _run_training(seed_offset=0, skip_degenerate_check=False):
        if seed_offset > 0:
            set_all_seeds(torch.initial_seed() % (2**31) + seed_offset)

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

        # ---- Phase 1: Contrastive + Asymmetric GIB (adaptive beta) ----
        if contrastive_epochs > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=contrastive_epochs, eta_min=1e-5)
            n_h, n_n = x_anchor.size(0), x_negative.size(0)
            temp_start, temp_end = 1.0, 0.3

            for epoch in range(contrastive_epochs):
                model.train()
                beta = min(1.0, epoch / max(warmup_epochs, 1)) * beta_eff
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

            print(f"    [V1-Phase1] {contrastive_epochs} epochs done | beta_eff={beta_eff:.4f} | "
                  f"last_loss={loss.item():.4f}")

        # ---- Phase 2: Supervised CE on raw GIB logits ----
        x_sup = x_negative
        y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous],
                              dtype=torch.long, device=device)
        n_sup = x_sup.size(0)

        # Deterministic val split (reproducible with current seed)
        perm_split = torch.randperm(n_sup, device=device)
        n_val = max(1, int(n_sup * 0.05))
        val_idx = perm_split[:n_val]
        train_idx_sup = perm_split[n_val:]
        x_val, y_val = x_sup[val_idx], y_sup[val_idx]
        x_train_sup, y_train_sup = x_sup[train_idx_sup], y_sup[train_idx_sup]
        n_train = x_train_sup.size(0)

        # Warmup cosine LR
        opt_sup = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sch_sup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_sup, T_0=50, T_mult=2, eta_min=1e-5)
        # MRR-based early stopping (patience=75)
        es = EarlyStopping(patience=patience)
        cascade_loss_fn = CausalCascadeAwareLoss()
        ew = edge_weights.to(device) if edge_weights is not None else None

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

                x_flat = x_b.view(B * N, num_features, -1)
                h_temporal = model.temporal_block(x_flat)
                if B > 1:
                    offsets = torch.arange(B, device=device) * N
                    batch_ei = ei.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
                    batch_ei = batch_ei.reshape(2, -1)
                else:
                    batch_ei = ei
                h_spatial = model.spatial_block(h_temporal, batch_ei)

                gib_logits = model.gib.mlp(h_spatial).squeeze(-1).view(B, N)
                loss = F.cross_entropy(gib_logits, y_b)
                if cascade_loss_weight > 0.0:
                    loss = loss + cascade_loss_weight * cascade_loss_fn(gib_logits, y_b, ei, ew)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_sup.step()

                preds = gib_logits.argmax(dim=1)
                ep_loss += loss.item()
                ep_correct += (preds == y_b).sum().item()
                nb += 1

            sch_sup.step()

            train_acc = ep_correct / n_train
            degen_thresh = max(5.0 / N, 0.05)
            if epoch == 50 and train_acc < degen_thresh and not skip_degenerate_check:
                # Keep training; use EarlyStopping (MRR patience) only.
                print(f"    [V1-Phase2] ⚠️  Epoch 50 acc={train_acc:.3f} < {degen_thresh:.3f} — "
                      f"slow/degenerate-looking convergence (no restart).")

            if epoch % 30 == 0 or epoch == supervised_epochs - 1:
                print(f"    [V1-Phase2] Ep {epoch:03d}/{supervised_epochs} | "
                      f"loss={ep_loss/nb:.4f} | acc={train_acc:.4f}")

            # MRR-based early stopping
            val_mrr = _compute_val_mrr(model, x_val, y_val, ei, device, N, num_features)
            if es.step(val_mrr, model):
                print(f"    [V1-Phase2] Early stop at epoch {epoch} "
                      f"(patience={es.patience}, best_mrr={es.best_mrr:.4f})")
                break

        es.restore(model)
        model.eval()
        return model

    # No degenerate-convergence restarts: rely on EarlyStopping only.
    return _run_training(0, False)


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
    out = {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model), "k": N, "N": N,
    }
    out.update(_attach_rcaeval(y_true, y_logits))
    return out


# =====================================================================
# Late Bottleneck GATv2 (no GIB sparsity)
# =====================================================================

def train_gat_compact(
    train_data, node_names, edge_index, device,
    hidden_dim=32,
    contrastive_epochs=80, supervised_epochs=150,
    lr=1e-3, batch_size=256,
    nce_temperature=0.07,
    warmup_epochs=40,
    patience=75,
    restart_seed_offset=1000,
    edge_weights=None,
    cascade_loss_weight=0.0,
):
    """Two-phase training: InfoNCE on pooled embeddings, then supervised CE on logits."""
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)

    def _run_training(seed_offset=0, skip_degenerate_check=False):
        if seed_offset > 0:
            set_all_seeds(torch.initial_seed() % (2**31) + seed_offset)

        healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
        anomalous = get_anomalies(train_data, name_to_idx)
        if len(healthy) < 2 or len(anomalous) < 2:
            raise ValueError(f"Not enough data: {len(healthy)} healthy, {len(anomalous)} anomalous")

        x_anchor = torch.stack([healthy[i].x for i in range(len(healthy) - 1)]).to(device)
        x_positive = torch.stack([healthy[i + 1].x for i in range(len(healthy) - 1)]).to(device)
        x_negative = torch.stack([d.x for d in anomalous]).to(device)
        ei = edge_index.to(device)
        num_features = x_anchor.size(2)

        model = SpatioTemporalRCA_GAT(
            num_features=num_features, hidden_dim=hidden_dim,
            num_nodes=N,
        ).to(device)

        if contrastive_epochs > 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=contrastive_epochs, eta_min=1e-5)
            n_h, n_n = x_anchor.size(0), x_negative.size(0)

            for epoch in range(contrastive_epochs):
                model.train()
                perm_h = torch.randperm(n_h, device=device)
                perm_n = torch.randperm(n_n, device=device)

                for start in range(0, min(n_h, n_n), batch_size):
                    idx_h = perm_h[start:start + batch_size]
                    idx_n = perm_n[start:start + batch_size]

                    optimizer.zero_grad()
                    _, _, h_a = model(x_anchor[idx_h], ei)
                    x_pos_aug = augment_positive(
                        x_positive[idx_h], noise_sigma=0.05,
                        jitter_shift=1, feature_mask_ratio=0.1,
                    )
                    _, _, h_p = model(x_pos_aug, ei)
                    _, _, h_n = model(x_negative[idx_n], ei)

                    z_a = h_a.mean(dim=1)
                    z_p = h_p.mean(dim=1)
                    z_n = h_n.mean(dim=1)

                    l_nce = info_nce_loss(z_a, z_p, z_n, nce_temperature)
                    l_nce.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()

            if contrastive_epochs > 0:
                print(f"    [GAT-Phase1] {contrastive_epochs} contrastive epochs done | last_nce={l_nce.item():.4f}")

        x_sup = x_negative
        y_sup = torch.tensor(
            [name_to_idx[d.root_cause[0]] for d in anomalous],
            dtype=torch.long, device=device,
        )
        n_sup = x_sup.size(0)
        perm_split = torch.randperm(n_sup, device=device)
        n_val = max(1, int(n_sup * 0.05))
        val_idx = perm_split[:n_val]
        train_idx_sup = perm_split[n_val:]
        x_val, y_val = x_sup[val_idx], y_sup[val_idx]
        x_train_sup, y_train_sup = x_sup[train_idx_sup], y_sup[train_idx_sup]
        n_train = x_train_sup.size(0)

        opt_sup = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sch_sup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_sup, T_0=50, T_mult=2, eta_min=1e-5)
        es = EarlyStopping(patience=patience)
        cascade_loss_fn = CausalCascadeAwareLoss()
        ew = edge_weights.to(device) if edge_weights is not None else None

        for epoch in range(supervised_epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            ep_loss = ep_correct = 0.0
            nb = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                x_b, y_b = x_train_sup[idx], y_train_sup[idx]
                B = x_b.size(0)
                opt_sup.zero_grad()
                logits, _, _ = model(x_b, ei)
                loss = F.cross_entropy(logits, y_b)
                if cascade_loss_weight > 0.0:
                    loss = loss + cascade_loss_weight * cascade_loss_fn(logits, y_b, ei, ew)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_sup.step()
                preds = logits.argmax(dim=1)
                ep_loss += loss.item()
                ep_correct += (preds == y_b).sum().item()
                nb += 1

            sch_sup.step()
            train_acc = ep_correct / n_train

            degen_thresh = max(5.0 / N, 0.05)
            if epoch == 50 and train_acc < degen_thresh and not skip_degenerate_check:
                # Keep training; use EarlyStopping (MRR patience) only.
                print(f"    [GAT-Phase2] ⚠️  Epoch 50 acc={train_acc:.3f} < {degen_thresh:.3f} — "
                      f"slow/degenerate-looking convergence (no restart).")

            if epoch % 30 == 0 or epoch == supervised_epochs - 1:
                print(f"    [GAT-Phase2] Ep {epoch:03d}/{supervised_epochs} | "
                      f"loss={ep_loss/nb:.4f} | acc={train_acc:.4f}")

            val_mrr = _compute_val_mrr_gat(model, x_val, y_val, ei, device)
            if es.step(val_mrr, model):
                print(f"    [GAT-Phase2] Early stop at epoch {epoch} "
                      f"(patience={es.patience}, best_mrr={es.best_mrr:.4f})")
                break

        es.restore(model)
        model.eval()
        return model

    # No degenerate-convergence restarts: rely on EarlyStopping only.
    return _run_training(0, False)


def eval_gat(model, test_anom, edge_index, name_to_idx, device):
    """Evaluate Late BN GAT."""
    N = model.num_nodes
    y_true, y_logits = [], []
    ei = edge_index.to(device)

    with torch.no_grad():
        _ = model(test_anom[0].x.unsqueeze(0).to(device), ei)

    latencies = []
    with torch.no_grad():
        for data in test_anom:
            x = data.x.unsqueeze(0).to(device)
            true_idx = name_to_idx[data.root_cause[0]]
            t0 = time.perf_counter()
            logits, _, _ = model(x, ei)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            y_true.append(true_idx)
            y_logits.append(logits.squeeze(0).cpu().numpy())

    tops, mrr = get_ranking_metrics(y_true, y_logits, top_ns=(1, 2, 5))
    out = {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model), "k": N, "N": N,
    }
    out.update(_attach_rcaeval(y_true, y_logits))
    return out


# =====================================================================
# STA Training (compact -- both phases)
# =====================================================================

def train_sta_compact(
    train_data, node_names, edge_index, device,
    hidden_dim=32, k=None,  # None = auto via get_optimal_k(N)
    warmup_epochs=20, supervised_epochs=150,
    lr=1e-3, batch_size=256,
    nce_temperature=0.07,
    k_anneal_epochs=60,
    skip_degenerate_check=False,
    use_gib: bool = True,
    mask_weighted_nce: bool = True,
    selector_type: str = "ste",
    temporal_encoder: str = "conv",
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

    # Use get_optimal_k if k not explicitly provided
    if k is None:
        k = get_optimal_k(N)
    print(f"    [STA] N={N}, k={k} (empirical optimal={get_optimal_k(N)})")

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
        selector_type=selector_type, temporal_encoder=temporal_encoder,
    ).to(device)

    # ---- Phase 1: Contrastive warmup (InfoNCE on Z_subgraph + variational GIB) ----
    if warmup_epochs > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        n_h, n_n = x_anchor.size(0), x_negative.size(0)
        target_rate = k / N
        model.set_temperature(2.0)

        for epoch in range(warmup_epochs):
            model.train()
            beta = min(1.0, epoch / max(warmup_epochs // 2, 1)) * 0.3
            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                B_h, B_n = idx_h.size(0), idx_n.size(0)

                optimizer.zero_grad()

                # Full forward — InfoNCE guided by mask-weighted embeddings
                _, M_a, _, _ = model(x_anchor[idx_h], ei)
                x_pos_aug = augment_positive(x_positive[idx_h], noise_sigma=0.05,
                                             jitter_shift=1, feature_mask_ratio=0.1)
                _, M_p, _, _ = model(x_pos_aug, ei)
                _, M_n, _, _ = model(x_negative[idx_n], ei)

                h_a_all = model.light_encoder(x_anchor[idx_h].view(B_h * N, num_features, -1)).view(B_h, N, -1)
                h_p_all = model.light_encoder(x_pos_aug.view(B_h * N, num_features, -1)).view(B_h, N, -1)
                h_n_all = model.light_encoder(x_negative[idx_n].view(B_n * N, num_features, -1)).view(B_n, N, -1)

                if mask_weighted_nce:
                    z_a = (h_a_all * M_a.unsqueeze(-1)).sum(1) / (M_a.sum(1, keepdim=True) + 1e-8)
                    z_p = (h_p_all * M_p.unsqueeze(-1)).sum(1) / (M_p.sum(1, keepdim=True) + 1e-8)
                    z_n = (h_n_all * M_n.unsqueeze(-1)).sum(1) / (M_n.sum(1, keepdim=True) + 1e-8)
                else:
                    z_a = h_a_all.mean(dim=1)
                    z_p = h_p_all.mean(dim=1)
                    z_n = h_n_all.mean(dim=1)

                l_nce = info_nce_loss(z_a, z_p, z_n, nce_temperature)
                if use_gib:
                    l_reg = variational_gib_loss(torch.cat([M_a, M_p, M_n], dim=0), target_rate)
                else:
                    l_reg = torch.cat([M_a, M_p, M_n], dim=0).abs().sum(dim=-1).mean()

                loss = l_nce + beta * l_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.set_temperature(1.0)
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
    sch_sup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_sup, T_0=50, T_mult=2, eta_min=1e-5)
    # MRR-based early stopping (patience=75)
    es = EarlyStopping(patience=75)

    k_warm = min(N // 2, N - 1)  # start with large k
    k_target = k  # already the optimal k from get_optimal_k()

    # Degenerate-convergence detection is only a warning; training continues.

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
        train_acc = ep_correct / n_train_sta

        degen_thresh = max(5.0 / N, 0.05)
        if epoch == 50 and train_acc < degen_thresh and not skip_degenerate_check:
            print(f"    [STA-Phase2] ⚠️  Epoch 50 acc={train_acc:.3f} < {degen_thresh:.3f} — "
                  f"slow/degenerate-looking convergence (no restart).")

        if epoch % 30 == 0 or epoch == supervised_epochs - 1:
            print(f"    [Phase2] Ep {epoch:03d}/{supervised_epochs} | loss={ep_loss/nb:.4f} | "
                  f"acc={train_acc:.4f} | RC-in-top-{k_cur}={ep_selected/n_train_sta:.4f} | k={k_cur}")

        # MRR-based early stopping (starts after k-annealing to ensure stable k)
        if epoch >= k_anneal_epochs:
            model.eval()
            with torch.no_grad():
                _, _, val_topk, val_sel = model(x_val_sta, ei)
                # Compute val MRR on selector logits
                val_mrr = 0.0
                val_sel_np = val_sel.cpu().numpy()
                y_val_np = y_val_sta.cpu().numpy()
                for i in range(len(y_val_np)):
                    ranked = np.argsort(val_sel_np[i])[::-1]
                    rk = int(np.where(ranked == y_val_np[i])[0][0]) + 1
                    val_mrr += 1.0 / rk
                val_mrr /= len(y_val_np)
            if es.step(val_mrr, model):
                print(f"    [Phase2] Early stop at epoch {epoch} "
                      f"(patience={es.patience}, best_mrr={es.best_mrr:.4f})")
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

            # Gated fusion: only adjust selected nodes with scorer signal.
            scorer = logits_full.squeeze(0)  # [N]
            combined = gated_selector_scorer_fusion(sel, scorer, top_k[0], gamma=0.35)

            y_true.append(true_idx)
            y_logits_combined.append(combined.cpu().numpy())
            y_logits_selector.append(sel.cpu().numpy())

            if true_idx in top_k[0].tolist():
                selection_hits += 1

    tops, mrr = get_ranking_metrics(y_true, y_logits_combined, top_ns=(1, 2, 5))
    tops_sel, mrr_sel = get_ranking_metrics(y_true, y_logits_selector, top_ns=(1, 2, 5))
    out = {
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
    out.update(_attach_rcaeval(y_true, y_logits_combined))
    split = split_logits_by_failure_type(test_anom, y_true, y_logits_combined)
    for ft in ("hard", "gray", "cascade"):
        if split[ft]["y_true"]:
            ft_m = RCAEvalMetrics().compute(split[ft]["y_true"], split[ft]["y_logits"])
            out[f"rcaeval_{ft}_avg1"] = ft_m["avg1"]
            out[f"rcaeval_{ft}_mrr"] = ft_m["mrr"]
    return out


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
    out = {
        "top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr,
        "mean_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "params": count_parameters(model), "k": N, "N": N,
    }
    out.update(_attach_rcaeval(y_true, y_logits))
    return out


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
    out = {"top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr, "params": count_parameters(model)}
    out.update(_attach_rcaeval(y_true, y_logits))
    return out


def eval_rf_baseline(clf, test_anom, name_to_idx, N):
    X_test = np.array([d.x.numpy().flatten() for d in test_anom])
    y_test = [name_to_idx[d.root_cause[0]] for d in test_anom]
    probs = clf.predict_proba(X_test)
    logits = np.zeros((len(X_test), N))
    for i, c in enumerate(clf.classes_):
        logits[:, c] = probs[:, i]
    tops, mrr = get_ranking_metrics(y_test, logits, top_ns=(1, 2, 5))
    y_logits_list = [logits[i] for i in range(len(y_test))]
    out = {"top1": tops[1], "top2": tops[2], "top5": tops[5], "mrr": mrr}
    out.update(_attach_rcaeval(y_test, y_logits_list))
    return out


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
        if m.get("missing_runs") or m.get("top1") is None:
            print(f"  {name:<38} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7}"
                  + (f" | {'N/A':>7}" if has_sel else "")
                  + (f" | {'N/A':>9} | {'—':>9}" if has_lat else ""))
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
    if not runs:
        return {
            "top1": None, "top2": None, "top5": None, "mrr": None,
            "selection_rate": None, "mean_latency_ms": None,
            "params": 0, "missing_runs": True,
        }
    if len(runs) == 1:
        return runs[0]
    keys = ["top1", "top2", "top5", "mrr", "selection_rate", "mean_latency_ms"]
    rcaeval_keys = sorted({k for r in runs for k in r if k.startswith("rcaeval_")})
    keys = keys + rcaeval_keys
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and r[k] is not None]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
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
    from rca_pipeline.simulated_data_scalable import generate_scalable_dataset

    if getattr(args, "deterministic", False):
        set_torch_deterministic(True)

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
    json_runs = []

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
        train_data, test_data = stratified_split_shuffle(
            dataset,
            train_ratio=0.8,
            name_to_idx=name_to_idx,
            seed=int(args.dataset_seed),
        )
        split_info = {
            "type": "stratified_split_shuffle",
            "mode": "fixed_split_by_dataset_seed",
            "seed_source": "dataset_seed",
            "seed": int(args.dataset_seed),
            "train_ratio": 0.8,
        }

        # Causal / hybrid graph construction (TRAIN-ONLY to avoid leakage)
        graph_type = getattr(args, "graph_type", "structural")
        edge_weights = None
        if graph_type in ("causal", "hybrid"):
            healthy_windows = [d for d in train_data if d.y.item() == 0]
            if healthy_windows:
                # Build [T, N] healthy matrix by concatenating temporal node means.
                xh_blocks = [d.x.mean(dim=1).transpose(0, 1) for d in healthy_windows]  # [W, N]
                X_h = torch.cat(xh_blocks, dim=0).cpu().numpy()
                builder = CausalGraphBuilder(method=graph_type)
                cg_result = builder.build(X_h, edge_index)
                print(
                    f"  Causal graph (train-only): method={cg_result.method_used}, "
                    f"edges={cg_result.edge_index.size(1)}, "
                    f"CS={cg_result.causal_sufficiency_score:.2f}"
                )
                edge_index = cg_result.edge_index
                edge_weights = cg_result.edge_weights
        train_data_raw, test_data_raw = train_data, test_data
        train_data = _apply_drift_filter(
            train_data,
            enable_drift=args.enable_drift,
            delta=args.drift_delta,
            cooldown_steps=args.drift_cooldown_steps,
        )
        test_data = _apply_drift_filter(
            test_data,
            enable_drift=args.enable_drift,
            delta=args.drift_delta,
            cooldown_steps=args.drift_cooldown_steps,
        )
        if not train_data or not test_data:
            # Fallback to unfiltered split when drift cooldown removes too many windows.
            train_data, test_data = train_data_raw, test_data_raw
        drift_info = {
            "enabled": bool(args.enable_drift),
            "delta": float(args.drift_delta),
            "cooldown_steps": int(args.drift_cooldown_steps),
            "n_train_raw": int(len(train_data_raw)),
            "n_test_raw": int(len(test_data_raw)),
            "n_train_kept": int(len(train_data)),
            "n_test_kept": int(len(test_data)),
            "train_drop_ratio": float(1.0 - (len(train_data) / max(len(train_data_raw), 1))),
            "test_drop_ratio": float(1.0 - (len(test_data) / max(len(test_data_raw), 1))),
        }

        norm_stats = None
        if not args.no_normalize:
            mean, std = compute_norm_stats(train_data)
            normalize_dataset(train_data, mean, std)
            normalize_dataset(test_data, mean, std)
            norm_stats = {
                "mean": mean.detach().cpu().flatten().tolist(),
                "std": std.detach().cpu().flatten().tolist(),
            }

        train_anom = get_anomalies(train_data, name_to_idx)
        test_anom = get_anomalies(test_data, name_to_idx)
        F_ = dataset[0].x.size(1)

        rc_dist = Counter(d.root_cause[0] for d in train_anom)
        print(f"  Train: {len(train_data)} ({len(train_anom)} anomalies, {len(rc_dist)} RC classes)")
        print(f"  Test:  {len(test_data)} ({len(test_anom)} anomalies)")

        # Collect multi-seed results
        sta_runs, late_runs, gat_runs, gcn_runs, mlp_runs, rf_runs = [], [], [], [], [], []

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  === Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed}) ===")
            set_all_seeds(seed)

            # ---- STA ----
            # k=None means get_optimal_k(N) is used automatically
            k_opt = get_optimal_k(N)
            print(f"\n  -- Training SelectThenAnalyze (k={k_opt}, auto from get_optimal_k) --")
            t0 = time.time()
            sta_model = train_sta_compact(
                train_data, node_names, edge_index, device,
                hidden_dim=hidden, k=k_opt,
                warmup_epochs=args.sta_warmup_epochs,
                supervised_epochs=args.supervised_epochs,
                batch_size=args.batch_size,
                k_anneal_epochs=args.sta_k_anneal_epochs,
                use_gib=not args.no_gib,
                mask_weighted_nce=not args.no_mask_weighted_nce,
                selector_type=args.selector_type,
                temporal_encoder=args.temporal_encoder,
            )
            print(f"  STA training completed in {time.time()-t0:.1f}s")
            sta_runs.append(eval_sta(sta_model, test_anom, edge_index, name_to_idx, device))

            # ---- Late Bottleneck (parallel approach) ----
            print("\n  -- Training STGCN+GIB (Late Bottleneck) --")
            t0 = time.time()
            try:
                late_model = train_v1_compact(
                    train_data, node_names, edge_index, device,
                    hidden_dim=hidden,
                    contrastive_epochs=args.late_contrastive_epochs,
                    supervised_epochs=args.supervised_epochs,
                    batch_size=args.batch_size,
                    edge_weights=edge_weights,
                    cascade_loss_weight=args.cascade_loss_weight,
                )
                print(f"  Late Bottleneck training completed in {time.time()-t0:.1f}s")
                late_runs.append(eval_v1(late_model, test_anom, edge_index, name_to_idx, device))
            except RuntimeError as e:
                print(f"  ⚠️  Late BN failed: {e}")

            # ---- Late Bottleneck GATv2 ----
            if not args.skip_late_gat:
                print("\n  -- Training Late Bottleneck (GATv2, no GIB) --")
                t0 = time.time()
                try:
                    gat_model = train_gat_compact(
                        train_data, node_names, edge_index, device,
                        hidden_dim=hidden,
                        contrastive_epochs=args.late_contrastive_epochs,
                        supervised_epochs=args.supervised_epochs,
                        batch_size=args.batch_size,
                        edge_weights=edge_weights,
                        cascade_loss_weight=args.cascade_loss_weight,
                    )
                    print(f"  Late GAT training completed in {time.time()-t0:.1f}s")
                    gat_runs.append(eval_gat(gat_model, test_anom, edge_index, name_to_idx, device))
                except RuntimeError as e:
                    print(f"  ⚠️  Late GAT failed: {e}")

            # ---- Baselines ----
            print("\n  -- Training Full-Graph GCN baseline --")
            gcn_model = train_baseline_gcn(train_anom, edge_index, name_to_idx, N, F_,
                                           hidden_dim=hidden,
                                           epochs=args.baseline_epochs,
                                           batch_size=args.batch_size,
                                           seed=seed)
            gcn_model.eval()
            gcn_runs.append(eval_gcn_baseline(gcn_model, test_anom, edge_index, name_to_idx, N))

            print("  -- Training MLP baseline --")
            mlp_model = train_baseline_mlp(train_anom, name_to_idx, N, hidden_dim=64,
                                           epochs=args.baseline_epochs,
                                           batch_size=args.batch_size,
                                           seed=seed)
            mlp_runs.append(eval_mlp_baseline(mlp_model, test_anom, name_to_idx, N))

            print("  -- Training RF baseline --")
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

        if not args.skip_late_gat:
            results[f"Late Bottleneck GATv2 (N={N})"] = aggregate_seeds(gat_runs)

        r_gcn = aggregate_seeds(gcn_runs)
        results[f"Full-Graph GCN (N={N})"] = r_gcn

        r_mlp = aggregate_seeds(mlp_runs)
        results["MLP Baseline"] = r_mlp

        r_rf = aggregate_seeds(rf_runs)
        results["Random Forest"] = r_rf

        # ---- Print results ----
        print_results_table(results, f"RESULTS FOR N={N} (mean over {NUM_SEEDS} seeds)")

        # Regression vs reference Top-1
        print("\n  --- Regression check (vs reference Top-1, Δ>0.5% triggers ⚠️) ---")
        short_map = {
            f"STA Select-then-Analyze (k={k})": "STA",
            f"Late Bottleneck STGCN+GIB (N={N})": "LateBN_GIB",
            f"Full-Graph GCN (N={N})": "FullGraphGCN",
            f"Late Bottleneck GATv2 (N={N})": "LateBN_GAT",
        }
        for disp, m in results.items():
            sk = short_map.get(disp)
            if sk is None or sk not in REFERENCE_TOP1 or N not in REFERENCE_TOP1[sk]:
                continue
            ref = REFERENCE_TOP1[sk][N]
            top1 = m.get("top1")
            if m.get("missing_runs") or top1 is None:
                print(f"  — {disp}: no metric (run missing or training failed)")
                continue
            if _regression_flag(sk, N, top1):
                print(f"  ⚠️  {disp}: Top-1={top1:.4f} ref={ref:.4f} (regression > 0.5%)")
            else:
                print(f"  OK {disp}: Top-1={top1:.4f} ref={ref:.4f}")

        json_runs.append({
            "N": N,
            "k_map_entry": k,
            "seeds": SEEDS,
            "no_normalize": args.no_normalize,
            "skip_late_gat": args.skip_late_gat,
            "metrics": results,
            "hyperparams": {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": args.batch_size,
                "hidden_dim": hidden,
                "sta_warmup_epochs": args.sta_warmup_epochs,
                "sta_k_anneal_epochs": args.sta_k_anneal_epochs,
                "supervised_epochs": args.supervised_epochs,
                "late_contrastive_epochs": args.late_contrastive_epochs,
                "nce_temperature": 0.07,
                "gib_beta_max": 0.3,
                "selector_type": args.selector_type,
                "temporal_encoder": args.temporal_encoder,
                "use_gib": not args.no_gib,
                "mask_weighted_nce": not args.no_mask_weighted_nce,
                "cascade_loss_weight": args.cascade_loss_weight,
                "graph_type": getattr(args, "graph_type", "structural"),
            },
            "reference_top1": {m: REFERENCE_TOP1[m].get(N) for m in REFERENCE_TOP1},
            "normalization": {
                "enabled": bool(not args.no_normalize),
                "train_only": True,
                "stats": norm_stats,
            },
            "split_mode": str(split_info.get("mode", "unknown")),
            "split": split_info,
            "drift": drift_info,
            "env": runtime_env_info(),
            "deterministic": bool(getattr(args, "deterministic", False)),
        })

        # ---- Collect latency for scaling plot ----
        sta_lat = r_sta.get("mean_latency_ms") or 0
        gcn_lat = r_gcn.get("mean_latency_ms") or 0
        all_scaling.append({"N": N, "k": k, "v3_lat": sta_lat, "gcn_lat": gcn_lat,
                            "v3_params": r_sta.get("params", 0), "gcn_params": r_gcn.get("params", 0)})

    # ---- Latency scaling summary ----
    if len(all_scaling) > 1:
        print_latency_scaling(all_scaling)

    if getattr(args, "output_json", None):
        out_path = args.output_json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"runs": json_runs, "reference_top1": REFERENCE_TOP1}, f, indent=2)
        print(f"\n  Wrote JSON results to {out_path}")

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
    parser.add_argument("--seeds", type=str, default="42,123,7,1337,2025",
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
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable z-score normalization (train stats only when enabled)")
    parser.add_argument("--graph-type", type=str, default="structural",
                        choices=["structural", "causal", "hybrid"],
                        help="Graph construction: structural (BA), causal (PC/Granger), hybrid (union)")
    parser.add_argument("--cascade-loss-weight", type=float, default=0.0,
                        help="Weight for optional causal cascade regularization in Late BN/GAT")
    parser.add_argument("--enable-drift", action="store_true",
                        help="Enable ADWIN drift cooldown filtering before training/eval split")
    parser.add_argument("--drift-delta", type=float, default=0.002,
                        help="ADWIN delta for drift detector")
    parser.add_argument("--drift-cooldown-steps", type=int, default=30,
                        help="Cooldown steps for drift detector filter")
    parser.add_argument("--skip-late-gat", action="store_true",
                        help="Skip Late Bottleneck GATv2 training/eval (faster)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write aggregated metrics to this JSON path")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Best-effort deterministic mode for PyTorch (may reduce performance).",
    )

    # Ablation flags for STA
    parser.add_argument("--no-gib", action="store_true",
                        help="[Ablation] Use L1 instead of variational GIB in warmup")
    parser.add_argument("--no-mask-weighted-nce", action="store_true",
                        help="[Ablation] Use mean-pooled H_fast for InfoNCE instead of mask-weighted")
    parser.add_argument("--selector-type", type=str, default="ste",
                        choices=["ste", "hierarchical"],
                        help="Selector type for STA")
    parser.add_argument("--temporal-encoder", type=str, default="conv",
                        choices=["conv", "s4"],
                        help="Temporal encoder type for STA deep path")

    from rca_pipeline.config import BenchmarkSimConfig, parse_args_with_config
    main(parse_args_with_config(parser, BenchmarkSimConfig))
