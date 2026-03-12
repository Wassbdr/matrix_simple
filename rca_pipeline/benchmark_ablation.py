#!/usr/bin/env python3
"""
Ablation Study for RCA Pipeline.

Systematically isolates the contribution of each architectural component:

    A. Component Decomposition (N=20 and N=100):
       A1. Full Late Bottleneck (STGCN + GIB)          [reference]
       A2. Full-Graph GCN (STGCN, no GIB)              [reference]
       A3. Spatial-only GCN (no temporal conv, no GIB)  [new]
       A4. Temporal-only (no GCN, no GIB)               [new]

    B. Training Protocol for Late Bottleneck (N=20 and N=100):
       B1. Late BN full (contrastive + supervised)      [reference]
       B2. Late BN supervised only (no contrastive)     [new]

    C. STA k-Sensitivity (N=100 only):
       C1. k=10 (10% of nodes)                          [reference]
       C2. k=15 (15%)                                   [new]
       C3. k=20 (20%)                                   [new]
       C4. k=30 (30%)                                   [new]

    D. STA Training Protocol (N=100 only):
       D1. Full STA (warmup + k-annealing)              [reference]
       D2. STA no contrastive warmup                    [new]
       D3. STA no k-annealing                           [new]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings
from collections import Counter
from typing import List, Dict
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

from model_sta import SelectThenAnalyze
from model import SpatioTemporalRCA
from losses import info_nce_loss, asymmetric_gib_loss
from augment import augment_positive
from benchmark import (
    count_parameters, get_ranking_metrics, stratified_split, get_anomalies,
    EarlyStopping, aggregate_seeds, print_header, print_results_table,
    train_v1_compact, eval_v1, train_sta_compact, eval_sta,
    train_baseline_gcn, eval_gcn_baseline,
)

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =====================================================================
# Ablation Model A3: Spatial-Only GCN (no temporal convolutions)
# =====================================================================

class SpatialOnlyGCN(nn.Module):
    """
    GCN on temporally-aggregated features (mean + std over time).
    Tests whether the temporal convolutions contribute beyond simple
    statistical summarization of the window.
    Input: [B, N, F, W] -> aggregate to [B*N, 2F] -> GCN -> scorer
    """
    def __init__(self, num_features, hidden_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.proj = nn.Sequential(
            nn.Linear(num_features * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.g1 = GCNConv(hidden_dim, hidden_dim)
        self.g2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, ei):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, N, Ff, W = x.shape
        # Temporal aggregation: mean + std (no learned temporal processing)
        x_agg = torch.cat([x.mean(dim=-1), x.std(dim=-1).clamp(min=1e-8)], dim=-1)  # [B, N, 2F]
        h = self.proj(x_agg.reshape(B * N, -1))  # [B*N, H]
        # Batched GCN
        if B > 1:
            off = torch.arange(B, device=h.device) * N
            bei = ei.unsqueeze(0).expand(B, -1, -1) + off.view(-1, 1, 1)
            bei = bei.reshape(2, -1)
        else:
            bei = ei
        r = h
        h = F.relu(self.g1(h, bei))
        h = self.g2(h, bei)
        h = self.norm(h + r)
        h = h.view(B, N, -1)
        return self.scorer(h).squeeze(-1)  # [B, N]


# =====================================================================
# Ablation Model A4: Temporal-Only (no GCN)
# =====================================================================

class TemporalOnlyModel(nn.Module):
    """
    Temporal convolutions + per-node scoring, NO graph convolutions.
    Tests whether the graph structure contributes beyond independent
    per-node temporal pattern recognition.
    """
    def __init__(self, num_features, hidden_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.tc1 = nn.Conv1d(num_features, hidden_dim, 3, padding=1)
        self.tc2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.tn = nn.BatchNorm1d(hidden_dim)
        self.ta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, ei=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, N, Ff, W = x.shape
        h = x.reshape(B * N, Ff, W)
        h = F.relu(self.tc1(h))
        h = F.relu(self.tc2(h))
        ht = h.permute(0, 2, 1)  # [B*N, W, H]
        a = F.softmax(self.ta(ht).squeeze(-1), dim=-1)
        h = (ht * a.unsqueeze(-1)).sum(1)  # [B*N, H]
        h = self.tn(h)
        h = h.view(B, N, -1)
        return self.scorer(h).squeeze(-1)  # [B, N]


# =====================================================================
# Generic training for simple models (CE only)
# =====================================================================

def train_simple_model(model, train_anom, edge_index, name_to_idx, N,
                       epochs=150, batch_size=256, seed=42, needs_ei=True):
    """Train any model with simple supervised CE."""
    torch.manual_seed(seed)
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
            if needs_ei:
                lo = model(X[idx], edge_index)
            else:
                lo = model(X[idx])
            loss = F.cross_entropy(lo, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


def eval_simple_model(model, test_anom, edge_index, name_to_idx, N, needs_ei=True):
    """Evaluate any model that returns [B, N] logits."""
    y_true, y_logits = [], []
    model.eval()
    with torch.no_grad():
        # Warmup
        x0 = test_anom[0].x.unsqueeze(0)
        if needs_ei:
            _ = model(x0, edge_index)
        else:
            _ = model(x0)
    latencies = []
    with torch.no_grad():
        for data in test_anom:
            x = data.x.unsqueeze(0)
            t0 = time.perf_counter()
            if needs_ei:
                lo = model(x, edge_index)
            else:
                lo = model(x)
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
        "params": count_parameters(model), "k": N, "N": N,
    }


# =====================================================================
# Main Ablation
# =====================================================================

def run_ablation():
    from simulated_data_scalable import generate_scalable_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    NUM_SEEDS = 3
    SEEDS = [42, 123, 7]

    # Run on N=20 (baseline) and N=100 (the interesting case)
    configs = [
        {"N": 20,  "ba_m": 2, "num_samples": 3000, "k": 5,  "hidden": 32},
        {"N": 100, "ba_m": 3, "num_samples": 3000, "k": 10, "hidden": 32},
    ]

    all_results = {}

    for cfg in configs:
        N = cfg["N"]
        k = cfg["k"]
        hidden = cfg["hidden"]

        print_header(f"ABLATION STUDY — N={N}", char="#")

        # ---- Generate dataset (same seed=42 as main benchmark) ----
        print(f"\n  Generating BA({N}, m={cfg['ba_m']}) dataset...")
        dataset, node_names, edge_index = generate_scalable_dataset(
            num_nodes=N, ba_m=cfg["ba_m"], num_samples=cfg["num_samples"],
            window_size=12, num_features=5, anomaly_ratio=0.4, seed=42,
        )
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        train_data, test_data = stratified_split(dataset, name_to_idx)
        train_anom = get_anomalies(train_data, name_to_idx)
        test_anom = get_anomalies(test_data, name_to_idx)
        F_ = dataset[0].x.size(1)

        rc_dist = Counter(d.root_cause[0] for d in train_anom)
        print(f"  Train: {len(train_data)} ({len(train_anom)} anom, {len(rc_dist)} RC classes)")
        print(f"  Test:  {len(test_data)} ({len(test_anom)} anom)")

        # ==============================================================
        # ABLATION A: Component Decomposition
        # ==============================================================
        print_header(f"A. COMPONENT DECOMPOSITION (N={N})", char="-")

        # A1: Full Late Bottleneck (reference)
        late_runs = []
        for si, seed in enumerate(SEEDS):
            print(f"\n  [A1] Late Bottleneck — seed {seed}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = train_v1_compact(train_data, node_names, edge_index, device,
                                 hidden_dim=hidden, contrastive_epochs=80,
                                 supervised_epochs=150, batch_size=256)
            late_runs.append(eval_v1(m, test_anom, edge_index, name_to_idx, device))

        # A2: Full-Graph GCN / STGCN no GIB (reference)
        gcn_runs = []
        for si, seed in enumerate(SEEDS):
            print(f"\n  [A2] STGCN no GIB (FullGraphGCN) — seed {seed}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = train_baseline_gcn(train_anom, edge_index, name_to_idx, N, F_,
                                   hidden_dim=hidden, epochs=150, seed=seed)
            m.eval()
            gcn_runs.append(eval_gcn_baseline(m, test_anom, edge_index, name_to_idx, N))

        # A3: Spatial-only GCN (no temporal conv)
        spatial_runs = []
        for si, seed in enumerate(SEEDS):
            print(f"\n  [A3] Spatial-only GCN (no temporal) — seed {seed}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = SpatialOnlyGCN(F_, hidden, N)
            m = train_simple_model(m, train_anom, edge_index, name_to_idx, N,
                                   epochs=150, seed=seed, needs_ei=True)
            m.eval()
            spatial_runs.append(eval_simple_model(m, test_anom, edge_index, name_to_idx, N))

        # A4: Temporal-only (no GCN)
        temporal_runs = []
        for si, seed in enumerate(SEEDS):
            print(f"\n  [A4] Temporal-only (no GCN) — seed {seed}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = TemporalOnlyModel(F_, hidden, N)
            m = train_simple_model(m, train_anom, edge_index, name_to_idx, N,
                                   epochs=150, seed=seed, needs_ei=False)
            m.eval()
            temporal_runs.append(eval_simple_model(m, test_anom, edge_index, name_to_idx, N,
                                                   needs_ei=False))

        results_a = {
            "A1: Late BN (STGCN+GIB)": aggregate_seeds(late_runs),
            "A2: STGCN no GIB (FullGraphGCN)": aggregate_seeds(gcn_runs),
            "A3: Spatial-only GCN (no temporal)": aggregate_seeds(spatial_runs),
            "A4: Temporal-only (no GCN)": aggregate_seeds(temporal_runs),
        }
        print_results_table(results_a, f"ABLATION A — Component Decomposition (N={N})")
        all_results[f"A_N{N}"] = results_a

        # ==============================================================
        # ABLATION B: Training Protocol for Late Bottleneck
        # ==============================================================
        print_header(f"B. LATE BOTTLENECK TRAINING PROTOCOL (N={N})", char="-")

        # B2: Late BN supervised only (no contrastive pretraining)
        late_nosup_runs = []
        for si, seed in enumerate(SEEDS):
            print(f"\n  [B2] Late BN no contrastive — seed {seed}")
            torch.manual_seed(seed); np.random.seed(seed)
            m = train_v1_compact(train_data, node_names, edge_index, device,
                                 hidden_dim=hidden, contrastive_epochs=0,
                                 supervised_epochs=150, batch_size=256)
            late_nosup_runs.append(eval_v1(m, test_anom, edge_index, name_to_idx, device))

        results_b = {
            "B1: Late BN full (contrastive+sup)": aggregate_seeds(late_runs),
            "B2: Late BN supervised only": aggregate_seeds(late_nosup_runs),
        }
        print_results_table(results_b, f"ABLATION B — Late BN Training Protocol (N={N})")
        all_results[f"B_N{N}"] = results_b

        # ==============================================================
        # ABLATION C & D: STA (only at N=100)
        # ==============================================================
        if N == 100:
            # C: k-sensitivity
            print_header(f"C. STA k-SENSITIVITY (N={N})", char="-")

            k_values = [10, 15, 20, 30]
            sta_k_results = {}

            for k_val in k_values:
                runs = []
                for si, seed in enumerate(SEEDS):
                    print(f"\n  [C] STA k={k_val} — seed {seed}")
                    torch.manual_seed(seed); np.random.seed(seed)
                    m = train_sta_compact(
                        train_data, node_names, edge_index, device,
                        hidden_dim=hidden, k=k_val,
                        warmup_epochs=20, supervised_epochs=150,
                        batch_size=256, k_anneal_epochs=60,
                    )
                    runs.append(eval_sta(m, test_anom, edge_index, name_to_idx, device))
                sta_k_results[f"STA k={k_val} ({k_val}% nodes)"] = aggregate_seeds(runs)

            print_results_table(sta_k_results, f"ABLATION C — STA k-Sensitivity (N={N})")
            all_results["C"] = sta_k_results

            # D: STA training protocol
            print_header(f"D. STA TRAINING PROTOCOL (N={N})", char="-")

            # D1: reference (full STA k=10) — reuse from C
            sta_ref_runs = []
            for si, seed in enumerate(SEEDS):
                print(f"\n  [D1] STA full (warmup + k-anneal) — seed {seed}")
                torch.manual_seed(seed); np.random.seed(seed)
                m = train_sta_compact(
                    train_data, node_names, edge_index, device,
                    hidden_dim=hidden, k=k,
                    warmup_epochs=20, supervised_epochs=150,
                    batch_size=256, k_anneal_epochs=60,
                )
                sta_ref_runs.append(eval_sta(m, test_anom, edge_index, name_to_idx, device))

            # D2: STA no contrastive warmup
            sta_nowarmup_runs = []
            for si, seed in enumerate(SEEDS):
                print(f"\n  [D2] STA no warmup — seed {seed}")
                torch.manual_seed(seed); np.random.seed(seed)
                m = train_sta_compact(
                    train_data, node_names, edge_index, device,
                    hidden_dim=hidden, k=k,
                    warmup_epochs=0, supervised_epochs=150,
                    batch_size=256, k_anneal_epochs=60,
                )
                sta_nowarmup_runs.append(eval_sta(m, test_anom, edge_index, name_to_idx, device))

            # D3: STA no k-annealing (start directly at target k)
            sta_noanneal_runs = []
            for si, seed in enumerate(SEEDS):
                print(f"\n  [D3] STA no k-annealing — seed {seed}")
                torch.manual_seed(seed); np.random.seed(seed)
                m = train_sta_compact(
                    train_data, node_names, edge_index, device,
                    hidden_dim=hidden, k=k,
                    warmup_epochs=20, supervised_epochs=150,
                    batch_size=256, k_anneal_epochs=0,
                )
                sta_noanneal_runs.append(eval_sta(m, test_anom, edge_index, name_to_idx, device))

            results_d = {
                "D1: STA full (warmup + k-anneal)": aggregate_seeds(sta_ref_runs),
                "D2: STA no warmup": aggregate_seeds(sta_nowarmup_runs),
                "D3: STA no k-annealing": aggregate_seeds(sta_noanneal_runs),
            }
            print_results_table(results_d, f"ABLATION D — STA Training Protocol (N={N}, k={k})")
            all_results["D"] = results_d

    # ==============================================================
    # SUMMARY
    # ==============================================================
    print_header("ABLATION STUDY — SUMMARY", char="*")

    print("\n  KEY FINDINGS:\n")
    print("  A. Component Decomposition:")
    for n_key in ["A_N20", "A_N100"]:
        if n_key in all_results:
            r = all_results[n_key]
            N_val = 20 if "20" in n_key else 100
            print(f"\n    N={N_val}:")
            for name, m in r.items():
                std_str = f" ±{m['top1_std']:.3f}" if 'top1_std' in m else ""
                print(f"      {name:<45} Top-1={m['top1']:.4f}{std_str}  MRR={m['mrr']:.4f}")

            # Compute deltas
            base = r["A2: STGCN no GIB (FullGraphGCN)"]["top1"]
            for name, m in r.items():
                if name != "A2: STGCN no GIB (FullGraphGCN)":
                    delta = m["top1"] - base
                    print(f"        Δ vs STGCN: {name.split(':')[0]}={delta:+.4f}")

    if "B_N20" in all_results or "B_N100" in all_results:
        print("\n  B. Late BN Training Protocol:")
        for n_key in ["B_N20", "B_N100"]:
            if n_key in all_results:
                r = all_results[n_key]
                N_val = 20 if "20" in n_key else 100
                full = r["B1: Late BN full (contrastive+sup)"]["top1"]
                nosup = r["B2: Late BN supervised only"]["top1"]
                print(f"    N={N_val}: Full={full:.4f}, No-contrastive={nosup:.4f}, "
                      f"Δ={full-nosup:+.4f}")

    if "C" in all_results:
        print("\n  C. STA k-Sensitivity (N=100):")
        for name, m in all_results["C"].items():
            std_str = f" ±{m['top1_std']:.3f}" if 'top1_std' in m else ""
            sel = m.get("selection_rate", 0)
            print(f"    {name:<35} Top-1={m['top1']:.4f}{std_str}  "
                  f"MRR={m['mrr']:.4f}  Sel%={sel:.4f}")

    if "D" in all_results:
        print("\n  D. STA Training Protocol (N=100):")
        for name, m in all_results["D"].items():
            std_str = f" ±{m['top1_std']:.3f}" if 'top1_std' in m else ""
            print(f"    {name:<45} Top-1={m['top1']:.4f}{std_str}  MRR={m['mrr']:.4f}")

    print_header("ABLATION COMPLETE", char="*")


if __name__ == "__main__":
    run_ablation()
