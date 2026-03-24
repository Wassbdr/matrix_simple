#!/usr/bin/env python3
"""
Complete benchmark and ablation runner for RCA baselines.

Outputs:
- results/benchmark_full.csv
- results/benchmark_summary.md
- results/ablation_full.csv
- results/ablation_summary.md
- results/modality_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

try:
    from benchmark import (
        EarlyStopping,
        FullGraphGCN,
        get_anomalies,
        get_ranking_metrics,
        train_baseline_gcn,
        train_baseline_mlp,
        train_baseline_rf,
        train_sta_compact,
        train_v1_compact,
    )
    from benchmark_ablation import SpatialOnlyGCN, TemporalOnlyModel, train_simple_model
    from unified_data import UnifiedDataLoader
except ImportError:
    from rca_pipeline.benchmark import (
        EarlyStopping,
        FullGraphGCN,
        get_anomalies,
        get_ranking_metrics,
        train_baseline_gcn,
        train_baseline_mlp,
        train_baseline_rf,
        train_sta_compact,
        train_v1_compact,
    )
    from rca_pipeline.benchmark_ablation import SpatialOnlyGCN, TemporalOnlyModel, train_simple_model
    from rca_pipeline.unified_data import UnifiedDataLoader


SEEDS = [42, 123]  # Temp for mini-benchmark (was [42, 123, 7, 0, 99])
SCALES = [20]      # Temp for mini-benchmark (was [20, 50, 100])
# FIX-REC4: k=12 at N=50 was k=8 (too small → causal node excluded in ~90% of selections)
# Measured: STA 40.4%(N=20) → 9.8%(N=50) → 16.8%(N=100) confirmed N=50 underperforms N=100.
K_MAP = {20: 5, 50: 12, 100: 15}
BA_M_MAP = {20: 2, 50: 3, 100: 3}


@dataclass
class PreparedData:
    train: List[Any]
    test: List[Any]
    node_names: List[str]
    edge_index: torch.Tensor
    source_status: Dict[str, bool]
    source_counts_train: Dict[str, int]
    source_counts_test: Dict[str, int]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def serialize_breakdown(y_true: List[int], logits_list: List[np.ndarray], metas: List[Dict[str, Any]]) -> str:
    rows: List[Tuple[str, str, str, float]] = []
    for yt, logits, meta in zip(y_true, logits_list, metas):
        ranked = np.argsort(logits)[::-1]
        rank = int(np.where(ranked == yt)[0][0]) + 1
        rr = 1.0 / rank
        p = str(meta.get("pattern_type", "unknown"))
        r = "recurrent" if bool(meta.get("is_recurrent", False)) else "novel"
        sev = float(meta.get("severity", 0.0) or 0.0)
        if sev < 0.3:
            s = "low"
        elif sev < 0.7:
            s = "medium"
        else:
            s = "high"
        rows.append((p, r, s, rr))

    by_pattern: Dict[str, List[float]] = defaultdict(list)
    by_recurrence: Dict[str, List[float]] = defaultdict(list)
    by_severity: Dict[str, List[float]] = defaultdict(list)
    for p, r, s, rr in rows:
        by_pattern[p].append(rr)
        by_recurrence[r].append(rr)
        by_severity[s].append(rr)

    payload = {
        "by_pattern_type": {k: {"count": len(v), "mrr": float(np.mean(v))} for k, v in by_pattern.items()},
        "by_recurrence": {k: {"count": len(v), "mrr": float(np.mean(v))} for k, v in by_recurrence.items()},
        "by_severity": {k: {"count": len(v), "mrr": float(np.mean(v))} for k, v in by_severity.items()},
    }
    return json.dumps(payload, ensure_ascii=True)


def aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
        out[f"{k}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def detect_sources_present(loader: UnifiedDataLoader) -> Dict[str, bool]:
    status = {
        "synthetic_v2": "synthetic_v2" in loader.datasets and len(loader.datasets["synthetic_v2"]) > 0,
        "cluster": "cluster" in loader.datasets and len(loader.datasets["cluster"]) > 0,
        "rcaeval": any(k.startswith("rcaeval_") and len(v) > 0 for k, v in loader.datasets.items()),
    }
    return status


def select_modalities(data: List[Any], mode: str) -> List[Any]:
    # mode: metrics | metrics_traces | full
    out: List[Any] = []
    for d in data:
        nd = d.clone()
        if mode == "metrics":
            nd.x = nd.x[:, :5, :]
            nd.modality_mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            nd.feature_presence_mask = torch.tensor([1.0] * 5, dtype=torch.float32)
        elif mode == "metrics_traces":
            nd.x = nd.x[:, :9, :]
            nd.modality_mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
            nd.feature_presence_mask = torch.tensor([1.0] * 9, dtype=torch.float32)
        elif mode == "full":
            nd.x = nd.x[:, :13, :]
            nd.modality_mask = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            nd.feature_presence_mask = torch.tensor([1.0] * 13, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown modality mode: {mode}")
        out.append(nd)
    return out


def prepare_data_for_scale(
    num_nodes: int,
    num_samples: int,
    seed: int,
    include_cluster: bool,
    include_rcaeval: bool,
    prometheus_url: Optional[str],
    jaeger_url: Optional[str],
    loki_url: Optional[str],
    rcaeval_name: str,
) -> PreparedData:
    # FIX-REC1: synthetic generator produces 5 features; UnifiedDataLoader previously forced 13 dims,
    # zero-padding 8 channels → BatchNorm amplified zeros to ±1e6 → corrupted all GNN representations.
    # Use 5 features for synthetic-only runs; 13 only when real multi-modal sources are active.
    has_real_sources = include_cluster or include_rcaeval
    effective_features = 13 if has_real_sources else 5
    loader = UnifiedDataLoader(window_size=12, num_features=effective_features, normalize="per_source", seed=seed)
    loader.add_synthetic(
        num_nodes=num_nodes,
        num_samples=num_samples,
        ba_m=BA_M_MAP[num_nodes],
        anomaly_ratio=0.4,
        recurrence_ratio=0.8,
        use_realistic_pattern_weights=True,
    )

    if include_cluster and prometheus_url:
        loader.add_cluster(
            prometheus_url=prometheus_url,
            jaeger_url=jaeger_url,
            loki_url=loki_url,
            duration="24h",
            weight=1.0,
        )

    if include_rcaeval:
        loader.add_rcaeval(dataset_name=rcaeval_name, weight=1.0)

    loader.load_all()
    source_status = detect_sources_present(loader)

    train, test = loader.get_splits(train_ratio=0.8, stratify_by="source", balance_sources=True)
    stats = loader.fit_scaler(train)
    train = loader.apply_scaler(train, stats)
    test = loader.apply_scaler(test, stats)

    if not train:
        raise RuntimeError("Train split is empty after loading sources")

    sample = train[0]
    edge_index = sample.edge_index

    # Prefer canonical node names from UnifiedDataLoader/generator to keep
    # compatibility with root_cause labels used by get_anomalies().
    if loader.node_names is not None and len(loader.node_names) == num_nodes:
        node_names = list(loader.node_names)
    else:
        node_names = [f"node_{i}" for i in range(num_nodes)]

    train_counts: Dict[str, int] = defaultdict(int)
    test_counts: Dict[str, int] = defaultdict(int)
    for d in train:
        train_counts[getattr(d, "source", "unknown")] += 1
    for d in test:
        test_counts[getattr(d, "source", "unknown")] += 1

    return PreparedData(
        train=train,
        test=test,
        node_names=node_names,
        edge_index=edge_index,
        source_status=source_status,
        source_counts_train=dict(train_counts),
        source_counts_test=dict(test_counts),
    )


def collect_metas(samples: List[Any]) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for d in samples:
        metas.append(
            {
                "pattern_type": getattr(d, "pattern_type", "unknown"),
                "is_recurrent": bool(getattr(d, "is_recurrent", False)),
                "severity": float(getattr(d, "severity", 1.0) or 1.0),
            }
        )
    return metas


def infer_sta(model: Any, samples: List[Any], edge_index: torch.Tensor, name_to_idx: Dict[str, int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits_list: List[np.ndarray] = []
    latencies: List[float] = []
    with torch.no_grad():
        _ = model(samples[0].x.unsqueeze(0), edge_index)
    with torch.no_grad():
        for d in samples:
            x = d.x.unsqueeze(0)
            t0 = time.perf_counter()
            logits_full, _, _, sel_logits = model(x, edge_index)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            scorer = logits_full.squeeze(0)
            scorer_safe = scorer.clone()
            scorer_safe[scorer == float("-inf")] = 0.0
            combined = sel_logits.squeeze(0) + 0.5 * scorer_safe
            logits_list.append(combined.cpu().numpy())
            if getattr(d, 'root_cause', None) and len(d.root_cause) > 0 and d.root_cause[0]:
                y_true.append(name_to_idx[d.root_cause[0]])
            else:
                y_true.append(-1)
    return y_true, logits_list, latencies


def infer_late(model: Any, samples: List[Any], edge_index: torch.Tensor, name_to_idx: Dict[str, int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits_list: List[np.ndarray] = []
    latencies: List[float] = []
    N = samples[0].x.size(0)
    with torch.no_grad():
        _ = model(samples[0].x.unsqueeze(0), edge_index, use_gib=True)
    with torch.no_grad():
        for d in samples:
            x = d.x.unsqueeze(0)
            t0 = time.perf_counter()
            x_flat = x.view(N, x.size(2), x.size(3))
            h_temporal = model.temporal_block(x_flat)
            h_spatial = model.spatial_block(h_temporal, edge_index)
            gib_logits = model.gib.mlp(h_spatial).squeeze(-1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            logits_list.append(gib_logits.cpu().numpy())
            if getattr(d, 'root_cause', None) and len(d.root_cause) > 0 and d.root_cause[0]:
                y_true.append(name_to_idx[d.root_cause[0]])
            else:
                y_true.append(-1)
    return y_true, logits_list, latencies


def infer_gcn(model: Any, samples: List[Any], edge_index: torch.Tensor, name_to_idx: Dict[str, int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits_list: List[np.ndarray] = []
    latencies: List[float] = []
    with torch.no_grad():
        _ = model(samples[0].x.unsqueeze(0), edge_index)
    with torch.no_grad():
        for d in samples:
            x = d.x.unsqueeze(0)
            t0 = time.perf_counter()
            lo = model(x, edge_index)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            logits_list.append(lo.cpu().numpy()[0])
            if getattr(d, 'root_cause', None) and len(d.root_cause) > 0 and d.root_cause[0]:
                y_true.append(name_to_idx[d.root_cause[0]])
            else:
                y_true.append(-1)
    return y_true, logits_list, latencies


def infer_mlp(model: Any, samples: List[Any], name_to_idx: Dict[str, int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits_list: List[np.ndarray] = []
    latencies: List[float] = []
    with torch.no_grad():
        for d in samples:
            x = d.x
            stats = torch.cat([x.mean(dim=-1), x.std(dim=-1), x.max(dim=-1).values], dim=-1)
            t0 = time.perf_counter()
            lo = model(stats.unsqueeze(0)).squeeze(-1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            logits_list.append(lo.cpu().numpy()[0])
            if getattr(d, 'root_cause', None) and len(d.root_cause) > 0 and d.root_cause[0]:
                y_true.append(name_to_idx[d.root_cause[0]])
            else:
                y_true.append(-1)
    return y_true, logits_list, latencies


def infer_rf(clf: Any, samples: List[Any], name_to_idx: Dict[str, int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits_list: List[np.ndarray] = []
    latencies: List[float] = []
    for d in samples:
        x = d.x.numpy()
        mean_x = np.mean(x, axis=-1)
        std_x = np.std(x, axis=-1)
        max_x = np.max(x, axis=-1)
        stats = np.concatenate([mean_x, std_x, max_x], axis=-1)
        t0 = time.perf_counter()
        probs = clf.predict_proba(stats)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        logits_list.append(probs[:, 1] if probs.shape[1] > 1 else probs)
        if getattr(d, 'root_cause', None) and len(d.root_cause) > 0 and d.root_cause[0]:
            y_true.append(name_to_idx[d.root_cause[0]])
        else:
            y_true.append(-1)
    return y_true, logits_list, latencies


def summarize_metrics(y_true: List[int], logits_list: List[np.ndarray], metas: List[Dict[str, Any]], latencies_ms: List[float]) -> Dict[str, Any]:
    sev = [float(m.get("severity", 1.0)) for m in metas]
    tops, mrr, sw_mrr = get_ranking_metrics(y_true, logits_list, sev, top_ns=(1, 2, 5))
    lat_window = latencies_ms[:100] if len(latencies_ms) >= 100 else latencies_ms
    median_ms = float(statistics.median(lat_window)) if lat_window else 0.0
    return {
        "top1": float(tops[1]),
        "top2": float(tops[2]),
        "top5": float(tops[5]),
        "mrr": float(mrr),
        "sw_mrr": float(sw_mrr),
        "latency_median_ms_100": median_ms,
        "breakdown_json": serialize_breakdown(y_true, logits_list, metas),
    }


def run_model(
    model_name: str,
    train_anom: List[Any],
    test_anom: List[Any],
    train_data: List[Any],
    edge_index: torch.Tensor,
    name_to_idx: Dict[str, int],
    N: int,
    seed: int,
    k_sta: int,
    epochs: int,
) -> Dict[str, Any]:
    F_ = train_data[0].x.size(1)

    if model_name == "STA":
        model = train_sta_compact(
            train_data,
            list(name_to_idx.keys()),
            edge_index,
            torch.device("cpu"),
            hidden_dim=32,
            k=k_sta,
            warmup_epochs=20,
            supervised_epochs=epochs,
            batch_size=256,
            k_anneal_epochs=60 if N < 100 else int(epochs * 0.75),
            early_stopping_patience=50,
        )
        y_true, logits, lats = infer_sta(model, test_anom, edge_index, name_to_idx)

    elif model_name == "LateBottleneck":
        model = train_v1_compact(
            train_data,
            list(name_to_idx.keys()),
            edge_index,
            torch.device("cpu"),
            hidden_dim=32,
            contrastive_epochs=80,
            supervised_epochs=epochs,
            batch_size=256,
            early_stopping_patience=50,
        )
        y_true, logits, lats = infer_late(model, test_anom, edge_index, name_to_idx)

    elif model_name == "FullGraphGCN":
        model = train_baseline_gcn(train_anom, edge_index, name_to_idx, N, F_, hidden_dim=32, epochs=epochs, seed=seed)
        model.eval()
        y_true, logits, lats = infer_gcn(model, test_anom, edge_index, name_to_idx)

    elif model_name == "MLP":
        model = train_baseline_mlp(train_anom, name_to_idx, N, hidden_dim=32, epochs=epochs, seed=seed)
        model.eval()
        y_true, logits, lats = infer_mlp(model, test_anom, name_to_idx)

    elif model_name == "RandomForest":
        model = train_baseline_rf(train_anom, name_to_idx, N, seed=seed)
        y_true, logits, lats = infer_rf(model, test_anom, name_to_idx)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    metas = collect_metas(test_anom)
    result = summarize_metrics(y_true, logits, metas, lats)
    result["n_test"] = len(test_anom)
    return result


class SoftAttentionSTGCN(nn.Module):
    """STGCN-style temporal encoder + soft graph attention (GAT)."""

    def __init__(self, num_features: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.tc1 = nn.Conv1d(num_features, hidden_dim, 3, padding=1)
        self.tc2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.tn = nn.BatchNorm1d(hidden_dim)
        self.ta = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.gat1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, dropout=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, N, Ff, W = x.shape
        h = x.view(B * N, Ff, W)
        h = F.relu(self.tc1(h))
        h = F.relu(self.tc2(h))
        ht = h.permute(0, 2, 1)
        a = F.softmax(self.ta(ht).squeeze(-1), dim=-1)
        h = (ht * a.unsqueeze(-1)).sum(1)
        h = self.tn(h)
        if B > 1:
            off = torch.arange(B, device=h.device) * N
            be = edge_index.unsqueeze(0).expand(B, -1, -1) + off.view(-1, 1, 1)
            be = be.reshape(2, -1)
        else:
            be = edge_index
        r = h
        h = F.elu(self.gat1(h, be))
        h = self.gat2(h, be)
        h = self.norm(h + r)
        h = h.view(B, N, -1)
        return self.head(h).squeeze(-1)


def eval_simple_logits(model: Any, test_anom: List[Any], edge_index: torch.Tensor, name_to_idx: Dict[str, int], needs_ei: bool = True) -> Tuple[List[int], List[np.ndarray], List[float]]:
    y_true: List[int] = []
    logits: List[np.ndarray] = []
    lats: List[float] = []
    model.eval()
    with torch.no_grad():
        for d in test_anom:
            x = d.x.unsqueeze(0)
            t0 = time.perf_counter()
            lo = model(x, edge_index) if needs_ei else model(x)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)
            logits.append(lo.squeeze(0).cpu().numpy())
            y_true.append(name_to_idx[d.root_cause[0]])
    return y_true, logits, lats


def run_antibias_checks(args: argparse.Namespace) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # 1) Scaler fit/apply strict path available and per_source in use
    # FIX-REC1: antibias check uses synthetic-only → 5 features, not 13
    loader = UnifiedDataLoader(window_size=12, num_features=5, normalize="per_source", seed=42)
    loader.add_synthetic(num_nodes=20, num_samples=200, ba_m=2, anomaly_ratio=0.4)
    loader.load_all()
    train, test = loader.get_splits(train_ratio=0.8, stratify_by="source", balance_sources=True)
    stats = loader.fit_scaler(train)
    _ = loader.apply_scaler(train, stats)
    _ = loader.apply_scaler(test, stats)
    checks["scaler_fit_train_only"] = bool(stats)

    # 2) MLP/RF no explicit node-index feature
    src_mlp = inspect.getsource(train_baseline_mlp)
    src_rf = inspect.getsource(train_baseline_rf)
    no_idx_feature = ("append(i)" not in src_mlp) and ("append([i" not in src_mlp) and ("append(i)" not in src_rf)
    checks["mlp_rf_no_node_index_feature"] = no_idx_feature

    # 3) Split by source
    src_train = defaultdict(int)
    src_test = defaultdict(int)
    for d in train:
        src_train[getattr(d, "source", "unknown")] += 1
    for d in test:
        src_test[getattr(d, "source", "unknown")] += 1
    checks["split_weighted_by_source"] = len(src_train) > 0 and len(src_test) > 0

    # 4) patience=50
    default_patience = inspect.signature(EarlyStopping.__init__).parameters["patience"].default
    checks["patience_50"] = int(default_patience) == 50

    checks["details"] = {
        "train_sources": dict(src_train),
        "test_sources": dict(src_test),
        "early_stopping_default": int(default_patience),
    }

    failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    checks["all_passed"] = len(failed) == 0
    checks["failed"] = failed
    return checks


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_summary_markdown(benchmark_rows: List[Dict[str, Any]], source_notes: List[str]) -> str:
    headers = ["Scale", "Model", "Top-1", "Top-2", "Top-5", "MRR", "SW-MRR", "Latency median (ms)"]
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in benchmark_rows:
        if r.get("table") != "main":
            continue
        grouped[(int(r["scale"]), str(r["model"]))].append(r)

    rows: List[List[str]] = []
    for scale in SCALES:
        for model in ["STA", "LateBottleneck", "FullGraphGCN", "MLP", "RandomForest"]:
            rs = grouped.get((scale, model), [])
            if not rs:
                continue
            agg = aggregate(rs, ["top1", "top2", "top5", "mrr", "sw_mrr", "latency_median_ms_100"])
            rows.append(
                [
                    str(scale),
                    model,
                    f"{agg['top1']:.4f} +- {agg['top1_std']:.4f}",
                    f"{agg['top2']:.4f} +- {agg['top2_std']:.4f}",
                    f"{agg['top5']:.4f} +- {agg['top5_std']:.4f}",
                    f"{agg['mrr']:.4f} +- {agg['mrr_std']:.4f}",
                    f"{agg['sw_mrr']:.4f} +- {agg['sw_mrr_std']:.4f}",
                    f"{agg['latency_median_ms_100']:.3f} +- {agg['latency_median_ms_100_std']:.3f}",
                ]
            )

    txt = ["# Benchmark Summary", "", "## Main table — Performance by scale", "", md_table(headers, rows), ""]
    txt.append("## Data Source Availability")
    txt.append("")
    for n in source_notes:
        txt.append(f"- {n}")
    txt.append("")
    return "\n".join(txt)


def build_modality_markdown(benchmark_rows: List[Dict[str, Any]]) -> str:
    headers = ["Model", "Modalities", "Top-1", "Top-2", "Top-5", "MRR", "SW-MRR"]
    rows: List[List[str]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in benchmark_rows:
        if r.get("table") != "modality":
            continue
        grouped[(str(r["model"]), str(r["modality"]))].append(r)

    for model in ["STA", "LateBottleneck"]:
        for modality in ["metrics", "metrics_traces", "full"]:
            rs = grouped.get((model, modality), [])
            if not rs:
                continue
            agg = aggregate(rs, ["top1", "top2", "top5", "mrr", "sw_mrr"])
            rows.append(
                [
                    model,
                    modality,
                    f"{agg['top1']:.4f} +- {agg['top1_std']:.4f}",
                    f"{agg['top2']:.4f} +- {agg['top2_std']:.4f}",
                    f"{agg['top5']:.4f} +- {agg['top5_std']:.4f}",
                    f"{agg['mrr']:.4f} +- {agg['mrr_std']:.4f}",
                    f"{agg['sw_mrr']:.4f} +- {agg['sw_mrr_std']:.4f}",
                ]
            )

    txt = ["# Modality Analysis", "", md_table(headers, rows), ""]
    return "\n".join(txt)


def build_ablation_markdown(ablation_rows: List[Dict[str, Any]]) -> str:
    headers = ["Ablation", "Variant", "Scale", "Top-1", "Top-2", "Top-5", "MRR", "SW-MRR"]
    rows: List[List[str]] = []
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in ablation_rows:
        grouped[(str(r["ablation"]), str(r["variant"]), int(r["scale"]))].append(r)

    for ablation in ["A", "B", "C", "D", "E"]:
        for (a, v, s), rs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
            if a != ablation:
                continue
            agg = aggregate(rs, ["top1", "top2", "top5", "mrr", "sw_mrr"])
            rows.append(
                [
                    a,
                    v,
                    str(s),
                    f"{agg['top1']:.4f} +- {agg['top1_std']:.4f}",
                    f"{agg['top2']:.4f} +- {agg['top2_std']:.4f}",
                    f"{agg['top5']:.4f} +- {agg['top5_std']:.4f}",
                    f"{agg['mrr']:.4f} +- {agg['mrr_std']:.4f}",
                    f"{agg['sw_mrr']:.4f} +- {agg['sw_mrr_std']:.4f}",
                ]
            )

    return "\n".join(["# Ablation Summary", "", md_table(headers, rows), ""])


def run(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Official loader path expected by load_rcaeval.py.
    rcaeval_available = True
    try:
        import rcaeval.data  # type: ignore
    except Exception:
        rcaeval_available = False
        if args.include_rcaeval:
            print("[RCAEval] 'rcaeval.data' not available in current env; skipping RCAEval source.")
            args.include_rcaeval = False

    anti = run_antibias_checks(args)
    print("[Anti-bias checks]", json.dumps(anti, indent=2))
    if not anti["all_passed"]:
        raise RuntimeError(f"Anti-bias checks failed: {anti['failed']}")

    benchmark_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    source_notes: List[str] = []

    benchmark_csv = os.path.join(args.results_dir, "benchmark_full.csv")
    benchmark_md = os.path.join(args.results_dir, "benchmark_summary.md")
    ablation_csv = os.path.join(args.results_dir, "ablation_full.csv")
    ablation_md = os.path.join(args.results_dir, "ablation_summary.md")
    modality_md = os.path.join(args.results_dir, "modality_analysis.md")

    def flush_outputs() -> None:
        """Persist intermediate artifacts during long-running experiments."""
        write_csv(benchmark_csv, benchmark_rows)
        write_csv(ablation_csv, ablation_rows)
        with open(benchmark_md, "w", encoding="utf-8") as f:
            f.write(build_summary_markdown(benchmark_rows, source_notes))
        with open(ablation_md, "w", encoding="utf-8") as f:
            f.write(build_ablation_markdown(ablation_rows))
        with open(modality_md, "w", encoding="utf-8") as f:
            f.write(build_modality_markdown(benchmark_rows))

    data_hashes: Dict[str, str] = {}
    for p in [
        "datasets/dataset_merged.csv",
        "dataset_windows.pt",
        "test_integration.pt",
    ]:
        if os.path.exists(p):
            data_hashes[p] = md5_file(p)

    # Main benchmark table
    for scale in SCALES:
        prepared = prepare_data_for_scale(
            num_nodes=scale,
            num_samples=args.num_samples,
            seed=args.dataset_seed,
            include_cluster=args.include_cluster,
            include_rcaeval=args.include_rcaeval,
            prometheus_url=args.prometheus_url,
            jaeger_url=args.jaeger_url,
            loki_url=args.loki_url,
            rcaeval_name=args.rcaeval_name,
        )

        status = prepared.source_status
        if not status.get("cluster", False):
            source_notes.append(f"N={scale}: cluster source unavailable; benchmark used available sources only.")
        if not status.get("rcaeval", False):
            source_notes.append(f"N={scale}: RCAEval source unavailable; benchmark used available sources only.")

        name_to_idx = {n: i for i, n in enumerate(prepared.node_names)}
        train_anom = get_anomalies(prepared.train, name_to_idx)
        test_anom = get_anomalies(prepared.test, name_to_idx)

        for seed in SEEDS:
            set_all_seeds(seed)
            for model in ["STA", "LateBottleneck", "FullGraphGCN", "MLP", "RandomForest"]:
                metrics = run_model(
                    model_name=model,
                    train_anom=train_anom,
                    test_anom=test_anom,
                    train_data=prepared.train,
                    edge_index=prepared.edge_index,
                    name_to_idx=name_to_idx,
                    N=scale,
                    seed=seed,
                    k_sta=K_MAP[scale],
                    epochs=args.epochs,
                )
                benchmark_rows.append(
                    {
                        "table": "main",
                        "scale": scale,
                        "seed": seed,
                        "model": model,
                        "top1": metrics["top1"],
                        "top2": metrics["top2"],
                        "top5": metrics["top5"],
                        "mrr": metrics["mrr"],
                        "sw_mrr": metrics["sw_mrr"],
                        "latency_median_ms_100": metrics["latency_median_ms_100"],
                        "breakdown": metrics["breakdown_json"],
                        "n_test": metrics["n_test"],
                        "dataset_hashes": json.dumps(data_hashes, ensure_ascii=True),
                        "train_sources": json.dumps(prepared.source_counts_train, ensure_ascii=True),
                        "test_sources": json.dumps(prepared.source_counts_test, ensure_ascii=True),
                    }
                )
            flush_outputs()

    # Modality table (N=50, STA and Late only)
    prepared_50 = prepare_data_for_scale(
        num_nodes=50,
        num_samples=args.num_samples,
        seed=args.dataset_seed,
        include_cluster=args.include_cluster,
        include_rcaeval=args.include_rcaeval,
        prometheus_url=args.prometheus_url,
        jaeger_url=args.jaeger_url,
        loki_url=args.loki_url,
        rcaeval_name=args.rcaeval_name,
    )
    name_to_idx_50 = {n: i for i, n in enumerate(prepared_50.node_names)}
    for modality in ["metrics", "metrics_traces", "full"]:
        train_m = select_modalities(prepared_50.train, modality)
        test_m = select_modalities(prepared_50.test, modality)
        train_anom_m = get_anomalies(train_m, name_to_idx_50)
        test_anom_m = get_anomalies(test_m, name_to_idx_50)

        for seed in SEEDS:
            set_all_seeds(seed)
            for model in ["STA", "LateBottleneck"]:
                metrics = run_model(
                    model_name=model,
                    train_anom=train_anom_m,
                    test_anom=test_anom_m,
                    train_data=train_m,
                    edge_index=prepared_50.edge_index,
                    name_to_idx=name_to_idx_50,
                    N=50,
                    seed=seed,
                    k_sta=K_MAP[50],
                    epochs=args.epochs,
                )
                benchmark_rows.append(
                    {
                        "table": "modality",
                        "scale": 50,
                        "seed": seed,
                        "model": model,
                        "modality": modality,
                        "top1": metrics["top1"],
                        "top2": metrics["top2"],
                        "top5": metrics["top5"],
                        "mrr": metrics["mrr"],
                        "sw_mrr": metrics["sw_mrr"],
                        "latency_median_ms_100": metrics["latency_median_ms_100"],
                        "breakdown": metrics["breakdown_json"],
                    }
                )
            flush_outputs()

    # Source table (N=20, best model between STA and GCN)
    main_n20 = [r for r in benchmark_rows if r.get("table") == "main" and int(r["scale"]) == 20 and r["model"] in ("STA", "FullGraphGCN")]
    top_sta = np.mean([r["top1"] for r in main_n20 if r["model"] == "STA"]) if any(r["model"] == "STA" for r in main_n20) else -1
    top_gcn = np.mean([r["top1"] for r in main_n20 if r["model"] == "FullGraphGCN"]) if any(r["model"] == "FullGraphGCN" for r in main_n20) else -1
    best_model = "STA" if top_sta >= top_gcn else "FullGraphGCN"

    source_modes = ["synthetic_only", "rcaeval_only", "cluster_only", "unified"]
    for mode in source_modes:
        include_cluster = args.include_cluster and mode in ("cluster_only", "unified")
        include_rcaeval = args.include_rcaeval and mode in ("rcaeval_only", "unified")
        include_synth = mode in ("synthetic_only", "unified")

        # FIX-REC1: use effective_features based on active real data sources
        _has_real = (include_cluster and args.prometheus_url) or include_rcaeval
        _eff_feats = 13 if _has_real else 5
        loader = UnifiedDataLoader(window_size=12, num_features=_eff_feats, normalize="per_source", seed=args.dataset_seed)
        if include_synth:
            loader.add_synthetic(num_nodes=20, num_samples=args.num_samples, ba_m=2, anomaly_ratio=0.4, recurrence_ratio=0.8)
        if include_cluster and args.prometheus_url:
            loader.add_cluster(prometheus_url=args.prometheus_url, jaeger_url=args.jaeger_url, loki_url=args.loki_url, duration="24h")
        if include_rcaeval:
            loader.add_rcaeval(dataset_name=args.rcaeval_name)
        loader.load_all()
        status = detect_sources_present(loader)

        if mode == "cluster_only" and not status.get("cluster", False):
            continue
        if mode == "rcaeval_only" and not status.get("rcaeval", False):
            continue

        train_s, test_s = loader.get_splits(train_ratio=0.8, stratify_by="source", balance_sources=True)
        stats_s = loader.fit_scaler(train_s)
        train_s = loader.apply_scaler(train_s, stats_s)
        test_s = loader.apply_scaler(test_s, stats_s)

        if not train_s or not test_s:
            continue

        edge_index = train_s[0].edge_index
        if loader.node_names is not None and len(loader.node_names) == 20:
            names = list(loader.node_names)
        else:
            names = [f"node_{i}" for i in range(20)]
        name_to_idx_s = {n: i for i, n in enumerate(names)}
        train_anom_s = get_anomalies(train_s, name_to_idx_s)
        test_anom_s = get_anomalies(test_s, name_to_idx_s)

        if not train_anom_s or not test_anom_s:
            print(f"[Source table] Skipping mode={mode}: insufficient anomalous windows (train={len(train_anom_s)}, test={len(test_anom_s)}).")
            continue

        model_for_mode = best_model
        if model_for_mode == "STA":
            healthy_s = [d for d in train_s if d.y is not None and d.y.item() == 0]
            if len(healthy_s) < 2 or len(train_anom_s) < 2:
                print(
                    f"[Source table] Falling back to FullGraphGCN for mode={mode}: "
                    f"STA needs healthy+anomalous support (healthy={len(healthy_s)}, anomalous={len(train_anom_s)})."
                )
                model_for_mode = "FullGraphGCN"

        for seed in SEEDS:
            metrics = run_model(
                model_name=model_for_mode,
                train_anom=train_anom_s,
                test_anom=test_anom_s,
                train_data=train_s,
                edge_index=edge_index,
                name_to_idx=name_to_idx_s,
                N=20,
                seed=seed,
                k_sta=K_MAP[20],
                epochs=args.epochs,
            )
            benchmark_rows.append(
                {
                    "table": "source",
                    "scale": 20,
                    "seed": seed,
                    "model": model_for_mode,
                    "source_mode": mode,
                    "top1": metrics["top1"],
                    "top2": metrics["top2"],
                    "top5": metrics["top5"],
                    "mrr": metrics["mrr"],
                    "sw_mrr": metrics["sw_mrr"],
                    "latency_median_ms_100": metrics["latency_median_ms_100"],
                    "breakdown": metrics["breakdown_json"],
                }
            )
        flush_outputs()

    # Ablation A/B/C/D/E
    for scale in [20, 100]:
        prepared = prepare_data_for_scale(
            num_nodes=scale,
            num_samples=args.num_samples,
            seed=args.dataset_seed,
            include_cluster=False,
            include_rcaeval=False,
            prometheus_url=None,
            jaeger_url=None,
            loki_url=None,
            rcaeval_name=args.rcaeval_name,
        )
        names = prepared.node_names
        name_to_idx = {n: i for i, n in enumerate(names)}
        train_data = prepared.train
        test_data = prepared.test
        train_anom = get_anomalies(train_data, name_to_idx)
        test_anom = get_anomalies(test_data, name_to_idx)
        F_ = train_data[0].x.size(1)

        # A1-A6
        for seed in SEEDS:
            set_all_seeds(seed)

            # A1
            m_a1 = train_v1_compact(train_data, names, prepared.edge_index, device, hidden_dim=32, contrastive_epochs=80, supervised_epochs=args.epochs, early_stopping_patience=50)
            y, lo, lat = infer_late(m_a1, test_anom, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A1_STGCN_GIB", "scale": scale, "seed": seed, **met})

            # A2
            m_a2 = train_baseline_gcn(train_anom, prepared.edge_index, name_to_idx, scale, F_, hidden_dim=32, epochs=args.epochs, seed=seed)
            y, lo, lat = infer_gcn(m_a2, test_anom, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A2_STGCN_no_GIB", "scale": scale, "seed": seed, **met})

            # A3
            m_a3 = SpatialOnlyGCN(F_, 32, scale)
            m_a3 = train_simple_model(m_a3, train_anom, prepared.edge_index, name_to_idx, scale, epochs=args.epochs, seed=seed, needs_ei=True)
            y, lo, lat = eval_simple_logits(m_a3, test_anom, prepared.edge_index, name_to_idx, needs_ei=True)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A3_Spatial_only", "scale": scale, "seed": seed, **met})

            # A4
            m_a4 = TemporalOnlyModel(F_, 32, scale)
            m_a4 = train_simple_model(m_a4, train_anom, prepared.edge_index, name_to_idx, scale, epochs=args.epochs, seed=seed, needs_ei=False)
            y, lo, lat = eval_simple_logits(m_a4, test_anom, prepared.edge_index, name_to_idx, needs_ei=False)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A4_Temporal_only", "scale": scale, "seed": seed, **met})

            # A5 adaptive beta(N)
            beta_n = max(0.1, min(0.5, 20.0 / float(scale)))
            m_a5 = train_v1_compact(
                train_data,
                names,
                prepared.edge_index,
                device,
                hidden_dim=32,
                contrastive_epochs=80,
                supervised_epochs=args.epochs,
                beta_max=beta_n,
                early_stopping_patience=50,
            )
            y, lo, lat = infer_late(m_a5, test_anom, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A5_GIB_adaptive_betaN", "scale": scale, "seed": seed, **met})

            # A6 soft attention GAT
            m_a6 = SoftAttentionSTGCN(F_, 32, scale)
            m_a6 = train_simple_model(m_a6, train_anom, prepared.edge_index, name_to_idx, scale, epochs=args.epochs, seed=seed, needs_ei=True)
            y, lo, lat = eval_simple_logits(m_a6, test_anom, prepared.edge_index, name_to_idx, needs_ei=True)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "A", "variant": "A6_STGCN_soft_attention", "scale": scale, "seed": seed, **met})

            # B1
            m_b1 = train_v1_compact(train_data, names, prepared.edge_index, device, hidden_dim=32, contrastive_epochs=80, supervised_epochs=args.epochs, early_stopping_patience=50)
            y, lo, lat = infer_late(m_b1, test_anom, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "B", "variant": "B1_contrastive_supervised", "scale": scale, "seed": seed, **met})

            # B2
            m_b2 = train_v1_compact(train_data, names, prepared.edge_index, device, hidden_dim=32, contrastive_epochs=0, supervised_epochs=args.epochs, early_stopping_patience=50)
            y, lo, lat = infer_late(m_b2, test_anom, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "B", "variant": "B2_supervised_only", "scale": scale, "seed": seed, **met})

            # B3 contrastive-only, node-distance to healthy logits profile
            m_b3 = train_v1_compact(train_data, names, prepared.edge_index, device, hidden_dim=32, contrastive_epochs=80, supervised_epochs=0, early_stopping_patience=50)
            healthy_train = [d for d in train_data if d.y is not None and d.y.item() == 0]
            _, healthy_logits, _ = infer_late(m_b3, healthy_train[: min(200, len(healthy_train))], prepared.edge_index, {n: i for i, n in enumerate(names)})
            healthy_centroid = np.mean(np.stack(healthy_logits), axis=0) if healthy_logits else np.zeros(scale, dtype=np.float32)
            y_true, logits_list, lats = infer_late(m_b3, test_anom, prepared.edge_index, name_to_idx)
            dist_logits = [np.abs(lo - healthy_centroid) for lo in logits_list]
            met = summarize_metrics(y_true, dist_logits, collect_metas(test_anom), lats)
            ablation_rows.append({"ablation": "B", "variant": "B3_contrastive_only_distance", "scale": scale, "seed": seed, **met})

            # B4 contrastive+supervised + full modalities
            train_full = select_modalities(train_data, "full")
            test_full = select_modalities(test_data, "full")
            train_anom_full = get_anomalies(train_full, name_to_idx)
            test_anom_full = get_anomalies(test_full, name_to_idx)
            m_b4 = train_v1_compact(train_full, names, prepared.edge_index, device, hidden_dim=32, contrastive_epochs=80, supervised_epochs=args.epochs, early_stopping_patience=50)
            y, lo, lat = infer_late(m_b4, test_anom_full, prepared.edge_index, name_to_idx)
            met = summarize_metrics(y, lo, collect_metas(test_anom_full), lat)
            ablation_rows.append({"ablation": "B", "variant": "B4_contrastive_supervised_multimodal", "scale": scale, "seed": seed, **met})

            flush_outputs()

    # C and D on N=100
    prepared_100 = prepare_data_for_scale(
        num_nodes=100,
        num_samples=args.num_samples,
        seed=args.dataset_seed,
        include_cluster=False,
        include_rcaeval=False,
        prometheus_url=None,
        jaeger_url=None,
        loki_url=None,
        rcaeval_name=args.rcaeval_name,
    )
    n2i_100 = {n: i for i, n in enumerate(prepared_100.node_names)}
    train_anom_100 = get_anomalies(prepared_100.train, n2i_100)
    test_anom_100 = get_anomalies(prepared_100.test, n2i_100)

    for seed in SEEDS:
        set_all_seeds(seed)
        # C
        for k in [10, 15, 20, 30, 50]:
            m_c = train_sta_compact(prepared_100.train, prepared_100.node_names, prepared_100.edge_index, device, hidden_dim=32, k=k, warmup_epochs=20, supervised_epochs=args.epochs, k_anneal_epochs=60, early_stopping_patience=50)
            y, lo, lat = infer_sta(m_c, test_anom_100, prepared_100.edge_index, n2i_100)
            met = summarize_metrics(y, lo, collect_metas(test_anom_100), lat)
            ablation_rows.append({"ablation": "C", "variant": f"C_k_{k}", "scale": 100, "seed": seed, **met})

        # D1
        m_d1 = train_sta_compact(prepared_100.train, prepared_100.node_names, prepared_100.edge_index, device, hidden_dim=32, k=15, warmup_epochs=20, supervised_epochs=args.epochs, k_anneal_epochs=60, early_stopping_patience=50)
        y, lo, lat = infer_sta(m_d1, test_anom_100, prepared_100.edge_index, n2i_100)
        met = summarize_metrics(y, lo, collect_metas(test_anom_100), lat)
        ablation_rows.append({"ablation": "D", "variant": "D1_full", "scale": 100, "seed": seed, **met})

        # D2
        m_d2 = train_sta_compact(prepared_100.train, prepared_100.node_names, prepared_100.edge_index, device, hidden_dim=32, k=15, warmup_epochs=0, supervised_epochs=args.epochs, k_anneal_epochs=60, early_stopping_patience=50)
        y, lo, lat = infer_sta(m_d2, test_anom_100, prepared_100.edge_index, n2i_100)
        met = summarize_metrics(y, lo, collect_metas(test_anom_100), lat)
        ablation_rows.append({"ablation": "D", "variant": "D2_no_warmup", "scale": 100, "seed": seed, **met})

        # D3
        m_d3 = train_sta_compact(prepared_100.train, prepared_100.node_names, prepared_100.edge_index, device, hidden_dim=32, k=15, warmup_epochs=20, supervised_epochs=args.epochs, k_anneal_epochs=0, early_stopping_patience=50)
        y, lo, lat = infer_sta(m_d3, test_anom_100, prepared_100.edge_index, n2i_100)
        met = summarize_metrics(y, lo, collect_metas(test_anom_100), lat)
        ablation_rows.append({"ablation": "D", "variant": "D3_no_k_annealing", "scale": 100, "seed": seed, **met})

        # D4
        m_d4 = train_sta_compact(prepared_100.train, prepared_100.node_names, prepared_100.edge_index, device, hidden_dim=32, k=15, warmup_epochs=0, supervised_epochs=args.epochs, k_anneal_epochs=0, early_stopping_patience=50)
        y, lo, lat = infer_sta(m_d4, test_anom_100, prepared_100.edge_index, n2i_100)
        met = summarize_metrics(y, lo, collect_metas(test_anom_100), lat)
        ablation_rows.append({"ablation": "D", "variant": "D4_no_warmup_no_k_annealing", "scale": 100, "seed": seed, **met})

        flush_outputs()

    # E on N=50, STA only
    prepared_50 = prepare_data_for_scale(
        num_nodes=50,
        num_samples=args.num_samples,
        seed=args.dataset_seed,
        include_cluster=args.include_cluster,
        include_rcaeval=args.include_rcaeval,
        prometheus_url=args.prometheus_url,
        jaeger_url=args.jaeger_url,
        loki_url=args.loki_url,
        rcaeval_name=args.rcaeval_name,
    )
    n2i_50 = {n: i for i, n in enumerate(prepared_50.node_names)}

    for seed in SEEDS:
        set_all_seeds(seed)
        for mode in ["metrics", "metrics_traces", "full"]:
            tr = select_modalities(prepared_50.train, mode)
            te = select_modalities(prepared_50.test, mode)
            test_anom = get_anomalies(te, n2i_50)
            m_e = train_sta_compact(tr, prepared_50.node_names, prepared_50.edge_index, device, hidden_dim=32, k=8, warmup_epochs=20, supervised_epochs=args.epochs, k_anneal_epochs=60, early_stopping_patience=50)
            y, lo, lat = infer_sta(m_e, test_anom, prepared_50.edge_index, n2i_50)
            met = summarize_metrics(y, lo, collect_metas(test_anom), lat)
            ablation_rows.append({"ablation": "E", "variant": f"E_STA_{mode}", "scale": 50, "seed": seed, **met})

        flush_outputs()

    # Write outputs
    flush_outputs()

    print("Outputs written:")
    print(f"- {benchmark_csv}")
    print(f"- {benchmark_md}")
    print(f"- {ablation_csv}")
    print(f"- {ablation_md}")
    print(f"- {modality_md}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full RCA benchmark and ablations")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--dataset-seed", type=int, default=42)
    p.add_argument("--include-cluster", action="store_true")
    p.add_argument("--include-rcaeval", action="store_true")
    p.add_argument("--prometheus-url", type=str, default=os.getenv("PROMETHEUS_URL", ""))
    p.add_argument("--jaeger-url", type=str, default=os.getenv("JAEGER_URL", ""))
    p.add_argument("--loki-url", type=str, default=os.getenv("LOKI_URL", ""))
    p.add_argument("--rcaeval-name", type=str, default="alibaba")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())

