#!/usr/bin/env python3
"""
benchmark_real.py — Validate RCA models on RCAEval real-world datasets.

Bridges the gap between simulated BA benchmarks and production realism (§7.1).

Usage:
    python3 benchmark_real.py --dataset trainticket --seeds 42,123,7
    python3 benchmark_real.py --dataset all
    python3 benchmark_real.py --observit-csv path/to/dataset.csv   # ou répertoire dataset_*.csv
"""

import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rca_pipeline.common import (
    runtime_env_info,
    set_all_seeds,
    set_torch_deterministic,
    stratified_split_shuffle,
)
from rca_pipeline.drift_detector import make_k8s_drift_detector
from rca_pipeline.graph_builder import GraphBuilder
from rca_pipeline.k_selection import get_optimal_k
from rca_pipeline.loaders.rcaeval_loader import SUPPORTED_DATASETS, RCAEvalLoader
from rca_pipeline.losses import info_nce_loss, variational_gib_loss
from rca_pipeline.metrics_rcaeval import (
    RCAEVAL_PUBLISHED_BASELINES,
    RCAEvalMetrics,
    bootstrap_ci,
    compare_with_rcaeval_baselines,
    top1_per_service,
    top1_per_service_detailed,
)
from rca_pipeline.model_late_gat import SpatioTemporalRCA_GAT
from rca_pipeline.model_late_gat_mm import SpatioTemporalRCA_GAT_MM
from rca_pipeline.model_sta import SelectThenAnalyze
from rca_pipeline.model_sta_mm import SelectThenAnalyze_MM

logging.basicConfig(level=logging.WARNING)


def _clone_pyg_data(d):
    """Clone manuel des `Data` PyG (évite soucis `Data.clone()` selon versions)."""
    from torch_geometric.data import Data

    out = Data(x=d.x.clone(), edge_index=d.edge_index.clone())
    if hasattr(d, "y") and d.y is not None:
        out.y = d.y.clone()
    if hasattr(d, "root_cause"):
        out.root_cause = list(d.root_cause)
    for attr in ("x_logs", "x_traces"):
        if hasattr(d, attr) and getattr(d, attr) is not None:
            setattr(out, attr, getattr(d, attr).clone())
    return out


def _sample_to_metric_dict(x: torch.Tensor) -> Dict[str, float]:
    """
    Convert x [N, F, W] to the 6 canonical metric keys expected by drift detector.
    """
    x_np = x.detach().cpu().numpy()
    agg = x_np.mean(axis=(0, 2)) if x_np.size else np.zeros((0,), dtype=np.float32)
    names = ["cpu_usage", "memory_bytes", "network_rx", "network_tx", "latency_p99", "error_rate"]
    vals: Dict[str, float] = {}
    for i, n in enumerate(names):
        vals[n] = float(agg[i]) if i < len(agg) else 0.0
    return vals


def _maybe_apply_drift_filter(
    data: List,
    enable_drift: bool,
    drift_delta: float,
    drift_cooldown_steps: int,
) -> Tuple[List, Dict[str, float]]:
    """
    Streaming-like drift gate:
      - updates detector sequentially over samples
      - drops samples during cooldown windows
    """
    if not enable_drift:
        return data, {"drift_enabled": False, "dropped": 0, "kept": len(data), "drop_ratio": 0.0}

    detector = make_k8s_drift_detector(delta=drift_delta, cooldown_steps=drift_cooldown_steps)
    kept = []
    dropped = 0
    for d in data:
        detector.update_batch(_sample_to_metric_dict(d.x))
        if detector.is_in_cooldown():
            dropped += 1
            continue
        kept.append(d)
    ratio = dropped / max(len(data), 1)
    return kept, {
        "drift_enabled": True,
        "dropped": dropped,
        "kept": len(kept),
        "drop_ratio": ratio,
    }


def _stack_optional_features(data_list: List, attr: str) -> Optional[torch.Tensor]:
    if not data_list:
        return None
    if not hasattr(data_list[0], attr):
        return None
    return torch.stack([getattr(d, attr) for d in data_list])


def _has_attr_on_all(data_list: List, attr: str) -> bool:
    return bool(data_list) and all(hasattr(d, attr) for d in data_list)


def _merge_sta_logits(
    selector_logits: torch.Tensor,
    logits_full: torch.Tensor,
    top_k_indices: torch.Tensor,
    alpha: float = 0.15,
) -> torch.Tensor:
    """
    Build a dense ranking vector for STA:
      - keep selector logits on all nodes (global ranking support),
      - overwrite top-k nodes with deep scorer-blended logits from logits_full.
    """
    merged = selector_logits.clone()
    topk_vals = torch.gather(logits_full, 1, top_k_indices)
    merged.scatter_(1, top_k_indices, topk_vals)
    # Soft rerank to avoid destabilizing selector signal on real data.
    if alpha <= 0.0:
        return selector_logits
    if not torch.isfinite(merged).all():
        return selector_logits
    return (1.0 - alpha) * selector_logits + alpha * merged


def _mm_coverage(data_list: List) -> Dict[str, float]:
    if not data_list:
        return {"n": 0, "logs_ratio": 0.0, "traces_ratio": 0.0, "both_ratio": 0.0}
    n = len(data_list)
    eps = 1e-8

    def _has_signal(d, attr: str) -> bool:
        if not hasattr(d, attr):
            return False
        try:
            t = getattr(d, attr)
            if t is None:
                return False
            return float(t.std().item()) > eps
        except Exception:
            return False

    logs = sum(1 for d in data_list if _has_signal(d, "x_logs"))
    traces = sum(1 for d in data_list if _has_signal(d, "x_traces"))
    both = sum(1 for d in data_list if _has_signal(d, "x_logs") and _has_signal(d, "x_traces"))
    return {
        "n": n,
        "logs_ratio": logs / n,
        "traces_ratio": traces / n,
        "both_ratio": both / n,
    }


def _compute_mm_dataset_status(args, requested_seeds: List[int], skipped_seeds: List[int], executed_seeds: int) -> Dict[str, object]:
    if not bool(args.use_mm):
        return {
            "status": "not_applicable",
            "reason": "Multimodal mode is disabled for this run.",
            "requested_seeds": int(len(requested_seeds)),
            "executed_seeds": int(executed_seeds),
            "skipped_seeds": [int(s) for s in skipped_seeds],
        }

    if executed_seeds <= 0:
        if bool(args.mm_strict):
            reason = "MM strict required modality signal was missing on all requested seeds."
        else:
            reason = "No multimodal seed produced evaluable predictions."
        return {
            "status": "skipped_with_reason",
            "reason": reason,
            "requested_seeds": int(len(requested_seeds)),
            "executed_seeds": int(executed_seeds),
            "skipped_seeds": [int(s) for s in skipped_seeds],
        }

    if skipped_seeds:
        reason = (
            "Some seeds were skipped because required modality signal was missing; "
            "dataset remains comparable only on executed seeds."
        )
    else:
        reason = "All requested seeds executed with required modalities."
    return {
        "status": "comparable",
        "reason": reason,
        "requested_seeds": int(len(requested_seeds)),
        "executed_seeds": int(executed_seeds),
        "skipped_seeds": [int(s) for s in skipped_seeds],
    }


def _build_mm_global_summary(all_results: Dict[str, Dict]) -> Dict[str, object]:
    dataset_status: Dict[str, str] = {}
    blocked: List[str] = []
    for ds_name, ds_blob in all_results.items():
        if not isinstance(ds_blob, dict):
            continue
        cfg = ds_blob.get("config", {})
        if not isinstance(cfg, dict):
            continue
        status = cfg.get("mm_dataset_status")
        if status is None:
            continue
        dataset_status[str(ds_name)] = str(status)
        if status != "comparable":
            blocked.append(str(ds_name))

    allow = len(dataset_status) > 0 and len(blocked) == 0
    reason = (
        "All multimodal datasets are comparable."
        if allow
        else "Global MM aggregation blocked because at least one dataset is skipped or non-comparable."
    )
    return {
        "allow_global_aggregation": bool(allow),
        "reason": reason,
        "dataset_status": dataset_status,
        "blocked_by_datasets": blocked,
    }


def _enforce_real_warmup_guard(sta_contrastive_warmup: int, allow_real_warmup: bool) -> None:
    if int(sta_contrastive_warmup) <= 0:
        return
    if bool(allow_real_warmup):
        return
    raise SystemExit(
        "Refusing non-zero STA contrastive warmup on real datasets without explicit acknowledgement. "
        "Re-run with --allow-real-warmup only for controlled A/B analysis."
    )


def _class_counts(data: List, name_to_idx: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in data:
        if not hasattr(d, "root_cause") or not d.root_cause:
            continue
        rc = d.root_cause[0]
        if rc in name_to_idx:
            out[rc] = out.get(rc, 0) + 1
    return out


def _warn_split_quality(train_data: List, test_data: List, name_to_idx: Dict[str, int], seed: int) -> None:
    train_counts = _class_counts(train_data, name_to_idx)
    test_counts = _class_counts(test_data, name_to_idx)
    all_rc = sorted(set(train_counts) | set(test_counts))
    if not all_rc:
        print(f"  Seed {seed}: no labeled root-cause windows after split.")
        return
    only_train = [rc for rc in all_rc if train_counts.get(rc, 0) > 0 and test_counts.get(rc, 0) == 0]
    only_test = [rc for rc in all_rc if test_counts.get(rc, 0) > 0 and train_counts.get(rc, 0) == 0]
    if only_train:
        print(f"  Seed {seed}: classes with train-only support: {only_train}")
    if only_test:
        print(f"  Seed {seed}: classes with test-only support: {only_test}")


def _attach_per_root_metrics(
    ds_results: Dict,
    prefix: str,
    model_key: str,
    runs: List[Tuple[List[int], List[np.ndarray]]],
    node_names: List[str],
    rca_metrics: RCAEvalMetrics,
    dataset_name: str,
) -> None:
    """
    Attach per-root-cause Top-1 diagnostics and consistency checks.

    This is exported for all real datasets (RCAEval + ObservIT) to make class
    imbalance and per-service behavior auditable.
    """
    if not runs:
        return

    tol = 1e-6
    out_simple: List[Dict[str, float]] = []
    out_details: List[Dict[str, object]] = []
    gaps: List[float] = []

    mean_top1_sum: Dict[str, float] = {}
    mean_top1_count: Dict[str, int] = {}
    support_total: Dict[str, int] = {}

    for yt, yp in runs:
        simple = top1_per_service(yt, yp, node_names)
        detailed = top1_per_service_detailed(yt, yp, node_names)
        global_avg1 = float(rca_metrics.compute(yt, yp)["avg1"])
        weighted = float(detailed["weighted_top1"])
        gap = abs(global_avg1 - weighted)

        out_simple.append(simple)
        out_details.append(
            {
                "per_service_top1": detailed["per_service_top1"],
                "per_service_counts": detailed["per_service_counts"],
                "per_service_hits": detailed["per_service_hits"],
                "weighted_top1": weighted,
                "global_avg1": global_avg1,
                "abs_gap_vs_global_avg1": gap,
            }
        )
        gaps.append(gap)

        for svc, val in detailed["per_service_top1"].items():
            mean_top1_sum[svc] = mean_top1_sum.get(svc, 0.0) + float(val)
            mean_top1_count[svc] = mean_top1_count.get(svc, 0) + 1
        for svc, c in detailed["per_service_counts"].items():
            support_total[svc] = support_total.get(svc, 0) + int(c)

    max_gap = float(max(gaps)) if gaps else 0.0
    if max_gap > tol:
        raise RuntimeError(
            f"Per-root-cause consistency check failed for {prefix} on {dataset_name}: "
            f"max_abs_gap={max_gap:.8f} > {tol}."
        )

    support_sum = float(sum(support_total.values()))
    mean_top1_by_service = {
        svc: float(mean_top1_sum[svc] / max(mean_top1_count.get(svc, 1), 1))
        for svc in sorted(mean_top1_sum.keys())
    }
    support_share_by_service = {
        svc: float(support_total[svc] / support_sum) if support_sum > 0 else 0.0
        for svc in sorted(support_total.keys())
    }

    ds_results[f"{prefix}_per_root_cause"] = out_simple
    ds_results[f"{prefix}_per_root_cause_details"] = out_details
    ds_results[f"{prefix}_per_root_cause_consistency"] = {
        "tolerance": tol,
        "max_abs_gap": max_gap,
        "checked_runs": int(len(runs)),
    }
    ds_results[f"{prefix}_per_root_cause_summary"] = {
        "source_model_key": model_key,
        "mean_top1_by_service": mean_top1_by_service,
        "total_support_by_service": {k: int(v) for k, v in sorted(support_total.items())},
        "support_share_by_service": support_share_by_service,
        "num_seeds_aggregated": int(len(runs)),
    }


def train_and_eval_sta(
    train_data, test_data, node_names, edge_index, device,
    hidden_dim=32, epochs=100, seed=42, use_mm=False,
    use_logs: bool = True, use_traces: bool = True,
    contrastive_warmup: int = 10,
):
    set_all_seeds(seed)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)
    k = get_optimal_k(N)
    F_ = train_data[0].x.size(1)

    if use_mm:
        model = SelectThenAnalyze_MM(
            F_metric=F_, hidden_dim=hidden_dim, num_nodes=N, k=k,
        ).to(device)
    else:
        model = SelectThenAnalyze(
            num_features=F_, hidden_dim=hidden_dim, num_nodes=N, k=k,
        ).to(device)

    anomalous = [d for d in train_data if d.root_cause[0] in name_to_idx]
    if len(anomalous) < 2:
        return None

    x_sup = torch.stack([d.x for d in anomalous]).to(device)
    x_logs_sup = _stack_optional_features(anomalous, "x_logs")
    x_traces_sup = _stack_optional_features(anomalous, "x_traces")
    if x_logs_sup is not None:
        x_logs_sup = x_logs_sup.to(device)
    if x_traces_sup is not None:
        x_traces_sup = x_traces_sup.to(device)
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous], dtype=torch.long, device=device)
    ei = edge_index.to(device)

    # ---- Contrastive warmup (InfoNCE on Z_subgraph + variational GIB) ----
    if contrastive_warmup > 0 and x_sup.size(0) >= 4:
        opt_warm = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        target_rate = k / N
        model.set_temperature(2.0)
        for ep in range(contrastive_warmup):
            model.train()
            perm = torch.randperm(x_sup.size(0), device=device)
            half = x_sup.size(0) // 2
            idx_a, idx_n = perm[:half], perm[half:]

            opt_warm.zero_grad()
            if use_mm:
                xla = (x_logs_sup[idx_a] if x_logs_sup is not None else None) if use_logs else None
                xta = (x_traces_sup[idx_a] if x_traces_sup is not None else None) if use_traces else None
                _, M_a, _, sel_a = model(x_sup[idx_a], xla, xta, ei)
                xln = (x_logs_sup[idx_n] if x_logs_sup is not None else None) if use_logs else None
                xtn = (x_traces_sup[idx_n] if x_traces_sup is not None else None) if use_traces else None
                _, M_n, _, sel_n = model(x_sup[idx_n], xln, xtn, ei)
            else:
                _, M_a, _, sel_a = model(x_sup[idx_a], ei)
                _, M_n, _, sel_n = model(x_sup[idx_n], ei)

            # Mask-weighted embeddings (aligned with train_sta.py / train_sta_compact)
            B_a, B_n_sz = idx_a.size(0), idx_n.size(0)
            if hasattr(model, 'light_encoder'):
                h_a_full = model.light_encoder(x_sup[idx_a].view(B_a * N, F_, -1)).view(B_a, N, -1)
                h_n_full = model.light_encoder(x_sup[idx_n].view(B_n_sz * N, F_, -1)).view(B_n_sz, N, -1)
            elif hasattr(model, 'metric_encoder'):
                h_a_full = model.metric_encoder(x_sup[idx_a].view(B_a * N, F_, -1)).view(B_a, N, -1)
                h_n_full = model.metric_encoder(x_sup[idx_n].view(B_n_sz * N, F_, -1)).view(B_n_sz, N, -1)
            else:
                h_a_full = sel_a.unsqueeze(-1).expand(B_a, N, 1)
                h_n_full = sel_n.unsqueeze(-1).expand(B_n_sz, N, 1)
            z_a = (h_a_full * M_a.unsqueeze(-1)).sum(1) / (M_a.sum(1, keepdim=True) + 1e-8)
            z_n = (h_n_full * M_n.unsqueeze(-1)).sum(1) / (M_n.sum(1, keepdim=True) + 1e-8)
            z_pos = z_a + torch.randn_like(z_a) * 0.05
            l_nce = info_nce_loss(z_a, z_pos, z_n, temperature=0.07)
            l_gib = variational_gib_loss(torch.cat([M_a, M_n], dim=0), target_rate)
            beta = min(1.0, ep / max(contrastive_warmup // 2, 1)) * 0.3

            loss = l_nce + beta * l_gib
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_warm.step()
        model.set_temperature(0.5)

    # ---- Supervised training ----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(x_sup.size(0), device=device)
        for s in range(0, x_sup.size(0), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            if use_mm:
                xb_l = (x_logs_sup[idx] if x_logs_sup is not None else None) if use_logs else None
                xb_t = (x_traces_sup[idx] if x_traces_sup is not None else None) if use_traces else None
                logits_full, M_soft, top_k, sel_logits = model(x_sup[idx], xb_l, xb_t, ei)
            else:
                logits_full, M_soft, top_k, sel_logits = model(x_sup[idx], ei)

            yb = y_sup[idx]
            # Primary selector supervision.
            loss_sel = F.cross_entropy(sel_logits, yb)

            # Secondary deep scorer supervision (without extra regularizers).
            # For k<N, only samples with RC in top-k contribute.
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
    test_anom = [d for d in test_data if d.root_cause[0] in name_to_idx]
    y_true, y_logits = [], []
    with torch.no_grad():
        for d in test_anom:
            x = d.x.unsqueeze(0).to(device)
            if use_mm:
                xl = d.x_logs.unsqueeze(0).to(device) if (use_logs and hasattr(d, "x_logs")) else None
                xt = d.x_traces.unsqueeze(0).to(device) if (use_traces and hasattr(d, "x_traces")) else None
                logits_full, _, top_k, sel = model(x, xl, xt, ei)
            else:
                logits_full, _, top_k, sel = model(x, ei)
            merged = _merge_sta_logits(sel, logits_full, top_k)
            y_true.append(name_to_idx[d.root_cause[0]])
            y_logits.append(merged.squeeze(0).cpu().numpy())
    return y_true, y_logits


def train_and_eval_gat(
    train_data, test_data, node_names, edge_index, device,
    hidden_dim=32, epochs=100, seed=42, use_mm=False,
    use_logs: bool = True, use_traces: bool = True,
):
    set_all_seeds(seed)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)
    F_ = train_data[0].x.size(1)

    if use_mm:
        model = SpatioTemporalRCA_GAT_MM(F_metric=F_, hidden_dim=hidden_dim, num_nodes=N).to(device)
    else:
        model = SpatioTemporalRCA_GAT(num_features=F_, hidden_dim=hidden_dim, num_nodes=N).to(device)

    anomalous = [d for d in train_data if d.root_cause[0] in name_to_idx]
    if len(anomalous) < 2:
        return None

    x_sup = torch.stack([d.x for d in anomalous]).to(device)
    x_logs_sup = _stack_optional_features(anomalous, "x_logs")
    x_traces_sup = _stack_optional_features(anomalous, "x_traces")
    if x_logs_sup is not None:
        x_logs_sup = x_logs_sup.to(device)
    if x_traces_sup is not None:
        x_traces_sup = x_traces_sup.to(device)
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous], dtype=torch.long, device=device)
    ei = edge_index.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(x_sup.size(0), device=device)
        for s in range(0, x_sup.size(0), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            if use_mm:
                xb_l = (x_logs_sup[idx] if x_logs_sup is not None else None) if use_logs else None
                xb_t = (x_traces_sup[idx] if x_traces_sup is not None else None) if use_traces else None
                logits, _, _ = model(x_sup[idx], xb_l, xb_t, ei)
            else:
                logits, _, _ = model(x_sup[idx], ei)
            loss = F.cross_entropy(logits, y_sup[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    test_anom = [d for d in test_data if d.root_cause[0] in name_to_idx]
    y_true, y_logits = [], []
    with torch.no_grad():
        for d in test_anom:
            x = d.x.unsqueeze(0).to(device)
            if use_mm:
                xl = d.x_logs.unsqueeze(0).to(device) if (use_logs and hasattr(d, "x_logs")) else None
                xt = d.x_traces.unsqueeze(0).to(device) if (use_traces and hasattr(d, "x_traces")) else None
                logits, _, _ = model(x, xl, xt, ei)
            else:
                logits, _, _ = model(x, ei)
            y_true.append(name_to_idx[d.root_cause[0]])
            y_logits.append(logits.squeeze(0).cpu().numpy())
    return y_true, y_logits


def train_and_eval_late_gib(
    train_data, test_data, node_names, edge_index, device,
    hidden_dim=32, epochs=100, seed=42,
):
    """Late-Bottleneck STGCN+GIB (Phase 2 supervisée seulement).

    train_v1_compact nécessite des samples healthy pour Phase 1. Sur observ-it
    (load_observit retourne uniquement les windows labellisées), on implémente
    directement la Phase 2 : CE supervisée sur les logits GIB bruts.
    Même interface (y_true, y_logits) que train_and_eval_sta/gat.
    """
    from rca_pipeline.model import SpatioTemporalRCA
    from rca_pipeline.training_utils import EarlyStopping

    set_all_seeds(seed)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)

    anomalous = [d for d in train_data if d.root_cause[0] in name_to_idx]
    if len(anomalous) < 2:
        return None

    x_sup = torch.stack([d.x for d in anomalous]).to(device)  # [B, N, F, W]
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous],
                         dtype=torch.long, device=device)
    ei = edge_index.to(device)
    F_ = x_sup.size(2)  # num_features (not N)

    model = SpatioTemporalRCA(
        num_features=F_, hidden_dim=hidden_dim,
        num_nodes=N, gib_temperature=1.0,
    ).to(device)

    n_sup = x_sup.size(0)
    perm_split = torch.randperm(n_sup, device=device)
    n_val = max(1, int(n_sup * 0.1))
    val_idx = perm_split[:n_val]
    train_idx = perm_split[n_val:]
    x_val, y_val = x_sup[val_idx], y_sup[val_idx]
    x_tr, y_tr = x_sup[train_idx], y_sup[train_idx]
    n_tr = x_tr.size(0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-5)
    es = EarlyStopping(patience=30)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, 64):
            idx = perm[s:s + 64]
            B = idx.size(0)
            opt.zero_grad()
            x_flat = x_tr[idx].view(B * N, F_, -1)
            h_t = model.temporal_block(x_flat)
            if B > 1:
                offsets = torch.arange(B, device=device) * N
                batch_ei = ei.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
                batch_ei = batch_ei.reshape(2, -1)
            else:
                batch_ei = ei
            h_s = model.spatial_block(h_t, batch_ei)
            logits = model.gib.mlp(h_s).squeeze(-1).view(B, N)
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        # Early stopping on val MRR
        model.eval()
        with torch.no_grad():
            B_v = x_val.size(0)
            x_flat_v = x_val.view(B_v * N, F_, -1)
            h_tv = model.temporal_block(x_flat_v)
            if B_v > 1:
                offsets_v = torch.arange(B_v, device=device) * N
                bei_v = ei.unsqueeze(0).expand(B_v, -1, -1) + offsets_v.view(-1, 1, 1)
                bei_v = bei_v.reshape(2, -1)
            else:
                bei_v = ei
            h_sv = model.spatial_block(h_tv, bei_v)
            logits_v = model.gib.mlp(h_sv).squeeze(-1).view(B_v, N)
        mrr_v = 0.0
        for i in range(B_v):
            ranked = logits_v[i].cpu().numpy().argsort()[::-1]
            rank = int(np.where(ranked == y_val[i].item())[0][0]) + 1
            mrr_v += 1.0 / rank
        if es.step(mrr_v / B_v, model):
            break

    es.restore(model)
    model.eval()

    test_anom = [d for d in test_data if d.root_cause[0] in name_to_idx]
    if not test_anom:
        return None

    y_true, y_logits = [], []
    with torch.no_grad():
        for d in test_anom:
            x = d.x.unsqueeze(0).to(device)
            x_flat = x.view(N, F_, -1)
            h_t = model.temporal_block(x_flat)
            h_s = model.spatial_block(h_t, ei)
            gib_logits = model.gib.mlp(h_s).squeeze(-1)
            y_true.append(name_to_idx[d.root_cause[0]])
            y_logits.append(gib_logits.cpu().numpy())
    return y_true, y_logits


def _run_single_dataset(
    ds_name: str,
    data_list: List,
    node_names: List[str],
    edge_index: torch.Tensor,
    args,
    seeds: List[int],
    device: torch.device,
    rca_metrics: RCAEvalMetrics,
    normalize_train_test: bool = False,
) -> Dict:
    """
    Évalue STA et GATv2 sur un jeu déjà chargé (RCAEval ou observ-it).

    Si normalize_train_test=True (export cluster), clone chaque seed puis
    StandardScaler fit sur train uniquement (comme graph_builder / Week2).
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    N = len(node_names)
    print(f"  Cases: {len(data_list)}, Nodes: {N}")

    if len(data_list) < 4:
        print(f"  Too few cases ({len(data_list)}), skipping.")
        return {}

    sta_runs, gat_runs, late_gib_runs = [], [], []
    sta_raw_runs, gat_raw_runs = [], []
    sta_f_runs, gat_f_runs = [], []
    drift_train_stats = []
    drift_test_stats = []
    mm_skipped_seeds: List[int] = []
    mm_executed_seeds = 0
    scaler_stats: Optional[Dict[str, object]] = None
    split_trace: Dict[str, object] = {
        "type": "stratified_split_shuffle",
        "mode": "varying_split_by_run_seed",
        "seed_source": "run_seed",
        "train_ratio": 0.8,
        "requested_seeds": [int(s) for s in seeds],
        "executed_seeds": [],
        "mm_skipped_seeds": [],
    }

    for seed in seeds:
        if normalize_train_test:
            base = [_clone_pyg_data(d) for d in data_list]
        else:
            base = data_list
        # Unified stratified split with shuffle (order-independent).
        train_data, test_data = stratified_split_shuffle(
            base,
            train_ratio=0.8,
            name_to_idx=name_to_idx,
            seed=seed,
            keep_at_least_one_in_test=True,
        )
        _warn_split_quality(train_data, test_data, name_to_idx, seed)

        if normalize_train_test:
            scaler = GraphBuilder.fit_scaler(train_data)
            train_data = GraphBuilder.apply_scaler(train_data, scaler)
            test_data = GraphBuilder.apply_scaler(test_data, scaler)
            scaler_stats = {
                "type": "StandardScaler",
                "mean": [float(x) for x in scaler.mean_.tolist()],
                "scale": [float(x) for x in scaler.scale_.tolist()],
            }
        elif bool(getattr(args, "normalize", False)):
            # Optional, consistent train-only normalization for RCAEval as well.
            base_cloned = [_clone_pyg_data(d) for d in base]
            train_data, test_data = stratified_split_shuffle(
                base_cloned,
                train_ratio=0.8,
                name_to_idx=name_to_idx,
                seed=seed,
                keep_at_least_one_in_test=True,
            )
            scaler = GraphBuilder.fit_scaler(train_data)
            train_data = GraphBuilder.apply_scaler(train_data, scaler)
            test_data = GraphBuilder.apply_scaler(test_data, scaler)
            scaler_stats = {
                "type": "StandardScaler",
                "mean": [float(x) for x in scaler.mean_.tolist()],
                "scale": [float(x) for x in scaler.scale_.tolist()],
            }

        if args.use_mm and args.mm_strict:
            eps = 1e-8
            def _has_signal_on_all(dl: list, attr: str) -> bool:
                return bool(dl) and all(
                    hasattr(d, attr)
                    and getattr(d, attr) is not None
                    and float(getattr(d, attr).std().item()) > eps
                    for d in dl
                )

            need_logs = not bool(args.mm_traces_only)
            need_traces = not bool(args.mm_logs_only)
            ok = True
            if need_logs:
                ok = ok and _has_signal_on_all(train_data, "x_logs") and _has_signal_on_all(test_data, "x_logs")
            if need_traces:
                ok = ok and _has_signal_on_all(train_data, "x_traces") and _has_signal_on_all(test_data, "x_traces")
            if not ok:
                print("  Seed %s: MM strict enabled but required modality signal missing; skipping this seed." % seed)
                mm_skipped_seeds.append(int(seed))
                split_trace["mm_skipped_seeds"].append(int(seed))
                continue

        mm_executed_seeds += 1
        split_trace["executed_seeds"].append(int(seed))

        use_logs = bool(args.use_mm and not args.mm_traces_only)
        use_traces = bool(args.use_mm and not args.mm_logs_only)

        if args.use_mm and args.mm_trace_prior:
            eps = 1e-8
            def _has_any_trace_signal(dl: list) -> bool:
                return any(
                    hasattr(d, "x_traces")
                    and d.x_traces is not None
                    and float(d.x_traces.std().item()) > eps
                    for d in dl
                )
            if _has_any_trace_signal(train_data) or _has_any_trace_signal(test_data):
                use_logs = False

        r_sta_raw = train_and_eval_sta(
            train_data, test_data, node_names, edge_index, device,
            hidden_dim=args.hidden_dim, epochs=args.epochs, seed=seed,
            use_mm=args.use_mm, use_logs=use_logs, use_traces=use_traces,
            contrastive_warmup=args.sta_contrastive_warmup,
        )
        if r_sta_raw:
            sta_raw_runs.append(r_sta_raw)
        r_gat_raw = train_and_eval_gat(
            train_data, test_data, node_names, edge_index, device,
            hidden_dim=args.hidden_dim, epochs=args.epochs, seed=seed,
            use_mm=args.use_mm, use_logs=use_logs, use_traces=use_traces,
        )
        if r_gat_raw:
            gat_raw_runs.append(r_gat_raw)

        if getattr(args, "include_late_gib", False):
            r_lgib = train_and_eval_late_gib(
                train_data, test_data, node_names, edge_index, device,
                hidden_dim=args.hidden_dim, epochs=args.epochs, seed=seed,
            )
            if r_lgib:
                late_gib_runs.append(r_lgib)

        if args.enable_drift:
            train_f, drift_train = _maybe_apply_drift_filter(
                train_data, True, args.drift_delta, args.drift_cooldown_steps
            )
            test_f, drift_test = _maybe_apply_drift_filter(
                test_data, True, args.drift_delta, args.drift_cooldown_steps
            )
            drift_train_stats.append(drift_train)
            drift_test_stats.append(drift_test)

            if train_f and test_f:
                r_sta_f = train_and_eval_sta(
                    train_f, test_f, node_names, edge_index, device,
                    hidden_dim=args.hidden_dim, epochs=args.epochs, seed=seed,
                    use_mm=args.use_mm, use_logs=use_logs, use_traces=use_traces,
                    contrastive_warmup=args.sta_contrastive_warmup,
                )
                if r_sta_f:
                    sta_f_runs.append(r_sta_f)

                r_gat_f = train_and_eval_gat(
                    train_f, test_f, node_names, edge_index, device,
                    hidden_dim=args.hidden_dim, epochs=args.epochs, seed=seed,
                    use_mm=args.use_mm, use_logs=use_logs, use_traces=use_traces,
                )
                if r_gat_f:
                    gat_f_runs.append(r_gat_f)

    sta_runs = sta_f_runs if (args.enable_drift and sta_f_runs) else sta_raw_runs
    gat_runs = gat_f_runs if (args.enable_drift and gat_f_runs) else gat_raw_runs

    ds_results: Dict = {}

    if sta_runs:
        sta_m = rca_metrics.compute_with_std(sta_runs)
        sta_name = "STA-MM" if args.use_mm else "STA"
        rca_metrics.print_summary(sta_m, model_name=f"{sta_name} ({ds_name})",
                                  reference=RCAEVAL_PUBLISHED_BASELINES.get("BARO"))
        compare_with_rcaeval_baselines(sta_m, dataset_name=ds_name, model_name=sta_name)
        ds_results[sta_name] = sta_m

    if gat_runs:
        gat_m = rca_metrics.compute_with_std(gat_runs)
        gat_name = "GATv2-MM" if args.use_mm else "GATv2"
        rca_metrics.print_summary(gat_m, model_name=f"{gat_name} ({ds_name})",
                                  reference=RCAEVAL_PUBLISHED_BASELINES.get("BARO"))
        compare_with_rcaeval_baselines(gat_m, dataset_name=ds_name, model_name=gat_name)
        ds_results[gat_name] = gat_m

    if late_gib_runs:
        lgib_m = rca_metrics.compute_with_std(late_gib_runs)
        rca_metrics.print_summary(lgib_m, model_name=f"Late-STGCN+GIB ({ds_name})",
                                  reference=RCAEVAL_PUBLISHED_BASELINES.get("BARO"))
        compare_with_rcaeval_baselines(lgib_m, dataset_name=ds_name, model_name="Late-STGCN+GIB")
        ds_results["Late-STGCN+GIB"] = lgib_m
        per_seed_lgib = [rca_metrics.compute(yt, yp) for (yt, yp) in late_gib_runs]
        ds_results["Late-STGCN+GIB"]["ci95"] = {
            k: bootstrap_ci([float(r[k]) for r in per_seed_lgib], n_boot=2000, alpha=0.05, seed=0)
            for k in RCAEvalMetrics.METRIC_KEYS
        }
        _attach_per_root_metrics(
            ds_results, "Late-STGCN+GIB", "Late-STGCN+GIB",
            late_gib_runs, node_names, rca_metrics, ds_name,
        )

    # Bootstrap CI across seed-level aggregate metrics (CI on mean across seeds).
    # With few seeds (e.g. 3), intervals are wide / unstable; see seed_level_bootstrap_reliable in JSON.
    def _add_ci(model_key: str) -> None:
        if model_key not in ds_results:
            return
        per_seed = [rca_metrics.compute(yt, yp) for (yt, yp) in (sta_runs if "STA" in model_key else gat_runs)]
        ds_results[model_key]["ci95"] = {
            k: bootstrap_ci([float(r[k]) for r in per_seed], n_boot=2000, alpha=0.05, seed=0)
            for k in RCAEvalMetrics.METRIC_KEYS
        }

    _add_ci("STA-MM" if args.use_mm else "STA")
    _add_ci("GATv2-MM" if args.use_mm else "GATv2")

    if getattr(args, "ensemble", False) and sta_runs and gat_runs:
        from rca_pipeline.ensemble import (
            disagreement_rate,
            ensemble_runs,
            expected_calibration_error,
            oracle_top1,
        )
        ens_runs = ensemble_runs(sta_runs, gat_runs, mode="soft")
        if ens_runs:
            ens_m = rca_metrics.compute_with_std(ens_runs)
            ens_name = "Ensemble-MM" if args.use_mm else "Ensemble"
            rca_metrics.print_summary(ens_m, model_name=f"{ens_name} STA+GATv2 ({ds_name})",
                                      reference=RCAEVAL_PUBLISHED_BASELINES.get("BARO"))
            ds_results[ens_name] = ens_m
            per_seed_ens = [rca_metrics.compute(yt, yp) for (yt, yp) in ens_runs]
            ds_results[ens_name]["ci95"] = {
                k: bootstrap_ci([float(r[k]) for r in per_seed_ens], n_boot=2000, alpha=0.05, seed=0)
                for k in RCAEvalMetrics.METRIC_KEYS
            }
            # Descriptive test-set calibration (no fitting → no leakage).
            ds_results[ens_name]["ece"] = {
                "sta": float(np.mean([
                    expected_calibration_error(yp, yt) for (yt, yp) in sta_runs
                ])),
                "gatv2": float(np.mean([
                    expected_calibration_error(yp, yt) for (yt, yp) in gat_runs
                ])),
                "ensemble": float(np.mean([
                    expected_calibration_error(yp, yt) for (yt, yp) in ens_runs
                ])),
            }
            _attach_per_root_metrics(
                ds_results, "Ensemble", ens_name, ens_runs, node_names, rca_metrics, ds_name,
            )
            # Routing headroom diagnostics: how often the models disagree,
            # what a perfect per-case router would score (oracle), and how
            # sensitive the soft vote is to the mixing weight.
            weight_sweep = {}
            for w in (0.25, 0.5, 0.75):
                runs_w = ensemble_runs(sta_runs, gat_runs, mode="soft", weight_a=w)
                weight_sweep[f"w_sta={w}"] = float(np.mean([
                    rca_metrics.compute(yt, yp)["avg1"] for (yt, yp) in runs_w
                ]))
            conf_runs = ensemble_runs(sta_runs, gat_runs, mode="confidence")
            ds_results[ens_name]["ensemble_analysis"] = {
                "disagreement_rate_top1": float(disagreement_rate(sta_runs, gat_runs)),
                "oracle_top1": float(oracle_top1(sta_runs, gat_runs)),
                "soft_vote_avg1_by_weight": weight_sweep,
                "confidence_vote_avg1": float(np.mean([
                    rca_metrics.compute(yt, yp)["avg1"] for (yt, yp) in conf_runs
                ])) if conf_runs else None,
            }
            ana = ds_results[ens_name]["ensemble_analysis"]
            print(f"  [ensemble] top-1 disagreement={ana['disagreement_rate_top1']:.3f}  "
                  f"oracle={ana['oracle_top1']:.3f}  "
                  f"conf-vote avg1={ana['confidence_vote_avg1']:.3f}  "
                  f"weight sweep={ {k: round(v, 3) for k, v in weight_sweep.items()} }")

    if args.enable_drift:
        if sta_raw_runs:
            ds_results["STA_raw"] = rca_metrics.compute_with_std(sta_raw_runs)
        if sta_f_runs:
            ds_results["STA_drift_filtered"] = rca_metrics.compute_with_std(sta_f_runs)
        if gat_raw_runs:
            ds_results["GATv2_raw"] = rca_metrics.compute_with_std(gat_raw_runs)
        if gat_f_runs:
            ds_results["GATv2_drift_filtered"] = rca_metrics.compute_with_std(gat_f_runs)

    mm_cov = _mm_coverage(data_list)
    mm_effective = False
    if args.use_mm and data_list:
        d0 = data_list[0]
        logs_ok = hasattr(d0, "x_logs") and float(d0.x_logs.std().item()) > 1e-8
        traces_ok = hasattr(d0, "x_traces") and float(d0.x_traces.std().item()) > 1e-8
        if args.mm_logs_only:
            mm_effective = logs_ok
        elif args.mm_traces_only:
            mm_effective = traces_ok
        else:
            mm_effective = logs_ok or traces_ok
    mm_status = _compute_mm_dataset_status(
        args=args,
        requested_seeds=seeds,
        skipped_seeds=mm_skipped_seeds,
        executed_seeds=mm_executed_seeds,
    )
    ds_results["config"] = {
        "dataset": ds_name,
        "seeds": [int(s) for s in seeds],
        "n_cases_loaded": int(len(data_list)),
        "n_nodes": int(N),
        "max_cases": None if args.max_cases is None else int(args.max_cases),
        "use_mm": bool(args.use_mm),
        "mm_strict": bool(args.mm_strict),
        "mm_logs_only": bool(args.mm_logs_only),
        "mm_traces_only": bool(args.mm_traces_only),
        "mm_trace_prior": bool(args.mm_trace_prior),
        "mm_effective": mm_effective,
        "mm_coverage": mm_cov,
        "mm_dataset_status": mm_status["status"],
        "mm_dataset_reason": mm_status["reason"],
        "mm_requested_seeds": mm_status["requested_seeds"],
        "mm_executed_seeds": mm_status["executed_seeds"],
        "mm_skipped_seeds": mm_status["skipped_seeds"],
        "enable_drift": bool(args.enable_drift),
        "drift_delta": float(args.drift_delta),
        "drift_cooldown_steps": int(args.drift_cooldown_steps),
        "observit_normalized": bool(normalize_train_test),
        "normalized": bool(normalize_train_test or bool(getattr(args, "normalize", False))),
        "split_mode": str(split_trace.get("mode", "unknown")),
        "hyperparams": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "sta_contrastive_warmup": args.sta_contrastive_warmup,
            "allow_real_warmup": bool(getattr(args, "allow_real_warmup", False)),
        },
        "split": split_trace,
    }
    if scaler_stats is not None:
        ds_results["scaler"] = scaler_stats
    if args.enable_drift and drift_train_stats and drift_test_stats:
        ds_results["drift_filter"] = {
            "train_drop_ratio_mean": float(np.mean([s["drop_ratio"] for s in drift_train_stats])),
            "test_drop_ratio_mean": float(np.mean([s["drop_ratio"] for s in drift_test_stats])),
        }

    print("\n  --- Published baselines (RCAEval paper) ---")
    for method, ref in RCAEVAL_PUBLISHED_BASELINES.items():
        print(f"  {method:>15s}: Avg@1={ref['avg1']:.2f}  Avg@5={ref['avg5']:.2f}  MRR={ref['mrr']:.2f}")

    # Per-root-cause Top-1 for all real datasets (RCAEval + ObservIT).
    sta_model_key = "STA-MM" if args.use_mm else "STA"
    gat_model_key = "GATv2-MM" if args.use_mm else "GATv2"
    _attach_per_root_metrics(ds_results, "STA", sta_model_key, sta_runs, node_names, rca_metrics, ds_name)
    _attach_per_root_metrics(ds_results, "GATv2", gat_model_key, gat_runs, node_names, rca_metrics, ds_name)

    return ds_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark on RCAEval real datasets")
    parser.add_argument("--dataset", type=str, default="trainticket",
                        help="Dataset name or 'all'")
    parser.add_argument("--seeds", type=str, default="42,123,7")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sta-contrastive-warmup", type=int, default=0,
                        help="Contrastive warmup epochs for STA (default 0 for RCAEval real)")
    parser.add_argument(
        "--allow-real-warmup",
        action="store_true",
        help="Acknowledge real-data warmup risk and allow sta-contrastive-warmup > 0.",
    )
    parser.add_argument("--rcaeval-root", type=str, default="rcaeval")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of failure cases loaded (speeds up MM extraction)")
    parser.add_argument("--use-mm", action="store_true",
                        help="Use multimodal variants (expects x_logs/x_traces when available)")
    parser.add_argument("--mm-strict", action="store_true",
                        help="When --use-mm is set, require logs+traces on all used samples")
    parser.add_argument("--mm-logs-only", action="store_true",
                        help="When --use-mm is set, use metrics+logs only (ignore traces)")
    parser.add_argument("--mm-traces-only", action="store_true",
                        help="When --use-mm is set, use metrics+traces only (ignore logs)")
    parser.add_argument("--mm-trace-prior", action="store_true",
                        help="When --use-mm is set, ignore logs if traces are present with signal (safe default)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Also report a post-hoc STA+GATv2 soft-vote ensemble (no retraining)")
    parser.add_argument("--enable-drift", action="store_true",
                        help="Enable ADWIN drift cooldown filter before train/eval")
    parser.add_argument("--drift-delta", type=float, default=0.002,
                        help="ADWIN delta for drift detector")
    parser.add_argument("--drift-cooldown-steps", type=int, default=30,
                        help="Cooldown steps to skip RCA after drift event")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply train-only z-score normalization (StandardScaler) for RCAEval datasets too.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Best-effort deterministic mode for PyTorch (may reduce performance).",
    )
    parser.add_argument(
        "--observit-csv",
        type=str,
        default=None,
        help="Fichier CSV ou répertoire dataset_*.csv (labo K8s observ-it). Ignore --dataset / RCAEval.",
    )
    parser.add_argument(
        "--observit-window",
        type=int,
        default=12,
        help="Longueur de fenêtre temporelle pour l'export observ-it (défaut: 12).",
    )
    from rca_pipeline.topologies import TOPOLOGIES as _AVAILABLE_TOPOLOGIES
    parser.add_argument(
        "--observit-topology",
        type=str,
        default="custom_app",
        choices=sorted(_AVAILABLE_TOPOLOGIES.keys()),
        help="Topologie de services pour GraphBuilder (défaut: custom_app).",
    )
    parser.add_argument(
        "--include-late-gib",
        action="store_true",
        help="Inclure Late-Bottleneck STGCN+GIB dans le benchmark (plus lent, deux phases).",
    )
    parser.add_argument(
        "--loki-url",
        type=str,
        default=None,
        help="URL Loki pour enrichir les fenêtres observit avec .x_logs (ex: http://localhost:3100).",
    )
    parser.add_argument(
        "--loki-namespace",
        type=str,
        default="sock-shop",
        help="Namespace Kubernetes à interroger dans Loki (défaut: sock-shop).",
    )
    from rca_pipeline.config import BenchmarkRealConfig, parse_args_with_config
    args = parse_args_with_config(parser, BenchmarkRealConfig)

    if args.mm_logs_only and args.mm_traces_only:
        raise SystemExit("--mm-logs-only and --mm-traces-only are mutually exclusive.")
    if args.mm_trace_prior and (args.mm_logs_only or args.mm_traces_only):
        raise SystemExit("--mm-trace-prior is incompatible with --mm-logs-only/--mm-traces-only.")
    _enforce_real_warmup_guard(args.sta_contrastive_warmup, bool(getattr(args, "allow_real_warmup", False)))

    seeds = [int(s) for s in args.seeds.split(",")]
    if bool(getattr(args, "deterministic", False)):
        set_torch_deterministic(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.observit_csv:
        from rca_pipeline.loaders.observit_loader import load_observit

        print("\n" + "=" * 70)
        print("  DATASET: observ-it (cluster K8s)")
        print("=" * 70)
        try:
            data_list, node_names, edge_index = load_observit(
                args.observit_csv,
                window_size=args.observit_window,
                topology_name=args.observit_topology,
                loki_url=getattr(args, "loki_url", None),
                loki_namespace=getattr(args, "loki_namespace", "sock-shop"),
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            print(f"Failed to load observ-it export: {e}")
            return

        rca_metrics = RCAEvalMetrics()
        ds_results = _run_single_dataset(
            "observit",
            data_list,
            node_names,
            edge_index,
            args,
            seeds,
            device,
            rca_metrics,
            normalize_train_test=True,
        )
        all_results = {}
        if ds_results:
            all_results["observit"] = ds_results
            all_results["observit"]["env"] = runtime_env_info()
            all_results["observit"]["deterministic"] = bool(getattr(args, "deterministic", False))
        if bool(args.use_mm) and all_results:
            all_results["_mm_global_summary"] = _build_mm_global_summary(all_results)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            print(f"\nResults saved to {args.output_json}")
        return

    # RCAEval Python API varies across versions; we default to the structured
    # layout loader to keep benchmarks reproducible and avoid CSV fallback.
    # When --use-mm is enabled, we also extract logs/traces features from
    # the downloaded RCAEval CSV layout.
    loader = RCAEvalLoader(
        rcaeval_root=args.rcaeval_root,
        use_api=False,
        extract_multimodal=bool(args.use_mm),
    )

    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    available = loader.available_datasets()
    datasets = [d for d in datasets if d in available]

    if not datasets:
        print(f"No datasets available in {args.rcaeval_root}/data/")
        print(f"Run: git clone https://github.com/NetManAIOps/RCAEval ./{args.rcaeval_root}/")
        return

    rca_metrics = RCAEvalMetrics()
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")

        data_list, node_names, edge_index = loader.load(ds_name, max_cases=args.max_cases)
        ds_results = _run_single_dataset(
            ds_name,
            data_list,
            node_names,
            edge_index,
            args,
            seeds,
            device,
            rca_metrics,
            normalize_train_test=False,
        )
        if ds_results:
            all_results[ds_name] = ds_results
            all_results[ds_name]["env"] = runtime_env_info()
            all_results[ds_name]["deterministic"] = bool(getattr(args, "deterministic", False))

    if bool(args.use_mm) and all_results:
        mm_summary = _build_mm_global_summary(all_results)
        all_results["_mm_global_summary"] = mm_summary
        if not bool(mm_summary["allow_global_aggregation"]):
            blocked = ", ".join(mm_summary["blocked_by_datasets"])
            print(f"\nMM global aggregation blocked: {blocked}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
