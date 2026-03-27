#!/usr/bin/env python3
"""
benchmark_real.py — Validate RCA models on RCAEval real-world datasets.

Bridges the gap between simulated BA benchmarks and production realism (§7.1).

Usage:
    python3 benchmark_real.py --dataset trainticket --seeds 42,123,7
    python3 benchmark_real.py --dataset all
    python3 benchmark_real.py --observit-csv path/to/dataset.csv   # ou répertoire dataset_*.csv
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from typing import List, Dict, Optional, Tuple

from model_sta import SelectThenAnalyze
from model_late_gat import SpatioTemporalRCA_GAT
from model_sta_mm import SelectThenAnalyze_MM
from model_late_gat_mm import SpatioTemporalRCA_GAT_MM
from common import set_all_seeds
from k_selection import get_optimal_k
from losses import info_nce_loss, variational_gib_loss
from metrics_rcaeval import (
    RCAEvalMetrics,
    RCAEVAL_PUBLISHED_BASELINES,
    compare_with_rcaeval_baselines,
)
from loaders.rcaeval_loader import RCAEvalLoader, SUPPORTED_DATASETS
from drift_detector import make_k8s_drift_detector
from graph_builder import GraphBuilder

import logging
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


def _split(data_list, name_to_idx, ratio=0.8, seed: int = 42):
    import random
    by_rc: Dict[str, list] = {}
    for d in data_list:
        rc = d.root_cause[0]
        if rc in name_to_idx:
            by_rc.setdefault(rc, []).append(d)
    rng = random.Random(seed)
    train, test = [], []
    for rc, items in sorted(by_rc.items()):
        rng.shuffle(items)
        if len(items) <= 1:
            s = 1
        else:
            # Keep at least one sample in both splits for evaluable classes.
            s = int(len(items) * ratio)
            s = max(1, min(s, len(items) - 1))
        train.extend(items[:s])
        test.extend(items[s:])
    return train, test


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

    sta_runs, gat_runs = [], []
    sta_raw_runs, gat_raw_runs = [], []
    sta_f_runs, gat_f_runs = [], []
    drift_train_stats = []
    drift_test_stats = []

    for seed in seeds:
        if normalize_train_test:
            base = [_clone_pyg_data(d) for d in data_list]
        else:
            base = data_list
        train_data, test_data = _split(base, name_to_idx, seed=seed)
        _warn_split_quality(train_data, test_data, name_to_idx, seed)

        if normalize_train_test:
            scaler = GraphBuilder.fit_scaler(train_data)
            train_data = GraphBuilder.apply_scaler(train_data, scaler)
            test_data = GraphBuilder.apply_scaler(test_data, scaler)

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
                continue

        use_logs = bool(args.use_mm and not args.mm_traces_only)
        use_traces = bool(args.use_mm and not args.mm_logs_only)

        if args.use_mm and args.mm_trace_prior:
            eps = 1e-8
            def _has_any_trace_signal(dl: list) -> bool:
                return any(
                    hasattr(d, "x_traces")
                    and getattr(d, "x_traces") is not None
                    and float(getattr(d, "x_traces").std().item()) > eps
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
    ds_results["config"] = {
        "use_mm": bool(args.use_mm),
        "mm_strict": bool(args.mm_strict),
        "mm_logs_only": bool(args.mm_logs_only),
        "mm_traces_only": bool(args.mm_traces_only),
        "mm_trace_prior": bool(args.mm_trace_prior),
        "mm_effective": mm_effective,
        "mm_coverage": mm_cov,
        "enable_drift": bool(args.enable_drift),
        "drift_delta": float(args.drift_delta),
        "drift_cooldown_steps": int(args.drift_cooldown_steps),
        "observit_normalized": bool(normalize_train_test),
    }
    if args.enable_drift and drift_train_stats and drift_test_stats:
        ds_results["drift_filter"] = {
            "train_drop_ratio_mean": float(np.mean([s["drop_ratio"] for s in drift_train_stats])),
            "test_drop_ratio_mean": float(np.mean([s["drop_ratio"] for s in drift_test_stats])),
        }

    print(f"\n  --- Published baselines (RCAEval paper) ---")
    for method, ref in RCAEVAL_PUBLISHED_BASELINES.items():
        print(f"  {method:>15s}: Avg@1={ref['avg1']:.2f}  Avg@5={ref['avg5']:.2f}  MRR={ref['mrr']:.2f}")

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
    parser.add_argument("--enable-drift", action="store_true",
                        help="Enable ADWIN drift cooldown filter before train/eval")
    parser.add_argument("--drift-delta", type=float, default=0.002,
                        help="ADWIN delta for drift detector")
    parser.add_argument("--drift-cooldown-steps", type=int, default=30,
                        help="Cooldown steps to skip RCA after drift event")
    parser.add_argument("--output-json", type=str, default=None)
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
    args = parser.parse_args()

    if args.mm_logs_only and args.mm_traces_only:
        raise SystemExit("--mm-logs-only and --mm-traces-only are mutually exclusive.")
    if args.mm_trace_prior and (args.mm_logs_only or args.mm_traces_only):
        raise SystemExit("--mm-trace-prior is incompatible with --mm-logs-only/--mm-traces-only.")

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.observit_csv:
        from loaders.observit_loader import load_observit

        print("\n" + "=" * 70)
        print("  DATASET: observ-it (cluster K8s)")
        print("=" * 70)
        try:
            data_list, node_names, edge_index = load_observit(
                args.observit_csv, window_size=args.observit_window
            )
        except Exception as e:
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

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
