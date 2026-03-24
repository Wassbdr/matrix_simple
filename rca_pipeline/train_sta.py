"""
Training loop for SelectThenAnalyze (STA).

Two-phase training aligned with the paper's formulation:

Phase 1 -- Contrastive pre-training (self-supervised):
    L = L_InfoNCE(z_anchor, z_pos, z_neg) + beta * ||M_soft||_1
    - InfoNCE separates healthy from anomalous dynamics.
    - L1 sparsity on M_soft acts as an L0 surrogate: pushes the mask
      toward zero globally.  InfoNCE counterbalances by requiring
      discriminative activations for anomalous windows.
    - Temperature annealing (warm -> cool) for stable exploration.
    - beta warm-up prevents premature sparsification.

Phase 2 -- Supervised fine-tuning:
    L = L_CE(selector_logits, y)            [all N nodes, primary]
      + L_CE(scorer_logits, y) [if RC in top-k]  [subgraph, secondary]
      + gamma * ||M_soft||_1                [continued sparsity]
    - Selector receives direct gradient from CE on its logits.
    - Scorer receives gradient from CE only when the root cause was
      correctly selected in top-k (adaptive per-sample).
    - Both paths also give gradient to the selector through the STE.

Usage:
    python train_sta.py --dataset scalable_dataset_N100.pt --num-nodes 100 --k 10
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Dict
import logging
import argparse
import os
import time

from model_sta import SelectThenAnalyze
from losses import (
    info_nce_loss,
    recurrence_classification_loss,
    memory_regularization_loss,
    temporal_causality_regularizer,
)
from augment import augment_positive
from common import stratified_split
from memory_module import MemoryAugmentedRCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_scalable_dataset(filepath: str):
    """Load a scalable dataset produced by simulated_data_scalable.py."""
    payload = torch.load(filepath, weights_only=False)
    if isinstance(payload, dict):
        return payload["dataset"], payload["node_names"]
    # Fallback: legacy format (plain list)
    return payload, None


# ---------------------------------------------------------------------------
# Sparsity loss (L1 on M_soft)
# ---------------------------------------------------------------------------

def sparsity_loss_l1(M_soft: torch.Tensor) -> torch.Tensor:
    """||M_soft||_1 averaged over batch.  M_soft: [B, N]."""
    return M_soft.abs().sum(dim=-1).mean()


def asymmetric_sparsity_loss(
    M_soft_healthy: torch.Tensor,
    M_soft_anomalous: torch.Tensor,
    min_active_nodes: int = 1,
    dispersion_weight: float = 0.25,
) -> torch.Tensor:
    """
    Asymmetric variant matching the dynamical system assumption:
        healthy  (U=0):     push M -> 0          (L2)
        anomalous:          push M -> sparse and optionally distributed
                            across multiple nodes when min_active_nodes > 1
    """
    loss_h = (M_soft_healthy ** 2).sum(dim=-1).mean()

    # Anomalous: L1 + anti-collapse entropy floor
    loss_l1 = M_soft_anomalous.abs().sum(dim=-1).mean()
    p = M_soft_anomalous / (M_soft_anomalous.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
    N = M_soft_anomalous.size(-1)
    max_ent = torch.log(torch.tensor(float(max(N, 2)), device=entropy.device))
    target_nodes = max(1, min(min_active_nodes, N))
    target_ent = torch.log(torch.tensor(float(target_nodes), device=entropy.device)) / max_ent
    ent_norm = entropy / max_ent
    loss_dispersion = F.relu(target_ent - ent_norm).mean()

    return loss_h + loss_l1 + dispersion_weight * loss_dispersion


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sta(
    dataset_path: str,
    node_names: List[str],
    # Architecture
    hidden_dim: int = 32,
    k: int = 10,
    # Phase 1
    contrastive_epochs: int = 100,
    lr: float = 1e-3,
    beta_max: float = 0.5,
    warmup_epochs: int = 30,
    nce_temperature: float = 0.5,  # FIX: Increased from 0.07 to 0.5 for stable contrastive learning
    min_active_nodes: int = 1,
    dispersion_weight: float = 0.25,
    # Phase 2
    supervised_epochs: int = 100,
    supervised_lr: float = 1e-3,
    gamma_sparsity: float = 0.1,
    use_severity_weights: bool = False,
    curriculum: bool = False,
    # Memory Module (SOTA)
    use_memory: bool = False,
    memory_slots: int = 128,
    memory_dim: int = 64,
    lambda_recurrence: float = 0.1,
    lambda_memory: float = 0.01,
    lambda_causality: float = 0.05,
    # General
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    out_checkpoint: str = "sta.pt",
    skip_contrastive: bool = False,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    num_nodes = len(node_names)

    dataset, _ = load_scalable_dataset(dataset_path)
    train_data, _ = stratified_split(dataset, name_to_idx=name_to_idx)
    logger.info("Train: %d / %d total", len(train_data), len(dataset))

    healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
    anomalous = [d for d in train_data
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and d.root_cause[0] in name_to_idx]
    logger.info("Healthy: %d, Anomalous: %d", len(healthy), len(anomalous))

    # Build tensors
    # NOTE: We use the SAME windows for anchors and positives (with augmentation)
    # to avoid data leakage from consecutive windows that may overlap healthy->anomaly
    # transitions. Positives are generated via augment_positive() during training.
    x_anchor = torch.stack([d.x for d in healthy]).to(device)
    # x_positive will be generated on-the-fly via augmentation (same window, different view)
    x_negative = torch.stack([d.x for d in anomalous]).to(device)
    edge_index = healthy[0].edge_index.to(device)
    num_features = x_anchor.size(2)

    model = SelectThenAnalyze(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        k=k,
        gib_temperature=2.0,
    ).to(device)

    # SOTA: Wrap with Memory Module for recurrence pattern recognition
    if use_memory:
        model = MemoryAugmentedRCA(
            base_model=model,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            use_memory_for_scoring=True,
        ).to(device)
        logger.info("Memory Module enabled: %d slots, dim=%d", memory_slots, memory_dim)

    # Extract recurrence metadata for memory training (SOTA)
    is_recurrent_train = torch.tensor(
        [getattr(d, 'is_recurrent', False) for d in anomalous],
        dtype=torch.bool, device=device,
    )
    has_recurrence_labels = is_recurrent_train.any()
    if has_recurrence_labels:
        recurrence_ratio = is_recurrent_train.float().mean().item()
        logger.info("Recurrence labels found: %.1f%% recurrent", recurrence_ratio * 100)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total_params)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ===================================================================
    # PHASE 1: Contrastive + L1 Sparsity
    # ===================================================================
    if contrastive_epochs > 0 and not skip_contrastive:
        logger.info("=== PHASE 1: Contrastive + Sparsity (%d epochs) ===", contrastive_epochs)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=contrastive_epochs, eta_min=1e-5
        )

        n_h = x_anchor.size(0)
        n_n = x_negative.size(0)
        temp_start, temp_end = 2.0, 0.5

        for epoch in range(contrastive_epochs):
            model.train()
            beta = min(1.0, epoch / max(warmup_epochs, 1)) * beta_max
            t_frac = epoch / max(contrastive_epochs - 1, 1)
            temp_value = temp_start + t_frac * (temp_end - temp_start)
            if use_memory:
                model.base_model.set_temperature(temp_value)
            else:
                model.set_temperature(temp_value)

            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                B_h, B_n = idx_h.size(0), idx_n.size(0)

                optimizer.zero_grad()

                out_a = model(x_anchor[idx_h], edge_index)
                M_soft_a = out_a["M_soft"] if isinstance(out_a, dict) else out_a[1]
                # Generate positive pairs via augmentation of the SAME anchor (no leakage)
                x_pos_aug = augment_positive(
                    x_anchor[idx_h], noise_sigma=0.1,
                    jitter_shift=2, feature_mask_ratio=0.15,
                )
                out_p = model(x_pos_aug, edge_index)
                M_soft_p = out_p["M_soft"] if isinstance(out_p, dict) else out_p[1]
                out_n = model(x_negative[idx_n], edge_index)
                M_soft_n = out_n["M_soft"] if isinstance(out_n, dict) else out_n[1]

                # Graph-level embeddings via mean-pooled light encoder
                x_a_flat = x_anchor[idx_h].view(B_h * num_nodes, num_features, -1)
                x_p_flat = x_pos_aug.view(B_h * num_nodes, num_features, -1)
                x_n_flat = x_negative[idx_n].view(B_n * num_nodes, num_features, -1)

                # Access methods through base model if using memory
                base_model = model.base_model if use_memory else model
                batch_ei_h = base_model._batch_edge_index(edge_index, B_h)
                batch_ei_n = base_model._batch_edge_index(edge_index, B_n)

                h_a = base_model.light_encoder(x_a_flat, batch_ei_h).view(B_h, num_nodes, -1).mean(1)
                h_p = base_model.light_encoder(x_p_flat, batch_ei_h).view(B_h, num_nodes, -1).mean(1)
                h_n = base_model.light_encoder(x_n_flat, batch_ei_n).view(B_n, num_nodes, -1).mean(1)

                l_nce = info_nce_loss(h_a, h_p, h_n, nce_temperature)

                # Asymmetric sparsity loss
                M_healthy = torch.cat([M_soft_a, M_soft_p], dim=0)
                l_sparse = asymmetric_sparsity_loss(
                    M_healthy,
                    M_soft_n,
                    min_active_nodes=min_active_nodes,
                    dispersion_weight=dispersion_weight,
                )

                loss = l_nce + beta * l_sparse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == contrastive_epochs - 1:
                with torch.no_grad():
                    out_ma = model(x_anchor[:64], edge_index)
                    ma = out_ma["M_soft"] if isinstance(out_ma, dict) else out_ma[1]
                    out_mn = model(x_negative[:64], edge_index)
                    mn = out_mn["M_soft"] if isinstance(out_mn, dict) else out_mn[1]
                logger.info(
                    "[Phase1] Ep %03d/%03d | Loss %.4f | beta %.2f | "
                    "M_h_mean %.3f | M_a_mean %.3f",
                    epoch, contrastive_epochs,
                    epoch_loss / max(n_batches, 1), beta,
                    ma.mean().item(), mn.mean().item(),
                )

    # ===================================================================
    # PHASE 2: Supervised fine-tuning
    # ===================================================================
    logger.info("=== PHASE 2: Supervised (%d epochs) ===", supervised_epochs)

    x_sup = x_negative
    n_sup = len(x_sup)  # FIX: Define n_sup here
    y_sup = torch.tensor(
        [name_to_idx[d.root_cause[0]] for d in anomalous],
        dtype=torch.long, device=device,
    )

    if use_severity_weights or curriculum:
        severity_weights = torch.tensor(
            [getattr(d, 'severity', 1.0) for d in anomalous],
            dtype=torch.float, device=device,
        )
        severity_weights = severity_weights / severity_weights.mean().clamp(min=1e-6)
    else:
        severity_weights = torch.ones(len(x_sup), dtype=torch.float, device=device)

    # Build parameter groups (handle memory-augmented model)
    if use_memory:
        # Access base model components through wrapper
        base = model.base_model
        param_groups = [
            {"params": base.selector.parameters(), "lr": supervised_lr},
            {"params": base.scorer.parameters(), "lr": supervised_lr},
            {"params": base.deep_temporal.parameters(), "lr": supervised_lr},
            {"params": base.deep_spatial.parameters(), "lr": supervised_lr},
            {"params": base.light_encoder.parameters(), "lr": supervised_lr * 0.1},
            {"params": model.incident_encoder.parameters(), "lr": supervised_lr},
            {"params": model.memory.parameters(), "lr": supervised_lr * 0.5},
            {"params": model.memory_scorer.parameters(), "lr": supervised_lr},
            {"params": model.recurrence_head.parameters(), "lr": supervised_lr},
        ]
    else:
        param_groups = [
            {"params": model.selector.parameters(), "lr": supervised_lr},
            {"params": model.scorer.parameters(), "lr": supervised_lr},
            {"params": model.deep_temporal.parameters(), "lr": supervised_lr},
            {"params": model.deep_spatial.parameters(), "lr": supervised_lr},
            {"params": model.light_encoder.parameters(), "lr": supervised_lr * 0.1},
        ]
    optimizer_sup = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler_sup = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_sup, T_max=supervised_epochs, eta_min=1e-5
    )

    # Helper to set model parameters (handles memory wrapper)
    def set_model_param(name, value):
        target = model.base_model if use_memory else model
        getattr(target, f'set_{name}')(value)

    set_model_param('temperature', 0.5)

    for epoch in range(supervised_epochs):
        model.train()

        # k-annealing (using parameters for standalone running)
        k_warm = min(num_nodes // 2, num_nodes - 1)
        k_target = k
        k_anneal_epochs_cur = int(supervised_epochs * 0.75) if num_nodes >= 50 else 60
        if epoch < k_anneal_epochs_cur:
            k_cur = k_warm - int((k_warm - k_target) * epoch / k_anneal_epochs_cur)
        else:
            k_cur = k_target
        set_model_param('k', k_cur)

        # Temperature annealing: 1.0 -> 0.3
        t_frac = epoch / max(supervised_epochs - 1, 1)
        set_model_param('temperature', 1.0 + t_frac * (0.3 - 1.0))

        # Soft Curriculum: dynamic exponent scaling of severity
        if curriculum:
            epoch_power = 3.0 - (2.0 * t_frac)
        else:
            epoch_power = 1.0

        # FIX-REC2: seed per-epoch shuffle for reproducibility
        _perm_gen = torch.Generator(device='cpu')
        _perm_gen.manual_seed(42 + epoch)
        perm = torch.randperm(n_sup, generator=_perm_gen).to(device)
        ep_loss = ep_correct = ep_selected = 0
        n_batches = 0

        for start in range(0, n_sup, batch_size):
            idx = perm[start:start + batch_size]
            x_b = x_sup[idx]
            y_b = y_sup[idx]
            
            optimizer_sup.zero_grad()

            # Apply exponent to severity and normalize per batch
            if use_severity_weights or curriculum:
                # Invert severity to focus on subtle/gray failures (SOTA approach)
                sev_b = 1.0 / (severity_weights[idx] + 0.1)
                sev_b = sev_b ** epoch_power
                sev_b = sev_b / sev_b.mean().clamp(min=1e-6)
            else:
                sev_b = torch.ones_like(y_b, dtype=torch.float)

            # Forward pass (handles both base model and memory-augmented model)
            if use_memory:
                # Memory-augmented forward
                is_novel_b = ~is_recurrent_train[idx] if has_recurrence_labels else None
                outputs = model(x_b, edge_index, is_novel=is_novel_b)
                logits_full = outputs["logits_full"]
                M_soft = outputs["M_soft"]
                top_k = outputs["top_k"]
                sel_logits = outputs["sel_logits"]
            else:
                logits_full, M_soft, top_k, sel_logits = model(x_b, edge_index)

            # Primary: CE on selector logits (all N, numerically stable)
            if use_severity_weights or curriculum:
                loss_sel_unreduced = F.cross_entropy(sel_logits, y_b, reduction='none')
                loss_sel = (loss_sel_unreduced * sev_b).mean()
            else:
                loss_sel = F.cross_entropy(sel_logits, y_b)

            # Secondary: CE on scorer (only when RC in top-k)
            in_topk = (top_k == y_b.unsqueeze(1)).any(dim=1)
            if in_topk.any():
                if use_severity_weights or curriculum:
                    loss_scorer_unreduced = F.cross_entropy(logits_full[in_topk], y_b[in_topk], reduction='none')
                    loss_scorer = (loss_scorer_unreduced * sev_b[in_topk]).mean()
                else:
                    loss_scorer = F.cross_entropy(logits_full[in_topk], y_b[in_topk])
            else:
                loss_scorer = torch.tensor(0.0, device=device)

            # Continued sparsity regularization
            l_sparse = sparsity_loss_l1(M_soft)

            # SOTA: Temporal causality regularizer
            l_causality = temporal_causality_regularizer(M_soft, x_b, edge_index, lag=2)

            # SOTA: Memory and recurrence losses
            l_recurrence = torch.tensor(0.0, device=device)
            l_memory = torch.tensor(0.0, device=device)

            if use_memory and has_recurrence_labels:
                # Recurrence classification loss
                rec_logits = outputs.get("recurrence_logits")
                if rec_logits is not None:
                    is_rec_b = is_recurrent_train[idx]
                    l_recurrence = recurrence_classification_loss(rec_logits, is_rec_b)

                # Memory regularization (encourage diverse usage)
                mem_attn = outputs.get("memory_attn")
                mem_gate = outputs.get("memory_gate")
                if mem_attn is not None and mem_gate is not None:
                    l_memory = memory_regularization_loss(mem_attn, mem_gate)

            loss = (
                5.0 * loss_sel
                + loss_scorer
                + gamma_sparsity * l_sparse
                + lambda_causality * l_causality
                + lambda_recurrence * l_recurrence
                + lambda_memory * l_memory
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_sup.step()

            preds = sel_logits.argmax(dim=1)
            ep_loss += loss.item()
            ep_correct += (preds == y_b).sum().item()
            ep_selected += in_topk.sum().item()
            n_batches += 1

        scheduler_sup.step()

        if epoch % 20 == 0 or epoch == supervised_epochs - 1:
            log_msg = (
                "[Phase2] Ep %03d/%03d | Loss %.4f | Acc %.4f | "
                "RC-in-top-%d %.4f | M_mean %.3f"
            )
            log_args = [
                epoch, supervised_epochs,
                ep_loss / max(n_batches, 1),
                ep_correct / n_sup if n_sup > 0 else 0,
                k_cur,
                ep_selected / n_sup if n_sup > 0 else 0,
                M_soft.mean().item(),
            ]
            if use_memory and has_recurrence_labels:
                log_msg += " | L_rec %.4f"
                log_args.append(l_recurrence.item())
            logger.info(log_msg, *log_args)

    # ===================================================================
    # Save
    # ===================================================================
    ckpt_path = os.path.join(checkpoint_dir, out_checkpoint)
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "hidden_dim": hidden_dim,
        "num_nodes": num_nodes,
        "k": k,
        "node_names": node_names,
        "mode": "select_then_analyze",
        "use_memory": use_memory,
    }
    if use_memory:
        checkpoint_data["memory_slots"] = memory_slots
        checkpoint_data["memory_dim"] = memory_dim
        # Save memory stats for analysis
        if hasattr(model, 'get_memory_stats'):
            checkpoint_data["memory_stats"] = model.get_memory_stats()

    torch.save(checkpoint_data, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SelectThenAnalyze (STA)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--node-names", type=str, default="auto",
                        help="'auto' reads from dataset, 'large' for 20-node, 'small' for 5-node")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Override num_nodes (for scalable datasets)")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--gamma-sparsity", type=float, default=0.1)
    parser.add_argument(
        "--min-active-nodes",
        type=int,
        default=1,
        help="Minimum active nodes encouraged in anomalous masks (set >1 for multi-root).",
    )
    parser.add_argument(
        "--dispersion-weight",
        type=float,
        default=0.25,
        help="Weight of anti-collapse entropy-floor regularization.",
    )
    parser.add_argument("--skip-contrastive", action="store_true")
    parser.add_argument("--use-severity-weights", action="store_true", help="Pondérer la CE par la sévérité (Data-Centric)")
    parser.add_argument("--curriculum", action="store_true", help="Activer le curriculum learning 3-phases")
    # SOTA: Memory Module arguments
    parser.add_argument("--use-memory", action="store_true", help="Enable Memory Module for recurrence recognition (SOTA)")
    parser.add_argument("--memory-slots", type=int, default=128, help="Number of memory slots")
    parser.add_argument("--memory-dim", type=int, default=64, help="Memory embedding dimension")
    parser.add_argument("--lambda-recurrence", type=float, default=0.1, help="Weight for recurrence classification loss")
    parser.add_argument("--lambda-memory", type=float, default=0.01, help="Weight for memory regularization loss")
    parser.add_argument("--lambda-causality", type=float, default=0.05, help="Weight for temporal causality loss")
    parser.add_argument("--out-checkpoint", type=str, default="sta.pt")
    args = parser.parse_args()

    # Resolve node names
    if args.node_names == "auto":
        payload = torch.load(args.dataset, weights_only=False)
        if isinstance(payload, dict) and "node_names" in payload:
            node_names = payload["node_names"]
        else:
            raise ValueError("Dataset does not contain node_names. "
                             "Specify --node-names large|small or regenerate.")
    elif args.node_names == "large":
        # Legacy 20-node topology (from simulated_data_large.py)
        NODE_NAMES_LARGE = [
            'gateway-web', 'gateway-mobile', 'gateway-admin',
            'auth', 'user-svc', 'order-svc', 'payment-svc', 'notification-svc',
            'search-svc', 'recommendation-svc', 'inventory-svc',
            'postgres-primary', 'postgres-replica', 'redis-cache', 'redis-session',
            'rabbitmq', 'elasticsearch', 'minio', 'monitoring', 'config-svc'
        ]
        node_names = NODE_NAMES_LARGE
    elif args.node_names == "small":
        from common import NODE_NAMES
        node_names = NODE_NAMES
    else:
        raise ValueError(f"Unknown --node-names: {args.node_names}")

    if args.num_nodes is not None and args.num_nodes != len(node_names):
        raise ValueError(f"--num-nodes {args.num_nodes} != len(node_names) {len(node_names)}")

    train_sta(
        dataset_path=args.dataset,
        node_names=node_names,
        hidden_dim=args.hidden_dim,
        k=args.k,
        contrastive_epochs=args.epochs,
        supervised_epochs=args.supervised_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        beta_max=args.beta_max,
        min_active_nodes=args.min_active_nodes,
        dispersion_weight=args.dispersion_weight,
        gamma_sparsity=args.gamma_sparsity,
        use_severity_weights=args.use_severity_weights,
        curriculum=args.curriculum,
        # SOTA: Memory Module
        use_memory=args.use_memory,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        lambda_recurrence=args.lambda_recurrence,
        lambda_memory=args.lambda_memory,
        lambda_causality=args.lambda_causality,
        skip_contrastive=args.skip_contrastive,
        out_checkpoint=args.out_checkpoint,
    )
