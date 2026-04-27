"""
Training loop for SelectThenAnalyze (STA) v2.

Two-phase training aligned with the formalization:

Phase 1 -- Contrastive pre-training (self-supervised):
    L = L_InfoNCE(z_a, z_p, z_n) + beta * L_GIB(M_soft, target_rate)
    - InfoNCE on mask-weighted embeddings: z = (H_fast * M_soft).sum / M_soft.sum
      Gradient flows end-to-end through the selection mask.
    - Variational GIB (KL Bernoulli) replaces L1 sparsity for a principled
      bound on I(Z; X), pushing M_soft toward a target selection rate k/N.
    - Temperature annealing (warm -> cool) for stable exploration.
    - beta warm-up prevents premature sparsification.

Phase 2 -- Supervised fine-tuning:
    L = L_CE(selector_logits, y)            [all N nodes, primary]
      + L_CE(scorer_logits, y) [if RC in top-k]  [subgraph, secondary]
      + gamma * L_GIB(M_soft, k/N)          [variational GIB regularization]
    - Selector receives direct gradient from CE on its logits.
    - Scorer receives gradient from CE only when the root cause was
      correctly selected in top-k (adaptive per-sample).
    - Both paths also give gradient to the selector through the STE.

Usage:
    python train_sta.py --dataset scalable_dataset_N100.pt --num-nodes 100 --k 10
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from augment import augment_positive
from common import set_all_seeds
from k_selection import get_optimal_k
from losses import info_nce_loss, variational_gib_loss
from model_sta import SelectThenAnalyze
from torch_geometric.data import Data

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


def stratified_split(dataset: List[Data], name_to_idx: Dict[str, int],
                     train_ratio: float = 0.8):
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


# ---------------------------------------------------------------------------
# Sparsity loss (L1 on M_soft)
# ---------------------------------------------------------------------------

def sparsity_loss_l1(M_soft: torch.Tensor) -> torch.Tensor:
    """||M_soft||_1 averaged over batch.  M_soft: [B, N]."""
    return M_soft.abs().sum(dim=-1).mean()


def asymmetric_sparsity_loss(
    M_soft_healthy: torch.Tensor,
    M_soft_anomalous: torch.Tensor,
) -> torch.Tensor:
    """
    Asymmetric variant matching the dynamical system assumption:
        healthy  (U=0):     push M -> 0          (L2)
        anomalous (|U|_0=1): push M -> sparse     (L1 + neg-entropy)
    """
    loss_h = (M_soft_healthy ** 2).sum(dim=-1).mean()

    # Anomalous: L1 + negative entropy
    loss_l1 = M_soft_anomalous.abs().sum(dim=-1).mean()
    p = M_soft_anomalous / (M_soft_anomalous.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
    N = M_soft_anomalous.size(-1)
    max_ent = torch.log(torch.tensor(float(N), device=entropy.device))
    loss_conc = (entropy / max_ent).mean()

    return loss_h + loss_l1 + loss_conc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sta(
    dataset_path: str,
    node_names: List[str],
    # Architecture
    hidden_dim: int = 32,
    k: Optional[int] = None,
    # Phase 1
    contrastive_epochs: int = 100,
    lr: float = 1e-3,
    beta_max: float = 0.5,
    warmup_epochs: int = 30,
    nce_temperature: float = 0.07,
    # Phase 2
    supervised_epochs: int = 100,
    supervised_lr: float = 1e-3,
    gamma_sparsity: float = 0.1,
    # General
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    out_checkpoint: str = "sta.pt",
    skip_contrastive: bool = False,
    seed: int = 42,
) -> None:

    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    num_nodes = len(node_names)
    if k is None:
        k = get_optimal_k(num_nodes)
    logger.info("Using k=%d (get_optimal_k if auto)", k)

    dataset, _ = load_scalable_dataset(dataset_path)
    train_data, _ = stratified_split(dataset, name_to_idx)
    logger.info("Train: %d / %d total", len(train_data), len(dataset))

    healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
    anomalous = [d for d in train_data
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and d.root_cause[0] in name_to_idx]
    logger.info("Healthy: %d, Anomalous: %d", len(healthy), len(anomalous))

    # Build tensors
    x_anchor = torch.stack([healthy[i].x for i in range(len(healthy) - 1)]).to(device)
    x_positive = torch.stack([healthy[i + 1].x for i in range(len(healthy) - 1)]).to(device)
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
            model.set_temperature(temp_start + t_frac * (temp_end - temp_start))

            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                B_h, B_n = idx_h.size(0), idx_n.size(0)

                optimizer.zero_grad()

                # Full forward through model — InfoNCE on Z_subgraph (not H_fast)
                logits_a, M_soft_a, _, _ = model(x_anchor[idx_h], edge_index)
                x_pos_aug = augment_positive(
                    x_positive[idx_h], noise_sigma=0.05,
                    jitter_shift=1, feature_mask_ratio=0.1,
                )
                logits_p, M_soft_p, _, _ = model(x_pos_aug, edge_index)
                logits_n, M_soft_n, _, _ = model(x_negative[idx_n], edge_index)

                # Graph-level embeddings from scorer logits (Z_subgraph path)
                # Replace -inf with 0 for mean-pooling stability
                z_a = logits_a.clamp(min=-10.0).mean(dim=1)                  # [B_h, 1] -> expand
                z_p = logits_p.clamp(min=-10.0).mean(dim=1)
                z_n = logits_n.clamp(min=-10.0).mean(dim=1)
                # Project to embedding space for contrastive loss
                z_a_full = model.light_encoder(x_anchor[idx_h].view(B_h * num_nodes, num_features, -1)).view(B_h, num_nodes, -1)
                z_p_full = model.light_encoder(x_pos_aug.view(B_h * num_nodes, num_features, -1)).view(B_h, num_nodes, -1)
                z_n_full = model.light_encoder(x_negative[idx_n].view(B_n * num_nodes, num_features, -1)).view(B_n, num_nodes, -1)

                # Weighted mean pool using M_soft as attention (end-to-end gradient)
                h_a = (z_a_full * M_soft_a.unsqueeze(-1)).sum(1) / (M_soft_a.sum(1, keepdim=True) + 1e-8)
                h_p = (z_p_full * M_soft_p.unsqueeze(-1)).sum(1) / (M_soft_p.sum(1, keepdim=True) + 1e-8)
                h_n = (z_n_full * M_soft_n.unsqueeze(-1)).sum(1) / (M_soft_n.sum(1, keepdim=True) + 1e-8)

                l_nce = info_nce_loss(h_a, h_p, h_n, nce_temperature)

                # Variational GIB (KL Bernoulli) replaces L1 sparsity
                target_rate = k / num_nodes
                M_all = torch.cat([M_soft_a, M_soft_p, M_soft_n], dim=0)
                l_gib = variational_gib_loss(M_all, target_rate)

                loss = l_nce + beta * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == contrastive_epochs - 1:
                with torch.no_grad():
                    _, ma, _, _ = model(x_anchor[:64], edge_index)
                    _, mn, _, _ = model(x_negative[:64], edge_index)
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
    y_sup = torch.tensor(
        [name_to_idx[d.root_cause[0]] for d in anomalous],
        dtype=torch.long, device=device,
    )

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

    n_sup = x_sup.size(0)
    model.set_temperature(0.5)

    for epoch in range(supervised_epochs):
        model.train()
        perm = torch.randperm(n_sup, device=device)
        ep_loss = ep_correct = ep_selected = 0
        n_batches = 0

        for start in range(0, n_sup, batch_size):
            idx = perm[start:start + batch_size]
            x_b = x_sup[idx]
            y_b = y_sup[idx]

            optimizer_sup.zero_grad()

            logits_full, M_soft, top_k, sel_logits = model(x_b, edge_index)

            # Primary: CE on selector logits (all N, numerically stable)
            loss_sel = F.cross_entropy(sel_logits, y_b)

            # Secondary: CE on scorer (only when RC in top-k)
            in_topk = (top_k == y_b.unsqueeze(1)).any(dim=1)
            if in_topk.any():
                loss_scorer = F.cross_entropy(logits_full[in_topk], y_b[in_topk])
            else:
                loss_scorer = torch.tensor(0.0, device=device)

            # Variational GIB regularization (replaces L1)
            l_gib = variational_gib_loss(M_soft, k / num_nodes)

            loss = loss_sel + loss_scorer + gamma_sparsity * l_gib
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
            logger.info(
                "[Phase2] Ep %03d/%03d | Loss %.4f | Acc %.4f | "
                "RC-in-top-%d %.4f | M_mean %.3f",
                epoch, supervised_epochs,
                ep_loss / max(n_batches, 1),
                ep_correct / n_sup,
                k,
                ep_selected / n_sup,
                M_soft.mean().item(),
            )

    # ===================================================================
    # Save
    # ===================================================================
    ckpt_path = os.path.join(checkpoint_dir, out_checkpoint)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "hidden_dim": hidden_dim,
        "num_nodes": num_nodes,
        "k": k,
        "node_names": node_names,
        "mode": "select_then_analyze",
    }, ckpt_path)
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
    parser.add_argument("--k", type=int, default=None,
                        help="Selector top-k; omit for automatic get_optimal_k(N)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--gamma-sparsity", type=float, default=0.1)
    parser.add_argument("--skip-contrastive", action="store_true")
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
        from simulated_data_large import NODE_NAMES_LARGE
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
        gamma_sparsity=args.gamma_sparsity,
        skip_contrastive=args.skip_contrastive,
        out_checkpoint=args.out_checkpoint,
        seed=args.seed,
    )
