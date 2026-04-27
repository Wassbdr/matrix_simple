"""
Training loop — Semi-Supervised with Asymmetric GIB.

Phase 1: Self-supervised contrastive pre-training (mini-batched)
    - InfoNCE separates healthy vs anomalous dynamics
    - Asymmetric GIB forces mask→0 on healthy, mask→sparse on anomalous
    - GIB temperature annealing: warm start (1.0) → cool (0.3)

Phase 2: Supervised calibration
    - GIB mode: BCE on mask probs vs one-hot root cause (mask IS the predictor)
    - No-GIB mode: CE on per-node scorer logits

Loss:
    Phase 1: L_InfoNCE + β(t) * L_asymmetric_GIB
    Phase 2 (GIB): L_BCE(mask_probs, one_hot_target)
    Phase 2 (No-GIB): L_CE(node_scorer_logits, target)
"""

import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from augment import augment_positive
from common import NAME_TO_IDX, load_windows_dataset, stratified_split
from k_selection import adaptive_gib_beta
from losses import asymmetric_gib_loss, info_nce_loss, root_cause_loss
from model import SpatioTemporalRCA
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> List[Data]:
    logger.info("Loading dataset from %s", filepath)
    dataset: List[Data] = load_windows_dataset(filepath)
    logger.info("Loaded %d windowed graph samples.", len(dataset))
    return dataset


def split_by_labels(dataset: List[Data]) -> Tuple[List[Data], List[Data]]:
    healthy = [d for d in dataset if d.y is not None and d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in NAME_TO_IDX]
    logger.info("Split: %d healthy, %d anomalous (with valid root cause).", len(healthy), len(anomalous))
    return healthy, anomalous





def compute_beta(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    if warmup_epochs <= 0:
        return beta_max
    return min(1.0, epoch / warmup_epochs) * beta_max


def train(
    dataset_path: str,
    # Phase 1 params
    contrastive_epochs: int = 150,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    beta_max: float = 0.5,
    warmup_epochs: int = 50,
    temperature: float = 0.07,
    target_sparsity: float = 0.2,
    # Phase 2 params
    supervised_epochs: int = 100,
    supervised_lr: float = 1e-3,
    # General
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    use_gib: bool = True,
    out_checkpoint: str = "stgcn_rca_final.pt",
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = load_dataset(dataset_path)

    # Deterministic train/test split (same logic as benchmark.py)
    train_data, _ = stratified_split(dataset, train_ratio=0.8)
    logger.info("Using %d train samples (of %d total).", len(train_data), len(dataset))

    healthy, anomalous = split_by_labels(train_data)

    if len(healthy) < 2:
        logger.error("Need at least 2 healthy windows.")
        return

    rc_dist: Dict[str, int] = {}
    for d in anomalous:
        rc = d.root_cause[0]
        rc_dist[rc] = rc_dist.get(rc, 0) + 1
    logger.info("Root cause distribution: %s", rc_dist)

    # -- Build tensors --
    anchors_x = [healthy[i].x for i in range(len(healthy) - 1)]
    positives_x = [healthy[i + 1].x for i in range(len(healthy) - 1)]
    negatives_x = [d.x for d in anomalous]

    x_anchor = torch.stack(anchors_x).to(device)
    x_positive = torch.stack(positives_x).to(device)
    x_negative = torch.stack(negatives_x).to(device)
    edge_index = healthy[0].edge_index.to(device)

    logger.info("Anchor: %s, Positive: %s, Negative: %s",
                x_anchor.shape, x_positive.shape, x_negative.shape)

    num_features = x_anchor.size(2)
    num_nodes = x_anchor.size(1)

    # -- Initialize model --
    model = SpatioTemporalRCA(
        num_features=num_features, hidden_dim=hidden_dim, num_nodes=num_nodes,
    ).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # =====================================================
    # PHASE 1: Contrastive pre-training with asymmetric GIB
    # =====================================================
    if contrastive_epochs > 0:
        logger.info("=== PHASE 1: Contrastive + Asymmetric GIB (%d epochs) ===", contrastive_epochs)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=contrastive_epochs, eta_min=1e-5)

        n_healthy = x_anchor.size(0)
        n_neg = x_negative.size(0)
        batch_size_p1 = 512  # mini-batch size for Phase 1
        # Temperature annealing: start warm (1.0) for exploration, cool to 0.3 for sharp masks
        temp_start, temp_end = 1.0, 0.3

        for epoch in range(contrastive_epochs):
            model.train()
            beta_eff = adaptive_gib_beta(beta_max, num_nodes)
            beta = compute_beta(epoch, warmup_epochs, beta_eff)
            # Anneal GIB temperature
            if use_gib:
                t = epoch / max(contrastive_epochs - 1, 1)
                model.gib.set_temperature(temp_start + t * (temp_end - temp_start))

            perm_h = torch.randperm(n_healthy, device=device)
            perm_n = torch.randperm(n_neg, device=device)
            epoch_loss = 0.0
            n_batches = 0

            # Iterate in mini-batches
            for start in range(0, min(n_healthy, n_neg), batch_size_p1):
                idx_h = perm_h[start:start + batch_size_p1]
                idx_n = perm_n[start:start + batch_size_p1]

                optimizer.zero_grad()

                z_a, probs_a, _ = model(x_anchor[idx_h], edge_index, use_gib=use_gib)
                x_pos_aug = augment_positive(x_positive[idx_h], noise_sigma=0.05, jitter_shift=1, feature_mask_ratio=0.1)
                z_p, probs_p, _ = model(x_pos_aug, edge_index, use_gib=use_gib)
                z_n, probs_n, _ = model(x_negative[idx_n], edge_index, use_gib=use_gib)

                l_nce = info_nce_loss(z_a, z_p, z_n, temperature)

                if use_gib:
                    probs_healthy = torch.cat([probs_a, probs_p], dim=0)
                    l_gib = asymmetric_gib_loss(probs_healthy, probs_n, target_sparsity)
                else:
                    l_gib = torch.tensor(0.0, device=device)

                loss = l_nce + beta * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == contrastive_epochs - 1:
                with torch.no_grad():
                    _, pa, _ = model(x_anchor[:256], edge_index, use_gib=use_gib)
                    _, pn, _ = model(x_negative[:256], edge_index, use_gib=use_gib)
                    h_mask = pa.mean().item()
                    a_mask = pn.squeeze(-1)
                    a_mean, a_max = a_mask.mean().item(), a_mask.max(-1).values.mean().item()
                logger.info("[Phase1] Ep %04d/%04d | Loss: %.4f | β: %.1f | "
                            "H_mask: %.4f | A_mask mean: %.3f max: %.3f",
                            epoch, contrastive_epochs, epoch_loss / max(n_batches, 1),
                            beta, h_mask, a_mean, a_max)

    # =====================================================
    # PHASE 2: Supervised fine-tuning (classifier head)
    # =====================================================
    logger.info("=== PHASE 2: Supervised fine-tuning (%d epochs) ===", supervised_epochs)

    # Build supervised batch
    x_sup = x_negative  # all anomalous windows
    y_sup = torch.tensor([NAME_TO_IDX[d.root_cause[0]] for d in anomalous],
                          dtype=torch.long, device=device)
    logger.info("Supervised batch: %s, targets: %s", x_sup.shape, y_sup.shape)

    # Optimizer -- GIB case: only train GIB + backbone (mask IS the predictor)
    #              no-GIB case: train root_cause_head + backbone
    if use_gib:
        param_groups = [
            {'params': model.gib.parameters(), 'lr': supervised_lr},
            {'params': model.temporal_block.parameters(), 'lr': supervised_lr * 0.1},
            {'params': model.spatial_block.parameters(), 'lr': supervised_lr * 0.1},
        ]
    else:
        param_groups = [
            {'params': model.node_scorer.parameters(), 'lr': supervised_lr},
            {'params': model.temporal_block.parameters(), 'lr': supervised_lr * 0.1},
            {'params': model.spatial_block.parameters(), 'lr': supervised_lr * 0.1},
        ]
    optimizer_sup = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler_sup = optim.lr_scheduler.CosineAnnealingLR(optimizer_sup, T_max=supervised_epochs, eta_min=1e-5)

    batch_size_sup = 256
    n_sup = x_sup.size(0)

    for epoch in range(supervised_epochs):
        model.train()
        perm = torch.randperm(n_sup, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n_sup, batch_size_sup):
            idx = perm[start:start + batch_size_sup]
            x_batch = x_sup[idx]
            y_batch = y_sup[idx]

            optimizer_sup.zero_grad()

            z_graph, probs, h_spatial = model(x_batch, edge_index, use_gib=use_gib)

            if use_gib:
                # GIB is the sole predictor: CE on raw GIB logits (principled N-way classification)
                gib_logits = model.gib_logits(h_spatial)  # [batch, num_nodes]
                loss = F.cross_entropy(gib_logits, y_batch)
                preds = gib_logits.argmax(dim=1)
            else:
                # No GIB: per-node scorer on spatial embeddings
                rc_logits = model.node_scorer(h_spatial)  # [batch, num_nodes]
                loss = root_cause_loss(rc_logits, y_batch)
                preds = rc_logits.argmax(dim=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_sup.step()

            epoch_loss += loss.item()
            epoch_correct += (preds == y_batch).sum().item()
            n_batches += 1

        scheduler_sup.step()

        if epoch % 20 == 0 or epoch == supervised_epochs - 1:
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_correct / n_sup
            with torch.no_grad():
                _, probs_all, _ = model(x_sup[:256], edge_index, use_gib=use_gib)
                mask_density = probs_all.mean().item()
            logger.info("[Phase2] Ep %04d/%04d | Loss: %.4f | Acc: %.4f | Mask: %.4f",
                        epoch, supervised_epochs, avg_loss, avg_acc, mask_density)

    # -- Save --
    checkpoint_path = os.path.join(checkpoint_dir, out_checkpoint)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_sup.state_dict(),
        "epoch": contrastive_epochs + supervised_epochs,
        "beta_max": beta_max,
        "num_features": num_features,
        "num_nodes": num_nodes,
        "hidden_dim": hidden_dim,
        "mode": "semi_supervised",
        "use_gib": use_gib,
    }, checkpoint_path)
    logger.info("Final checkpoint saved to %s", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RCA (Semi-Supervised: Contrastive + Classifier).")
    parser.add_argument("--dataset", type=str, default="dataset_windows.pt")
    parser.add_argument("--epochs", type=int, default=150, help="Phase 1 contrastive epochs")
    parser.add_argument("--supervised-epochs", type=int, default=100, help="Phase 2 classifier epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--target-sparsity", type=float, default=0.2)
    parser.add_argument("--no-gib", action="store_true", help="Disable GIB for ablation study")
    parser.add_argument("--out-checkpoint", type=str, default="stgcn_rca_final.pt", help="Output checkpoint filename")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        contrastive_epochs=args.epochs,
        supervised_epochs=args.supervised_epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        temperature=args.temperature,
        target_sparsity=args.target_sparsity,
        use_gib=not args.no_gib,
        out_checkpoint=args.out_checkpoint,
    )
