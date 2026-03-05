"""
Training loop for SelectThenAnalyze model (v2).

Two-phase training with GIB selection:
    Phase 1: Contrastive pre-training on full graph
        - InfoNCE to separate healthy vs anomalous dynamics
        - GIB sparsity loss: push mask→uniform on healthy, →peaked on anomalous
        - Temperature annealing: warm (2.0) → cool (0.5)

    Phase 2: Supervised fine-tuning
        - Joint loss: CE on subgraph scorer + λ * CE on GIB probs
        - Both the selector and the scorer learn to identify the root cause
        - GIB probs provide complementary signal (regularization)

Key difference from v1: the GIB loss uses KL(mask || uniform) for healthy
and negative entropy for anomalous — both respect the softmax constraint.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import logging
import argparse
import os

from model_v2 import SelectThenAnalyze
from losses import info_nce_loss
from augment import augment_positive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> List[Data]:
    dataset: List[Data] = torch.load(filepath, weights_only=False)
    logger.info("Loaded %d samples from %s", len(dataset), filepath)
    return dataset


def stratified_split(dataset: List[Data], node_names: List[str], train_ratio: float = 0.8):
    """Stratified split that works with any node name list."""
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


def softmax_gib_loss(probs_healthy: torch.Tensor, probs_anomalous: torch.Tensor) -> torch.Tensor:
    """
    GIB loss for softmax mask probabilities.
    
    Healthy: push toward uniform distribution → KL(probs || uniform)
        If the mask is uniform, no node is "selected" — correct for healthy windows.
    Anomalous: push toward peaked (one-hot) → negative entropy
        Minimizing entropy forces the mask to concentrate on one node.
    
    Both terms respect ∑ pᵢ = 1 by construction (softmax output).
    
    Args:
        probs_healthy:   [B_h, N] — softmax probs for healthy windows
        probs_anomalous: [B_a, N] — softmax probs for anomalous windows
    Returns:
        scalar loss
    """
    N = probs_healthy.size(-1)
    
    # Healthy: KL(probs || uniform) — want mask to be uninformative
    uniform = torch.ones_like(probs_healthy) / N
    loss_healthy = F.kl_div(
        (probs_healthy + 1e-8).log(), uniform, reduction='batchmean'
    )
    
    # Anomalous: minimize entropy → push toward one-hot
    entropy = -(probs_anomalous * (probs_anomalous + 1e-8).log()).sum(dim=-1)  # [B_a]
    max_entropy = torch.log(torch.tensor(float(N), device=entropy.device))
    loss_anomalous = (entropy / max_entropy).mean()  # normalized [0, 1]
    
    return loss_healthy + loss_anomalous


def train_v2(
    dataset_path: str,
    node_names: List[str],
    # Architecture
    hidden_dim: int = 32,
    k: int = 5,
    # Phase 1
    contrastive_epochs: int = 100,
    lr: float = 1e-3,
    beta_max: float = 0.5,
    warmup_epochs: int = 30,
    temperature: float = 0.07,
    # Phase 2
    supervised_epochs: int = 100,
    supervised_lr: float = 1e-3,
    lambda_gib: float = 0.3,
    # General
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    out_checkpoint: str = "select_then_analyze.pt",
    skip_contrastive: bool = False,
) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    num_nodes = len(node_names)
    
    dataset = load_dataset(dataset_path)
    train_data, _ = stratified_split(dataset, node_names, train_ratio=0.8)
    logger.info("Train: %d samples (of %d total)", len(train_data), len(dataset))
    
    healthy = [d for d in train_data if d.y is not None and d.y.item() == 0]
    anomalous = [d for d in train_data
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in name_to_idx]
    logger.info("Healthy: %d, Anomalous: %d", len(healthy), len(anomalous))
    
    rc_dist = {}
    for d in anomalous:
        rc = d.root_cause[0]
        rc_dist[rc] = rc_dist.get(rc, 0) + 1
    logger.info("Root cause distribution: %s", rc_dist)
    
    # Build tensors
    x_anchor = torch.stack([healthy[i].x for i in range(len(healthy) - 1)]).to(device)
    x_positive = torch.stack([healthy[i + 1].x for i in range(len(healthy) - 1)]).to(device)
    x_negative = torch.stack([d.x for d in anomalous]).to(device)
    edge_index = healthy[0].edge_index.to(device)
    
    num_features = x_anchor.size(2)
    
    model = SelectThenAnalyze(
        num_features=num_features, hidden_dim=hidden_dim,
        num_nodes=num_nodes, k=k, gib_temperature=2.0,
    ).to(device)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: Contrastive pre-training
    # ═══════════════════════════════════════════════════════
    if contrastive_epochs > 0 and not skip_contrastive:
        logger.info("=== PHASE 1: Contrastive + GIB Selector (%d epochs) ===", contrastive_epochs)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=contrastive_epochs, eta_min=1e-5)
        
        n_h = x_anchor.size(0)
        n_n = x_negative.size(0)
        batch_size = 256
        temp_start, temp_end = 2.0, 0.5
        
        for epoch in range(contrastive_epochs):
            model.train()
            beta = min(1.0, epoch / max(warmup_epochs, 1)) * beta_max
            
            # Temperature annealing
            t = epoch / max(contrastive_epochs - 1, 1)
            model.set_gib_temperature(temp_start + t * (temp_end - temp_start))
            
            perm_h = torch.randperm(n_h, device=device)
            perm_n = torch.randperm(n_n, device=device)
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                
                optimizer.zero_grad()
                
                # Forward all three through the model
                # For contrastive, we use the light encoder embeddings
                # (z_graph is not meaningful yet, but probs are)
                _, probs_a, _, _ = model(x_anchor[idx_h], edge_index)
                x_pos_aug = augment_positive(
                    x_positive[idx_h], noise_sigma=0.05,
                    jitter_shift=1, feature_mask_ratio=0.1,
                )
                _, probs_p, _, _ = model(x_pos_aug, edge_index)
                _, probs_n, _, _ = model(x_negative[idx_n], edge_index)
                
                # For InfoNCE, we need graph-level representations
                # Use the mean of light encoder embeddings (cheapest path)
                B_h = idx_h.size(0)
                B_n = idx_n.size(0)
                x_a_flat = x_anchor[idx_h].view(B_h * num_nodes, num_features, -1)
                x_p_flat = x_pos_aug.view(B_h * num_nodes, num_features, -1)
                x_n_flat = x_negative[idx_n].view(B_n * num_nodes, num_features, -1)
                
                if B_h > 1:
                    offsets = torch.arange(B_h, device=device) * num_nodes
                    batch_ei_h = edge_index.unsqueeze(0).expand(B_h, -1, -1) + offsets.view(-1, 1, 1)
                    batch_ei_h = batch_ei_h.reshape(2, -1)
                else:
                    batch_ei_h = edge_index
                if B_n > 1:
                    offsets_n = torch.arange(B_n, device=device) * num_nodes
                    batch_ei_n = edge_index.unsqueeze(0).expand(B_n, -1, -1) + offsets_n.view(-1, 1, 1)
                    batch_ei_n = batch_ei_n.reshape(2, -1)
                else:
                    batch_ei_n = edge_index
                
                h_a = model.light_encoder(x_a_flat, batch_ei_h).view(B_h, num_nodes, -1).mean(dim=1)
                h_p = model.light_encoder(x_p_flat, batch_ei_h).view(B_h, num_nodes, -1).mean(dim=1)
                h_n = model.light_encoder(x_n_flat, batch_ei_n).view(B_n, num_nodes, -1).mean(dim=1)
                
                l_nce = info_nce_loss(h_a, h_p, h_n, temperature)
                
                # GIB selection loss
                probs_healthy = torch.cat([probs_a, probs_p], dim=0)
                l_gib = softmax_gib_loss(probs_healthy, probs_n)
                
                loss = l_nce + beta * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            if epoch % 20 == 0 or epoch == contrastive_epochs - 1:
                with torch.no_grad():
                    _, pa, _, _ = model(x_anchor[:128], edge_index)
                    _, pn, _, _ = model(x_negative[:128], edge_index)
                    h_entropy = -(pa * (pa + 1e-8).log()).sum(-1).mean().item()
                    a_entropy = -(pn * (pn + 1e-8).log()).sum(-1).mean().item()
                    max_ent = torch.log(torch.tensor(float(num_nodes))).item()
                logger.info(
                    "[Phase1] Ep %03d/%03d | Loss: %.4f | β: %.2f | "
                    "H_entropy: %.2f/%.2f | A_entropy: %.2f/%.2f",
                    epoch, contrastive_epochs, epoch_loss / max(n_batches, 1),
                    beta, h_entropy, max_ent, a_entropy, max_ent,
                )
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: Supervised fine-tuning
    # ═══════════════════════════════════════════════════════
    logger.info("=== PHASE 2: Supervised fine-tuning (%d epochs) ===", supervised_epochs)
    
    x_sup = x_negative
    y_sup = torch.tensor([name_to_idx[d.root_cause[0]] for d in anomalous],
                          dtype=torch.long, device=device)
    
    logger.info("Supervised: %s, targets: %s", x_sup.shape, y_sup.shape)
    
    # All parameters with differential lr
    param_groups = [
        {'params': model.gib_selector.parameters(), 'lr': supervised_lr},
        {'params': model.scorer.parameters(), 'lr': supervised_lr},
        {'params': model.deep_temporal.parameters(), 'lr': supervised_lr},
        {'params': model.deep_spatial.parameters(), 'lr': supervised_lr},
        {'params': model.light_encoder.parameters(), 'lr': supervised_lr * 0.1},
    ]
    optimizer_sup = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler_sup = optim.lr_scheduler.CosineAnnealingLR(optimizer_sup, T_max=supervised_epochs, eta_min=1e-5)
    
    batch_size_sup = 256
    n_sup = x_sup.size(0)
    
    # Fix GIB temperature for Phase 2
    model.set_gib_temperature(0.5)
    
    for epoch in range(supervised_epochs):
        model.train()
        perm = torch.randperm(n_sup, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_selected = 0  # track if root cause is in top-k
        n_batches = 0
        
        for start in range(0, n_sup, batch_size_sup):
            idx = perm[start:start + batch_size_sup]
            x_batch = x_sup[idx]
            y_batch = y_sup[idx]
            
            optimizer_sup.zero_grad()
            
            logits_full, probs, top_k, gib_logits = model(x_batch, edge_index)
            
            # Primary loss: CE on raw GIB logits (numerically stable, covers all N)
            loss_gib = F.cross_entropy(gib_logits / model.gib_selector.temperature, y_batch)
            
            # Secondary loss: CE on scorer, only for samples where RC ∈ top-k
            in_topk = torch.zeros(y_batch.size(0), dtype=torch.bool, device=device)
            for i in range(y_batch.size(0)):
                if y_batch[i] in top_k[i]:
                    in_topk[i] = True
            
            if in_topk.any():
                loss_scorer = F.cross_entropy(logits_full[in_topk], y_batch[in_topk])
            else:
                loss_scorer = torch.tensor(0.0, device=device)
            
            loss = lambda_gib * loss_gib + loss_scorer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_sup.step()
            
            # Prediction: use GIB probs (always defined) for accuracy
            preds = probs.argmax(dim=1)
            epoch_loss += loss.item()
            epoch_correct += (preds == y_batch).sum().item()
            epoch_selected += in_topk.sum().item()
            n_batches += 1
        
        scheduler_sup.step()
        
        if epoch % 20 == 0 or epoch == supervised_epochs - 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_acc = epoch_correct / n_sup
            selection_rate = epoch_selected / n_sup
            logger.info(
                "[Phase2] Ep %03d/%03d | Loss: %.4f | Acc: %.4f | "
                "RC-in-top-%d: %.4f",
                epoch, supervised_epochs, avg_loss, avg_acc,
                model.k, selection_rate,
            )
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, out_checkpoint)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "hidden_dim": hidden_dim,
        "num_nodes": num_nodes,
        "k": model.k,
        "node_names": node_names,
        "mode": "select_then_analyze",
    }, checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SelectThenAnalyze (v2)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--node-names", type=str, default="large",
                        help="'large' for 20-node, 'small' for 5-node")
    parser.add_argument("--epochs", type=int, default=100, help="Phase 1 epochs")
    parser.add_argument("--supervised-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=5, help="Number of nodes to select")
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--lambda-gib", type=float, default=0.3)
    parser.add_argument("--skip-contrastive", action="store_true")
    parser.add_argument("--out-checkpoint", type=str, default="select_then_analyze.pt")
    args = parser.parse_args()
    
    if args.node_names == "large":
        from synthetic_data_large import NODE_NAMES_LARGE
        node_names = NODE_NAMES_LARGE
    else:
        from common import NODE_NAMES
        node_names = NODE_NAMES
    
    train_v2(
        dataset_path=args.dataset,
        node_names=node_names,
        hidden_dim=args.hidden_dim,
        k=args.k,
        contrastive_epochs=args.epochs,
        supervised_epochs=args.supervised_epochs,
        lr=args.lr,
        beta_max=args.beta_max,
        lambda_gib=args.lambda_gib,
        skip_contrastive=args.skip_contrastive,
        out_checkpoint=args.out_checkpoint,
    )
