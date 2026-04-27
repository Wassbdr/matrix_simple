"""
training_utils.py — Shared training infrastructure for RCA models.

Extracts common patterns from benchmark.py, train_sta.py, and benchmark_real.py:
    - EarlyStopping (MRR-based, patience-based)
    - get_scheduler (unified LR scheduler factory)
    - TwoPhaseTrainer (contrastive warmup + supervised fine-tuning protocol)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Patience-based early stopping monitored on MRR (higher=better).
    Minimum recommended patience: 75 epochs (validated for N>=50).
    """
    def __init__(self, patience: int = 75):
        self.patience = patience
        self.best_mrr = -float('inf')
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, val_mrr: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_mrr > self.best_mrr + 1e-6:
            self.best_mrr = val_mrr
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    total_epochs: int,
    eta_min: float = 1e-5,
    T_0: int = 50,
    T_mult: int = 2,
) -> optim.lr_scheduler.LRScheduler:
    """
    Unified LR scheduler factory.

    Args:
        name: "cosine" | "warm_restarts" | "none"
        optimizer: the optimizer to schedule
        total_epochs: T_max for cosine annealing
        eta_min: minimum learning rate
        T_0: restart period for warm_restarts
        T_mult: restart multiplier for warm_restarts
    """
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )
    elif name == "warm_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    elif name == "none":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {name}. Use 'cosine', 'warm_restarts', or 'none'.")


class TwoPhaseTrainer:
    """
    Protocol for two-phase training:
        Phase 1: Contrastive warmup (InfoNCE on Z_subgraph + variational GIB)
        Phase 2: Supervised fine-tuning with optional k-annealing

    Encapsulates the common training loop patterns shared across
    benchmark.py, train_sta.py, and benchmark_real.py.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: str = "cosine",
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name

    def contrastive_phase(
        self,
        x_anchor: torch.Tensor,
        x_positive: torch.Tensor,
        x_negative: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int,
        epochs: int = 20,
        batch_size: int = 256,
        nce_temperature: float = 0.07,
        beta_max: float = 0.3,
    ) -> float:
        """
        Phase 1: Contrastive warmup with mask-weighted InfoNCE + variational GIB.
        Returns final loss value.
        """
        from augment import augment_positive
        from losses import info_nce_loss, variational_gib_loss

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        target_rate = k / num_nodes
        self.model.set_temperature(2.0)
        n_h, n_n = x_anchor.size(0), x_negative.size(0)
        num_features = x_anchor.size(2)
        last_loss = 0.0

        for epoch in range(epochs):
            self.model.train()
            beta = min(1.0, epoch / max(epochs // 2, 1)) * beta_max
            perm_h = torch.randperm(n_h, device=self.device)
            perm_n = torch.randperm(n_n, device=self.device)

            for start in range(0, min(n_h, n_n), batch_size):
                idx_h = perm_h[start:start + batch_size]
                idx_n = perm_n[start:start + batch_size]
                B_h, B_n = idx_h.size(0), idx_n.size(0)

                optimizer.zero_grad()

                _, M_a, _, _ = self.model(x_anchor[idx_h], edge_index)
                x_pos_aug = augment_positive(x_positive[idx_h], noise_sigma=0.05,
                                             jitter_shift=1, feature_mask_ratio=0.1)
                _, M_p, _, _ = self.model(x_pos_aug, edge_index)
                _, M_n, _, _ = self.model(x_negative[idx_n], edge_index)

                h_a = self.model.light_encoder(x_anchor[idx_h].view(B_h * num_nodes, num_features, -1)).view(B_h, num_nodes, -1)
                h_p = self.model.light_encoder(x_pos_aug.view(B_h * num_nodes, num_features, -1)).view(B_h, num_nodes, -1)
                h_n = self.model.light_encoder(x_negative[idx_n].view(B_n * num_nodes, num_features, -1)).view(B_n, num_nodes, -1)

                z_a = (h_a * M_a.unsqueeze(-1)).sum(1) / (M_a.sum(1, keepdim=True) + 1e-8)
                z_p = (h_p * M_p.unsqueeze(-1)).sum(1) / (M_p.sum(1, keepdim=True) + 1e-8)
                z_n = (h_n * M_n.unsqueeze(-1)).sum(1) / (M_n.sum(1, keepdim=True) + 1e-8)

                l_nce = info_nce_loss(z_a, z_p, z_n, nce_temperature)
                l_gib = variational_gib_loss(torch.cat([M_a, M_p, M_n], dim=0), target_rate)

                loss = l_nce + beta * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                last_loss = loss.item()

        self.model.set_temperature(1.0)
        return last_loss

    def supervised_phase(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        edge_index: torch.Tensor,
        epochs: int = 150,
        batch_size: int = 256,
        k_anneal_epochs: int = 60,
        k_target: int = 5,
        k_warm: Optional[int] = None,
        val_data: Optional[tuple] = None,
        patience: int = 75,
    ) -> None:
        """
        Phase 2: Supervised with k-annealing and optional early stopping.
        """
        from losses import variational_gib_loss

        N = x_train.size(1)
        if k_warm is None:
            k_warm = min(N // 2, N - 1)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_scheduler(self.scheduler_name, optimizer, epochs)
        es = EarlyStopping(patience=patience) if val_data else None

        for epoch in range(epochs):
            self.model.train()

            if epoch < k_anneal_epochs:
                k_cur = k_warm - int((k_warm - k_target) * epoch / k_anneal_epochs)
            else:
                k_cur = k_target
            self.model.set_k(k_cur)

            t_frac = epoch / max(epochs - 1, 1)
            self.model.set_temperature(1.0 + t_frac * (0.3 - 1.0))

            perm = torch.randperm(x_train.size(0), device=self.device)
            for start in range(0, x_train.size(0), batch_size):
                idx = perm[start:start + batch_size]
                x_b, y_b = x_train[idx], y_train[idx]
                optimizer.zero_grad()

                logits_full, M_soft, top_k, sel_logits = self.model(x_b, edge_index)

                loss_sel = F.cross_entropy(sel_logits, y_b)
                in_topk = (top_k == y_b.unsqueeze(1)).any(dim=1)
                loss_scorer = F.cross_entropy(logits_full[in_topk], y_b[in_topk]) if in_topk.any() else torch.tensor(0.0, device=self.device)
                l_gib = variational_gib_loss(M_soft, k_target / N)

                loss = loss_sel + loss_scorer + 0.1 * l_gib
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            if es and val_data:
                x_val, y_val = val_data
                self.model.eval()
                with torch.no_grad():
                    _, _, _, sv = self.model(x_val, edge_index)
                    ranks = []
                    for i in range(sv.size(0)):
                        sorted_idx = sv[i].argsort(descending=True)
                        rank = (sorted_idx == y_val[i]).nonzero(as_tuple=True)[0].item() + 1
                        ranks.append(1.0 / rank)
                    val_mrr = sum(ranks) / len(ranks)
                if es.step(val_mrr, self.model):
                    logger.info("Early stopping at epoch %d (MRR=%.4f)", epoch, val_mrr)
                    es.restore(self.model)
                    break

        self.model.set_k(None)
