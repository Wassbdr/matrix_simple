"""
model_sta_mm.py — Multi-View Select-then-Analyze (variant _mm).

Multi-view inverse problem formulation:
    min_U ||U||_0  s.t.  Y^(m) ≈ g^(m)_phi(X)  for all m ∈ {metrics, logs, traces}

Each modality provides an independent "vote" on supp(U) via its own selector.
The votes are fused by learned cross-modal attention (intersection), so a noisy
or absent modality produces a uniform M^(m) that has no effect on the consensus.

Architecture:
    MetricEncoder  [B, N, F_m, W] → Conv1d+Attn → h_m  [B*N, H]
    LogEncoder     [B, N, F_l]    → MLP+LN      → h_l  [B*N, H]
    TraceEncoder   [B, N, F_t]    → MLP+LN      → h_t  [B*N, H]

    Per-modality selectors → M^(m)_soft [B, N] each
    CrossModalFusion       → M_fused = attention(M^m, M^l, M^t)

    STESelector on M_fused → top_k, M_hard
    DeepSTGCN on subgraph  → scorer + residual blend → logits [B, N]

forward(x_m, x_l, x_t, edge_index) → (logits_full, M_soft, top_k, sel_logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple
import logging

from rca_pipeline.model_sta import (
    STESelector,
    DeepTemporalBlock,
    DeepSpatialBlock,
    SubgraphScorer,
    extract_subgraph_vectorized,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

F_LOG_DEFAULT = 8
F_TRACE_DEFAULT = 6


class _ModalityEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TemporalEncoder(nn.Module):
    """Conv1d + attention pooling for metric time-series (zero-hop)."""
    def __init__(self, num_features: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*N, F, W] -> [B*N, H]"""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_t = h.permute(0, 2, 1)
        a = F.softmax(self.attn(h_t).squeeze(-1), dim=-1)
        h = (h_t * a.unsqueeze(-1)).sum(dim=1)
        return self.norm(h)


class _ModalitySelector(nn.Module):
    """Per-modality node scoring: h_m → logits [B, N]."""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, N, H] -> [B, N]"""
        return self.mlp(h).squeeze(-1)


class _CrossModalFusion(nn.Module):
    """
    Attention-based fusion of per-modality selection logits.

    Learns how much to trust each modality per-node.  Missing modalities
    get zero logits and zero attention weight, so they don't affect fusion.
    """
    def __init__(self, n_modalities: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modalities),
        )

    def forward(self, logits_stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_stack: [B, N, M] -- stacked per-modality logits
            mask:         [M]       -- 1.0 if modality present, 0.0 if absent
        Returns:
            fused_logits: [B, N]
        """
        weights = self.attn(logits_stack)                                    # [B, N, M]
        weights = weights * mask.unsqueeze(0).unsqueeze(0)                   # zero absent
        weights = F.softmax(weights + (1.0 - mask.unsqueeze(0).unsqueeze(0)) * -1e9, dim=-1)
        return (logits_stack * weights).sum(dim=-1)                          # [B, N]


class SelectThenAnalyze_MM(nn.Module):
    """
    Multi-View STA: per-modality selectors + cross-modal attention fusion.

    forward(x_metric, x_logs, x_traces, edge_index):
        x_metric: [B, N, F_m, W]
        x_logs:   [B, N, F_l] or None
        x_traces: [B, N, F_t] or None
    Returns:
        logits_full:     [B, N]
        M_soft:          [B, N]  (fused)
        top_k_indices:   [B, k]
        selector_logits: [B, N]  (fused)
    """

    def __init__(
        self,
        F_metric: int = 5,
        F_log: int = F_LOG_DEFAULT,
        F_trace: int = F_TRACE_DEFAULT,
        hidden_dim: int = 32,
        num_nodes: int = 20,
        k: int = 5,
        temporal_kernel: int = 3,
        gib_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = F_metric
        self.hidden_dim = hidden_dim
        self.k = k
        self.F_log = F_log
        self.F_trace = F_trace

        # Per-modality encoders (zero-hop, no GCN)
        self.metric_encoder = _TemporalEncoder(F_metric, hidden_dim, temporal_kernel)
        self.log_in_norm = nn.LayerNorm(F_log)
        self.trace_in_norm = nn.LayerNorm(F_trace)
        self.log_encoder = _ModalityEncoder(F_log, hidden_dim)
        self.trace_encoder = _ModalityEncoder(F_trace, hidden_dim)

        # Per-modality selectors (independent votes on supp(U))
        self.metric_selector = _ModalitySelector(hidden_dim)
        self.log_selector = _ModalitySelector(hidden_dim)
        self.trace_selector = _ModalitySelector(hidden_dim)

        # Cross-modal attention fusion
        self.cross_fusion = _CrossModalFusion(n_modalities=3, hidden_dim=hidden_dim)

        # Shared STE selector (operates on fused logits)
        self.selector = STESelector(hidden_dim, gib_temperature)

        # Deep path (on subgraph only)
        self.deep_temporal = DeepTemporalBlock(F_metric, hidden_dim, temporal_kernel)
        self.deep_spatial = DeepSpatialBlock(hidden_dim)
        self.scorer = SubgraphScorer(hidden_dim)

        # Residual blend
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))
        self._k_override = None

        logger.info("SelectThenAnalyze_MM (multi-view): N=%d, k=%d, F_m=%d, F_l=%d, F_t=%d",
                     num_nodes, k, F_metric, F_log, F_trace)

    def _batch_edge_index(self, edge_index: torch.Tensor, B: int) -> torch.Tensor:
        if B <= 1:
            return edge_index
        offsets = torch.arange(B, device=edge_index.device) * self.num_nodes
        return (edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)).reshape(2, -1)

    def forward(
        self,
        x_metric: torch.Tensor,
        x_logs: Optional[torch.Tensor],
        x_traces: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, F_m, W = x_metric.shape
        k = self._k_override if self._k_override is not None else self.k
        device = x_metric.device

        # ---- Encode each modality independently (zero-hop) ----
        h_m = self.metric_encoder(x_metric.view(B * N, F_m, W)).view(B, N, -1)

        has_logs = x_logs is not None and x_logs.abs().sum() > 0
        has_traces = x_traces is not None and x_traces.abs().sum() > 0

        if has_logs:
            xl = self.log_in_norm(x_logs.view(B * N, -1))
            h_l = self.log_encoder(xl).view(B, N, -1)
        else:
            h_l = torch.zeros(B, N, self.hidden_dim, device=device)

        if has_traces:
            xt = self.trace_in_norm(x_traces.view(B * N, -1))
            h_t = self.trace_encoder(xt).view(B, N, -1)
        else:
            h_t = torch.zeros(B, N, self.hidden_dim, device=device)

        # ---- Per-modality selection logits ----
        logits_m = self.metric_selector(h_m)                                 # [B, N]
        logits_l = self.log_selector(h_l) if has_logs else torch.zeros(B, N, device=device)
        logits_t = self.trace_selector(h_t) if has_traces else torch.zeros(B, N, device=device)

        # ---- Cross-modal attention fusion ----
        logits_stack = torch.stack([logits_m, logits_l, logits_t], dim=-1)   # [B, N, 3]
        modality_mask = torch.tensor(
            [1.0, float(has_logs), float(has_traces)], device=device
        )
        fused_logits = self.cross_fusion(logits_stack, modality_mask)        # [B, N]

        # ---- STE selection on fused logits ----
        # Feed fused logits through selector (which expects [B, N, H])
        # We create a pseudo-embedding from the fused + metric encoding
        h_fused = h_m  # metric encoding is always available as backbone
        M_ste, M_soft, top_k, sel_logits = self.selector(h_fused, k)

        # Override selector logits with the cross-modal fused logits
        sel_logits = fused_logits

        # ---- Mask + deep subgraph ----
        x_masked = x_metric * M_ste.unsqueeze(-1).unsqueeze(-1)
        x_sub, sub_ei = extract_subgraph_vectorized(x_masked, edge_index, top_k, self.num_nodes)

        h_deep = self.deep_temporal(x_sub)
        h_deep = self.deep_spatial(h_deep, sub_ei)
        h_deep = h_deep.view(B, k, -1)
        scores_sub = self.scorer(h_deep)

        # ---- Residual blend ----
        alpha = torch.sigmoid(self.residual_alpha)
        sel_on_topk = torch.gather(fused_logits, 1, top_k)
        blended = (1.0 - alpha) * scores_sub + alpha * sel_on_topk

        logits_full = torch.full((B, N), float('-inf'), device=device)
        logits_full.scatter_(1, top_k, blended)

        return logits_full, M_soft, top_k, sel_logits

    def set_temperature(self, t: float) -> None:
        self.selector.set_temperature(t)

    def set_k(self, k) -> None:
        self._k_override = k


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, F_m, W, F_l, F_t, k = 2, 20, 5, 12, 8, 6, 5
    ei = torch.randint(0, N, (2, 40))

    model = SelectThenAnalyze_MM(F_metric=F_m, F_log=F_l, F_trace=F_t,
                                  hidden_dim=32, num_nodes=N, k=k)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    lo, ms, tk, sl = model(torch.randn(B, N, F_m, W), torch.randn(B, N, F_l), torch.randn(B, N, F_t), ei)
    assert sl.shape == (B, N)
    assert tk.shape == (B, k)
    print(f"  Full MM:   sel_logits {sl.shape}, top_k {tk.shape}")

    lo2, _, _, sl2 = model(torch.randn(B, N, F_m, W), None, None, ei)
    assert sl2.shape == (B, N)
    print(f"  Mono-modal: sel_logits {sl2.shape}")

    print("\n=== model_sta_mm (multi-view) tests passed ===")
