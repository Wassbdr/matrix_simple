"""
model_late_gat_mm.py — Multimodal Late Bottleneck with GATv2 (variant _mm).

Late fusion architecture inspired by Chain-of-Event (FSE 2024) and DynaCausal (ICSE 2025):
    MetricEncoder  [B, N, F_m, W] → TemporalConvBlock → H_m    [B*N, H]
    LogEncoder     [B, N, F_l]    → Linear + LayerNorm → H_l    [B*N, H]
    TraceEncoder   [B, N, F_t]    → Linear + LayerNorm → H_t    [B*N, H]
    FusionMLP      concat([H_m; H_l; H_t]) → H_fused            [B*N, H]
    GATv2x2        H_fused + edge_index → H_graph                [B*N, H]
    NodeScorer     H_graph → [B, N]

Compatibility: if x_logs=None or x_traces=None → zero-pad (mono-modal fallback).

This is a SEPARATE variant — does not modify model_late_gat.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional, Tuple
import logging

from model_late_gat import TemporalConvBlock, GATv2SpatialBlock, compute_node_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

F_LOG_DEFAULT = 8
F_TRACE_DEFAULT = 6


class ModalityEncoder(nn.Module):
    """Lightweight encoder for log or trace features: Linear → LayerNorm → ReLU."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionMLP(nn.Module):
    """Fuse 2-3 modality embeddings via concatenation + MLP projection."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatioTemporalRCA_GAT_MM(nn.Module):
    """
    Multimodal Late Bottleneck with GATv2.

    forward(x_metric, x_logs, x_traces, edge_index):
        x_metric:  [B, N, F_m, W]  — required
        x_logs:    [B, N, F_l] or None  — optional
        x_traces:  [B, N, F_t] or None  — optional
    Returns:
        logits:    [B, N]
        node_imp:  [B, N]
        h_spatial: [B, N, H]
    """

    def __init__(
        self,
        F_metric: int = 5,
        F_log: int = F_LOG_DEFAULT,
        F_trace: int = F_TRACE_DEFAULT,
        hidden_dim: int = 32,
        num_nodes: int = 20,
        temporal_kernel: int = 3,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.F_log = F_log
        self.F_trace = F_trace

        self.metric_encoder = TemporalConvBlock(F_metric, hidden_dim, temporal_kernel)
        # Normalize raw log/trace feature vectors to stabilize fusion.
        self.log_in_norm = nn.LayerNorm(F_log)
        self.trace_in_norm = nn.LayerNorm(F_trace)
        self.log_encoder = ModalityEncoder(F_log, hidden_dim)
        self.trace_encoder = ModalityEncoder(F_trace, hidden_dim)
        # Conditional modality gating (per node).
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            if isinstance(self.gate_net[-1], nn.Linear):
                self.gate_net[-1].bias[:] = torch.tensor([-1.0, 0.0])
        self.fusion = FusionMLP(hidden_dim * 3, hidden_dim)

        self.spatial_block = GATv2SpatialBlock(hidden_dim, heads=gat_heads)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1),
        )

        logger.info(
            "SpatioTemporalRCA_GAT_MM: F_m=%d, F_l=%d, F_t=%d, H=%d, N=%d",
            F_metric, F_log, F_trace, hidden_dim, num_nodes,
        )

    def forward(
        self,
        x_metric: torch.Tensor,
        x_logs: Optional[torch.Tensor],
        x_traces: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, F_m, W = x_metric.shape
        device = x_metric.device

        # Metric encoder
        h_m = self.metric_encoder(x_metric.view(B * N, F_m, W))  # [B*N, H]

        # Log encoder (zero-pad fallback)
        if x_logs is not None:
            xl = self.log_in_norm(x_logs.view(B * N, -1))
            h_l = self.log_encoder(xl)
        else:
            h_l = torch.zeros(B * N, self.hidden_dim, device=device)

        # Trace encoder (zero-pad fallback)
        if x_traces is not None:
            xt = self.trace_in_norm(x_traces.view(B * N, -1))
            h_t = self.trace_encoder(xt)
        else:
            h_t = torch.zeros(B * N, self.hidden_dim, device=device)

        gates = torch.sigmoid(self.gate_net(torch.cat([h_m, h_l, h_t], dim=-1)))  # [B*N, 2]
        g_l = gates[:, 0:1]
        g_t = gates[:, 1:2]
        if x_logs is None:
            g_l = torch.zeros_like(g_l)
        if x_traces is None:
            g_t = torch.zeros_like(g_t)
        h_l = h_l * g_l
        h_t = h_t * g_t

        # Late fusion
        h_fused = self.fusion(torch.cat([h_m, h_l, h_t], dim=-1))  # [B*N, H]

        # Batched edge index
        if B > 1:
            offsets = torch.arange(B, device=device) * N
            batch_ei = (edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)).reshape(2, -1)
        else:
            batch_ei = edge_index

        h_spat, attn_w = self.spatial_block(h_fused, batch_ei)
        node_imp = compute_node_importance(batch_ei, attn_w, B * N).view(B, N)
        h_out = h_spat.view(B, N, -1)

        logits = self.scorer(h_out).squeeze(-1) + 0.3 * node_imp
        return logits, node_imp, h_out


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, F_m, W, F_l, F_t = 2, 20, 5, 12, 8, 6
    ei = torch.randint(0, N, (2, 40))

    model = SpatioTemporalRCA_GAT_MM(F_metric=F_m, F_log=F_l, F_trace=F_t, hidden_dim=32, num_nodes=N)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Full multimodal
    r1, imp1, _ = model(torch.randn(B, N, F_m, W), torch.randn(B, N, F_l), torch.randn(B, N, F_t), ei)
    assert r1.shape == (B, N), f"expected ({B},{N}), got {r1.shape}"
    print(f"  Full MM:   logits {r1.shape}")

    # Metric-only fallback
    r2, _, _ = model(torch.randn(B, N, F_m, W), None, None, ei)
    assert r2.shape == (B, N)
    print(f"  Mono-modal: logits {r2.shape}")

    # Partial: metric + logs
    r3, _, _ = model(torch.randn(B, N, F_m, W), torch.randn(B, N, F_l), None, ei)
    assert r3.shape == (B, N)
    print(f"  Metric+Log: logits {r3.shape}")

    print("\n=== model_late_gat_mm tests passed ===")
