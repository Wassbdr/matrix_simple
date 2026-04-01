"""
Late Bottleneck with GATv2 Attention (model_late_gat.py).

Replaces the Graph Information Bottleneck (GIB) binary mask with a soft
GATv2 attention layer (Brody et al., 2022). Node importance is derived from
the sum of incoming attention weights rather than a sampled Bernoulli mask.

Why GATv2 over GIB at N=100:
    The GIB binary mask forces a fixed sparsity (1 causal node out of N).
    With 100 root-cause classes the causal sub-graph spans multiple hops,
    making a single-node constraint detrimental (ablation: +/-9% variance).
    GATv2 lets the model learn a soft, differentiable importance distribution
    without any L1/L0 sparsity constraint — the network learns which nodes
    matter contextually. This is the approach of DynaCausal (ICSE 2025).

Design Decision — Why no GIB in GATv2:
    GIB (Graph Information Bottleneck) uses a Bernoulli mask to enforce
    explicit sparsity on node representations. GATv2's attention mechanism
    already provides implicit sparsity through learned attention weights.
    Stacking GIB on top of GATv2 introduces conflicting inductive biases
    (hard masking vs soft attention) and tends to hurt optimization in this
    setting, so this variant keeps pure GATv2 attention only.

Architecture:
    1. TemporalConvBlock  (same as SpatioTemporalRCA)
    2. GATv2Conv x2       (replace GCNConv + GIB with soft attention)
    3. Node importance    = sum of incoming attention weights per node
    4. NodeScorer         (same MLP head)
    5. CE loss on node scores (no sparsity loss needed)

Inference:
    - model.forward() returns logits [B, N] and attn_weights [E] for interpretability.
    - The predicted root cause is argmax(logits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal block (identical to model.py for fair comparison)
# ---------------------------------------------------------------------------

class TemporalConvBlock(nn.Module):
    """Conv1d temporal encoder with learned attention pooling. Input: [B*N, F, W]."""

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_t = h.permute(0, 2, 1)                                     # [B*N, W, H]
        attn = F.softmax(self.temporal_attn(h_t).squeeze(-1), dim=-1)  # [B*N, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)                    # [B*N, H]
        h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# GATv2 spatial encoder (replaces GCNConv + GIB)
# ---------------------------------------------------------------------------

class GATv2SpatialBlock(nn.Module):
    """
    Two-layer GATv2 with multi-head attention.
    Returns node embeddings AND per-edge attention weights for interpretability.
    """

    def __init__(self, hidden_dim: int, heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=0.1,
            share_weights=False,
            add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=0.1,
            share_weights=False,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:          [total_nodes, hidden_dim]
            edge_index: [2, E]
        Returns:
            h_out:       [total_nodes, hidden_dim]
            attn_w:      [E] — average attention weight per edge (for interpretability)
        """
        residual = h
        h1, a1 = self.gat1(h, edge_index, return_attention_weights=True)
        h1 = self.norm1(F.elu(h1) + residual)

        h2, a2 = self.gat2(h1, edge_index, return_attention_weights=True)
        h2 = self.norm2(F.elu(h2) + h1)

        # Average attention across heads for edge importance
        attn_w = a1[1].mean(dim=-1)  # [E]
        return h2, attn_w


# ---------------------------------------------------------------------------
# Node importance from attention (no GIB mask needed)
# ---------------------------------------------------------------------------

def compute_node_importance(
    edge_index: torch.Tensor,
    attn_weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Node importance = sum of incoming attention weights.
    Nodes that many other nodes attend to strongly are likely infrastructure
    (i.e., root causes in a dependency cascade).

    Args:
        edge_index:   [2, E] — edge_index[1] are destination nodes
        attn_weights: [E]    — per-edge attention weights
        num_nodes:    N
    Returns:
        importance: [N] — aggregated incoming attention per node
    """
    importance = torch.zeros(num_nodes, device=attn_weights.device)
    dst = edge_index[1]
    importance.scatter_add_(0, dst, attn_weights.abs())
    return importance


# ---------------------------------------------------------------------------
# Full model: Late Bottleneck with GATv2
# ---------------------------------------------------------------------------

class SpatioTemporalRCA_GAT(nn.Module):
    """
    Late Bottleneck with GATv2 soft attention.

    Forward output:
        logits:        [B, N]   root cause node scores (for CE loss)
        node_imp:      [B, N]   soft importance from attention (interpretability)
        h_spatial:     [B, N, H] raw spatial embeddings
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 32,
        num_nodes: int = 20,
        temporal_kernel: int = 3,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.temporal_block = TemporalConvBlock(num_features, hidden_dim, temporal_kernel)
        self.spatial_block = GATv2SpatialBlock(hidden_dim, heads=gat_heads)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        logger.info(
            "SpatioTemporalRCA_GAT: features=%d, hidden=%d, nodes=%d, heads=%d",
            num_features, hidden_dim, num_nodes, gat_heads,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [B, N, F, W]
            edge_index: [2, E]
        Returns:
            logits:    [B, N]
            node_imp:  [B, N]
            h_spatial: [B, N, H]
        """
        B, N, F_, W = x.shape
        x_flat = x.view(B * N, F_, W)
        h_temp = self.temporal_block(x_flat)                          # [B*N, H]

        # Add self-loops to edge_index before batching (GATv2 add_self_loops=False)
        self_loops = torch.arange(N, device=x.device)
        ei_with_sl = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)

        if B > 1:
            offsets = torch.arange(B, device=x.device) * N
            batch_ei = (
                ei_with_sl.unsqueeze(0).expand(B, -1, -1)
                + offsets.view(-1, 1, 1)
            ).reshape(2, -1)
        else:
            batch_ei = ei_with_sl

        h_spat, attn_w = self.spatial_block(h_temp, batch_ei)        # [B*N, H], [B*E]

        # Node importance from attention (batch-level)
        node_imp_flat = compute_node_importance(batch_ei, attn_w, B * N)  # [B*N]

        # Reshape to batch
        h_spat_bat = h_spat.view(B, N, -1)                           # [B, N, H]
        node_imp_bat = node_imp_flat.view(B, N)                       # [B, N]

        # Logits: combine scorer and attention importance
        logits = self.scorer(h_spat_bat).squeeze(-1)                  # [B, N]
        # Soft boost from attention (no hard masking, just additive signal)
        logits = logits + 0.3 * node_imp_bat

        return logits, node_imp_bat, h_spat_bat

    def classify_root_cause(self, h_spatial: torch.Tensor) -> torch.Tensor:
        """Alias for compatibility. Returns scorer logits [B, N]."""
        return self.scorer(h_spatial).squeeze(-1)
