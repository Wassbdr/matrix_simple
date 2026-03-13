"""
Spatio-Temporal Graph Neural Network with Graph Information Bottleneck
for Root Cause Analysis (RCA).

Architecture:
    1. Temporal Convolution (Conv1d) reduces the time window W into a
       fixed-length latent vector per node.
    2. Spatial Convolution (GCNConv x2) propagates information along
       the service dependency graph.
    3. Graph Information Bottleneck (MLP + relaxed Bernoulli) produces
       a differentiable sparse mask M over nodes, isolating the causal
       sub-graph.
    4. Output Z = M * H, where H is the spatial embedding.
    5. Root Cause Classifier maps GIB mask probabilities to a root cause
       node prediction (supervised fine-tuning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TemporalConvBlock(nn.Module):
    """
    Reduces the temporal dimension W of each node's feature vector using
    a 1D convolution followed by global average pooling.

    Input : [batch * num_nodes, num_features, W]
    Output: [batch * num_nodes, hidden_dim]
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        # Conv1d expects (N, C_in, L) -> (N, C_out, L')
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        # Learned temporal attention: preserves order information
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch * num_nodes, num_features, W]
        Returns:
            shape [batch * num_nodes, hidden_dim]
        """
        # shape: [batch*num_nodes, hidden_dim, W]
        h = F.relu(self.conv1(x))
        # shape: [batch*num_nodes, hidden_dim, W]
        h = F.relu(self.conv2(h))
        # Temporal attention: learned weighting preserves temporal order
        h_t = h.permute(0, 2, 1)  # [B*N, W, H]
        attn_scores = self.temporal_attn(h_t).squeeze(-1)  # [B*N, W]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*N, W]
        h = (h_t * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B*N, H]
        h = self.norm(h)
        return h


class SpatialConvBlock(nn.Module):
    """
    Two-layer GCN for message passing on the service dependency graph.

    Input : [num_nodes_total, hidden_dim] + edge_index
    Output: [num_nodes_total, hidden_dim]
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [num_nodes_total, hidden_dim]
            edge_index: shape [2, num_edges]
        Returns:
            shape [num_nodes_total, hidden_dim]
        """
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        # Residual connection preserves node identity through message passing
        h = self.norm(h + x)
        return h


class GraphInformationBottleneck(nn.Module):
    """
    MLP that predicts a per-node probability of belonging to the causal
    sub-graph. Uses the relaxed Bernoulli (Concrete distribution) for
    differentiable sampling of a binary mask M.

    Input : [num_nodes_total, hidden_dim]
    Output: (mask [num_nodes_total, 1], probabilities [num_nodes_total, 1])
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.5, mask_floor: float = 0.01) -> None:
        super().__init__()
        self.temperature_init = temperature
        self.temperature = temperature
        self.mask_floor = mask_floor
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_temperature(self, temperature: float) -> None:
        """Update the Concrete distribution temperature (for annealing)."""
        self.temperature = temperature

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: shape [num_nodes_total, hidden_dim]
        Returns:
            mask: shape [num_nodes_total, 1], differentiable binary mask
            probs: shape [num_nodes_total, 1], sigmoid probabilities
        """
        # shape: [num_nodes_total, 1]
        logits = self.mlp(h)
        # Sigmoid probabilities for the sparsity loss
        probs = torch.sigmoid(logits)

        if self.training:
            # Relaxed Bernoulli (Concrete distribution) reparameterization trick
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            mask = torch.sigmoid((gumbel_noise + logits) / self.temperature)
            mask = mask.clamp(min=self.mask_floor)
        else:
            # Soft deterministic mask at inference (matches training distribution)
            mask = probs

        return mask, probs


class NodeScorer(nn.Module):
    """
    Per-node scoring head for the No-GIB ablation.
    Applied independently to each node's spatial embedding.

    Input : h_spatial [batch, num_nodes, hidden_dim]
    Output: logits   [batch, num_nodes] (one score per candidate root cause)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: shape [batch, num_nodes, hidden_dim]
        Returns:
            logits: shape [batch, num_nodes]
        """
        return self.scorer(h).squeeze(-1)


class SpatioTemporalRCA(nn.Module):
    """
    Full model: Temporal Conv -> Spatial Conv -> GIB -> Z = M * H.

    Input:
        x     : [batch, num_nodes, num_features, W]
        edge_index : [2, num_edges] (static topology)
    Output:
        z_graph : [batch, hidden_dim]            (masked global pooled embedding)
        probs   : [batch, num_nodes, 1]          (mask probabilities = root cause scores)
        h_spatial_batched : [batch, num_nodes, hidden_dim] (raw spatial embeddings)
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 32,
        num_nodes: int = 5,
        temporal_kernel: int = 3,
        gib_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.temporal_block = TemporalConvBlock(num_features, hidden_dim, temporal_kernel)
        self.spatial_block = SpatialConvBlock(hidden_dim)
        self.gib = GraphInformationBottleneck(hidden_dim, gib_temperature)
        self.node_scorer = NodeScorer(hidden_dim)

        logger.info(
            "SpatioTemporalRCA initialized: features=%d, hidden=%d, nodes=%d",
            num_features,
            hidden_dim,
            num_nodes,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, use_gib: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape [batch, num_nodes, num_features, W]
            edge_index: shape [2, num_edges]
            use_gib: whether to apply the Graph Information Bottleneck masking
        Returns:
            z_graph: shape [batch, hidden_dim]  -- masked global pooled embedding
            probs:   shape [batch, num_nodes, 1] -- per-node mask probabilities
            h_spatial_batched: shape [batch, num_nodes, hidden_dim] -- raw embeddings
        """
        batch_size = x.size(0)

        # -- Temporal Convolution --
        x_flat = x.view(batch_size * self.num_nodes, x.size(2), x.size(3))
        h_temporal = self.temporal_block(x_flat)

        # -- Spatial Convolution (batched) --
        hidden_dim = h_temporal.size(1)
        if batch_size > 1:
            offsets = torch.arange(batch_size, device=h_temporal.device) * self.num_nodes
            batch_ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1) + offsets.view(-1, 1, 1)
            batch_ei = batch_ei.reshape(2, -1)
        else:
            batch_ei = edge_index
        h_spatial = self.spatial_block(h_temporal, batch_ei)

        # Reshape to [batch, num_nodes, hidden_dim]
        h_spatial_batched = h_spatial.view(batch_size, self.num_nodes, hidden_dim)

        # -- Graph Information Bottleneck --
        if use_gib:
            mask, probs = self.gib(h_spatial)
            mask = mask.view(batch_size, self.num_nodes, 1)
            probs = probs.view(batch_size, self.num_nodes, 1)
            # Masked Global Pooling: z_graph = sum(mask * h_spatial, dim=nodes)
            z_graph = (mask * h_spatial_batched).sum(dim=1)  # [batch, hidden_dim]
        else:
            probs = torch.ones(batch_size, self.num_nodes, 1, device=h_spatial.device) / self.num_nodes
            # Without GIB: simple mean pool over nodes
            z_graph = h_spatial_batched.mean(dim=1)  # [batch, hidden_dim]

        return z_graph, probs, h_spatial_batched

    def classify_root_cause(self, h_spatial_batched: torch.Tensor) -> torch.Tensor:
        """
        Per-node scoring for No-GIB ablation.
        Args:
            h_spatial_batched: shape [batch, num_nodes, hidden_dim]
        Returns:
            logits: shape [batch, num_nodes]
        """
        return self.node_scorer(h_spatial_batched)

    def gib_logits(self, h_spatial_batched: torch.Tensor) -> torch.Tensor:
        """
        Raw GIB MLP logits (before sigmoid) for N-way root cause classification.
        Use with F.cross_entropy for training and argmax for inference.

        Args:
            h_spatial_batched: shape [batch, num_nodes, hidden_dim]
        Returns:
            logits: shape [batch, num_nodes]
        """
        B, N, H = h_spatial_batched.shape
        return self.gib.mlp(h_spatial_batched.view(B * N, H)).squeeze(-1).view(B, N)
