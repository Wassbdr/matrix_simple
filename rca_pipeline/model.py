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
        # Global average pooling over the temporal dimension
        # shape: [batch*num_nodes, hidden_dim]
        h = h.mean(dim=-1)
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
        # shape: [num_nodes_total, hidden_dim]
        h = F.relu(self.gcn1(x, edge_index))
        # shape: [num_nodes_total, hidden_dim]
        h = self.gcn2(h, edge_index)
        h = self.norm(h)
        return h


class GraphInformationBottleneck(nn.Module):
    """
    MLP that predicts a per-node probability of belonging to the causal
    sub-graph. Uses the relaxed Bernoulli (Concrete distribution) for
    differentiable sampling of a binary mask M.

    Input : [num_nodes_total, hidden_dim]
    Output: (mask [num_nodes_total, 1], probabilities [num_nodes_total, 1])
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

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
            # Sample u ~ Uniform(0, 1)
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            # Concrete / Gumbel-Sigmoid: sigma((log(u) - log(1-u) + logits) / tau)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            mask = torch.sigmoid((gumbel_noise + logits) / self.temperature)
        else:
            # Hard threshold at inference
            mask = (probs > 0.5).float()

        return mask, probs


class RootCauseClassifier(nn.Module):
    """
    Supervised classification head that maps node-level hidden representations
    to a root cause prediction. Uses the GIB-masked spatial embeddings
    to predict which node is the root cause.

    Input : z [batch, num_nodes, hidden_dim] (GIB-masked embeddings)
    Output: logits [batch, num_nodes] (one logit per node = per possible root cause)
    """

    def __init__(self, hidden_dim: int, num_nodes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_nodes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: shape [batch, num_nodes, hidden_dim]
        Returns:
            logits: shape [batch, num_nodes]
        """
        batch_size = z.size(0)
        z_flat = z.view(batch_size, -1)  # [batch, num_nodes * hidden_dim]
        return self.classifier(z_flat)


class SpatioTemporalRCA(nn.Module):
    """
    Full model: Temporal Conv -> Spatial Conv -> GIB -> Z = M * H.

    Input:
        x     : [batch, num_nodes, num_features, W]
        edge_index : [2, num_edges] (static topology)
    Output:
        z     : [batch, num_nodes, hidden_dim]  (compressed representation)
        probs : [batch, num_nodes, 1]           (mask probabilities for GIB loss)
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
        self.root_cause_head = RootCauseClassifier(hidden_dim, num_nodes)

        logger.info(
            "SpatioTemporalRCA initialized: features=%d, hidden=%d, nodes=%d",
            num_features,
            hidden_dim,
            num_nodes,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape [batch, num_nodes, num_features, W]
            edge_index: shape [2, num_edges]
        Returns:
            z: shape [batch, num_nodes, hidden_dim]
            mask_probs: shape [batch, num_nodes, 1]
        """
        batch_size = x.size(0)

        # -- Temporal Convolution --
        # Reshape to [batch * num_nodes, num_features, W] for Conv1d
        x_flat = x.view(batch_size * self.num_nodes, x.size(2), x.size(3))
        # shape: [batch * num_nodes, hidden_dim]
        h_temporal = self.temporal_block(x_flat)

        # -- Spatial Convolution --
        # GCNConv expects a single flat node list with a matching edge_index.
        # We replicate edge_index for each sample in the batch by offsetting
        # node indices.
        h_all = []
        hidden_dim = h_temporal.size(1)
        for i in range(batch_size):
            # Extract per-sample node features: [num_nodes, hidden_dim]
            h_i = h_temporal[i * self.num_nodes : (i + 1) * self.num_nodes]
            # Apply GCN on the shared static topology
            h_i = self.spatial_block(h_i, edge_index)
            h_all.append(h_i)

        # shape: [batch * num_nodes, hidden_dim]
        h_spatial = torch.cat(h_all, dim=0)

        # -- Graph Information Bottleneck --
        # shape: mask [batch*num_nodes, 1], probs [batch*num_nodes, 1]
        mask, probs = self.gib(h_spatial)

        # Compressed representation: Z = M (element-wise) H
        # shape: [batch * num_nodes, hidden_dim]
        z = mask * h_spatial

        # Reshape back to [batch, num_nodes, ...]
        z = z.view(batch_size, self.num_nodes, hidden_dim)
        probs = probs.view(batch_size, self.num_nodes, 1)

        return z, probs

    def classify_root_cause(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predicts root cause node from GIB-masked embeddings.

        Args:
            z: shape [batch, num_nodes, hidden_dim]
        Returns:
            logits: shape [batch, num_nodes]
        """
        return self.root_cause_head(z)
