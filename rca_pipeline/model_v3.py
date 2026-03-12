"""
Select-then-Analyze v3: Correct STE + Vectorized Subgraph Extraction.

Architecture (Early Bottleneck):
    1. LightEncoder: 1-layer GCN + temporal attention on full graph  -- O(N)
    2. STESelector: Gumbel-sigmoid + top-k binarization + STE        -- O(N)
    3. Vectorized subgraph extraction via pyg_subgraph               -- O(E)
    4. DeepSTGCN: 2-layer temporal + 2-layer GCN on subgraph         -- O(k^2)
    5. SubgraphScorer: context-aware node scoring                    -- O(k)

    Total: O(N + k^2) vs O(N^2) in the Late Bottleneck baseline.

Key innovations over v2:
    - Proper Straight-Through Estimator (STE) with Gumbel noise:
        M_hard = binarize(M_soft) - sg(M_soft) + M_soft
      Forward uses M_hard (discrete top-k), backward flows through M_soft.
    - Sigmoid mask (independent per-node) instead of softmax.
      Allows the sparsity loss ||M_soft||_1 to act as a direct L0 surrogate.
    - Single vectorized pyg_subgraph call for the entire batch
      (eliminates Python for-loops in _extract_subgraph).
    - Dual gradient path to the selector:
        (a) Direct: GIB sparsity loss + supervised CE on selector logits.
        (b) Indirect: task loss through STE mask -> Deep STGCN -> scorer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph as pyg_subgraph
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Lightweight Encoder (full graph, O(N))
# ---------------------------------------------------------------------------

class LightEncoder(nn.Module):
    """
    Two-layer GCN + temporal attention pooling.
    Runs on all N nodes -- must remain cheap since it sees the full graph.
    """

    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.temporal1 = nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1)
        self.temporal2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temp_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.spat_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_flat:     [B*N, F, W]
            edge_index: [2, E_batched]
        Returns:
            [B*N, H]
        """
        h = F.relu(self.temporal1(x_flat))                                  # [B*N, H, W]
        h = F.relu(self.temporal2(h))                                       # [B*N, H, W]
        h_t = h.permute(0, 2, 1)                                           # [B*N, W, H]
        attn = F.softmax(self.temp_attn(h_t).squeeze(-1), dim=-1)          # [B*N, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)                          # [B*N, H]
        h = self.norm(h)
        residual = h
        h = F.relu(self.gcn1(h, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        h = self.spat_norm(h + residual)
        return h


# ---------------------------------------------------------------------------
# Stage 2: Straight-Through Estimator Selector
# ---------------------------------------------------------------------------

class STESelector(nn.Module):
    """
    Differentiable top-k node selection with Straight-Through Estimator.

    Mathematical formulation:
        logits       = MLP(h)                               -- [B, N]
        M_soft       = sigmoid((logits + gumbel) / tau)      -- [B, N], continuous
        top_k_idx    = topk(M_soft, k)                       -- [B, k], discrete
        M_hard       = zeros.scatter(top_k_idx, 1.0)         -- [B, N], binary
        M_ste        = M_hard - sg(M_soft) + M_soft          -- STE bridge

    Forward value:  M_ste == M_hard  (binary, exactly k ones)
    Backward grad:  dM_ste/dM_soft == 1  (gradient flows through M_soft)

    Pitfalls addressed:
        - Gumbel noise prevents premature convergence of the selector;
          without it, the same k nodes are always selected and the
          gradient provides no information about unexplored nodes.
        - Temperature floor (0.1) avoids vanishing sigmoid gradients
          at low temperatures.
        - top_k_indices are sorted within each batch element to match
          the relabeling convention of pyg_subgraph (ascending order).
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_temperature(self, t: float) -> None:
        self.temperature = max(t, 0.1)

    def forward(
        self, h: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, N, H]
            k: number of nodes to select
        Returns:
            M_ste:          [B, N]  -- STE mask (binary forward, continuous backward)
            M_soft:         [B, N]  -- continuous sigmoid probabilities (for losses)
            top_k_indices:  [B, k]  -- selected node indices (sorted ascending)
            logits:         [B, N]  -- raw logits before sigmoid (for CE loss)
        """
        logits = self.mlp(h).squeeze(-1)                                    # [B, N]

        if self.training:
            # Gumbel noise for stochastic exploration
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            logits_noisy = logits + gumbel
        else:
            logits_noisy = logits

        M_soft = torch.sigmoid(logits_noisy / self.temperature)             # [B, N]

        # Discrete top-k selection
        _, top_k_indices = torch.topk(M_soft, k, dim=-1)                   # [B, k]
        top_k_indices, _ = top_k_indices.sort(dim=-1)                       # sort for pyg_subgraph

        M_hard = torch.zeros_like(M_soft)
        M_hard.scatter_(1, top_k_indices, 1.0)                             # [B, N] binary

        # --- Straight-Through Estimator ---
        # forward:  M_ste.data  == M_hard  (binary)
        # backward: dM_ste/dM_soft == 1    (gradient bypasses top-k)
        M_ste = M_hard - M_soft.detach() + M_soft

        return M_ste, M_soft, top_k_indices, logits


# ---------------------------------------------------------------------------
# Stage 3: Deep Temporal Convolution (runs on k nodes only)
# ---------------------------------------------------------------------------

class DeepTemporalBlock(nn.Module):
    """Two-layer Conv1d + learned temporal attention.  Input: [B*k, F, W]."""

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_t = h.permute(0, 2, 1)                                           # [B*k, W, H]
        attn = F.softmax(self.temporal_attn(h_t).squeeze(-1), dim=-1)       # [B*k, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)                           # [B*k, H]
        h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# Stage 3 (cont): Deep Spatial Convolution (runs on k-node subgraph)
# ---------------------------------------------------------------------------

class DeepSpatialBlock(nn.Module):
    """Two-layer GCN + residual on the selected subgraph."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = h
        h = F.relu(self.gcn1(h, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        h = self.norm(h + residual)
        return h


# ---------------------------------------------------------------------------
# Stage 4: Per-node Scorer with subgraph context
# ---------------------------------------------------------------------------

class SubgraphScorer(nn.Module):
    """Context-aware scoring: [node_emb || attention_context] -> scalar."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, k, H] -> scores: [B, k]."""
        attn = F.softmax(self.context_attn(h).squeeze(-1), dim=-1)         # [B, k]
        context = (h * attn.unsqueeze(-1)).sum(dim=1, keepdim=True)         # [B, 1, H]
        context = context.expand_as(h)                                      # [B, k, H]
        h_cat = torch.cat([h, context], dim=-1)                             # [B, k, 2H]
        return self.scorer(h_cat).squeeze(-1)                               # [B, k]


# ---------------------------------------------------------------------------
# Vectorized subgraph extraction utility
# ---------------------------------------------------------------------------

def extract_subgraph_vectorized(
    x_masked: torch.Tensor,
    edge_index: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-call vectorized subgraph extraction for an entire batch.

    Instead of looping over batch elements in Python (O(B*E) in Python,
    the bottleneck in model_v2), this function:
        1. Builds a batched super-graph (standard PyG batch convention).
        2. Computes a global subset of selected nodes (B*k indices).
        3. Calls torch_geometric.utils.subgraph ONCE with relabel_nodes=True.

    The relabeling assigns indices [0..B*k-1] in ascending order of original
    (batched) node indices, which matches the ascending-sorted top_k_indices.

    Pitfalls addressed:
        - top_k_indices MUST be sorted within each batch element; otherwise
          the node->feature alignment is broken after relabeling.
        - For disconnected subgraphs (no edges between selected nodes),
          GCNConv automatically adds self-loops, so the model degrades
          gracefully to an MLP-like behavior.

    Args:
        x_masked:       [B, N, F, W] features gated by STE mask
        edge_index:     [2, E] base topology (same for all batch elements)
        top_k_indices:  [B, k] selected nodes (sorted ascending within each row)
        num_nodes:      N
    Returns:
        x_sub:          [B*k, F, W]
        sub_edge_index: [2, E_sub]  relabeled to [0..B*k-1]
    """
    B = x_masked.size(0)
    N = num_nodes
    k = top_k_indices.size(1)
    device = x_masked.device

    # Flatten: [B*N, F, W]
    x_flat = x_masked.reshape(B * N, x_masked.size(2), x_masked.size(3))

    # Batch offsets
    offsets = torch.arange(B, device=device) * N                            # [B]

    # Batched edge_index (standard PyG convention)
    if B > 1:
        batched_ei = (
            edge_index.unsqueeze(0).expand(B, -1, -1)
            + offsets.view(-1, 1, 1)
        ).reshape(2, -1)                                                    # [2, B*E]
    else:
        batched_ei = edge_index

    # Global indices of selected nodes
    batched_subset = (top_k_indices + offsets.unsqueeze(1)).reshape(-1)      # [B*k]

    # Single vectorized subgraph extraction
    sub_ei, _ = pyg_subgraph(
        batched_subset,
        batched_ei,
        relabel_nodes=True,
        num_nodes=B * N,
    )

    # Gather features
    x_sub = x_flat[batched_subset]                                          # [B*k, F, W]

    return x_sub, sub_ei


# ---------------------------------------------------------------------------
# Main Architecture: Select-then-Analyze v3
# ---------------------------------------------------------------------------

class SelectThenAnalyzeV3(nn.Module):
    """
    Full pipeline:
        x [B, N, F, W]  -->  LightEncoder  -->  STESelector  -->  subgraph
        subgraph  -->  DeepTemporal + DeepSpatial  -->  Scorer  -->  logits [B, N]

    Returns four tensors consumed by different loss terms:
        logits_full:      [B, N]   scorer output (for supervised CE, only k finite)
        M_soft:           [B, N]   continuous mask (for L1 sparsity / GIB loss)
        top_k_indices:    [B, k]   selected nodes  (for selection-rate tracking)
        selector_logits:  [B, N]   raw selector logits (for supervised CE on all N)
    """

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 32,
        num_nodes: int = 20,
        k: int = 5,
        temporal_kernel: int = 3,
        gib_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.k = k

        self.light_encoder = LightEncoder(num_features, hidden_dim)
        self.selector = STESelector(hidden_dim, gib_temperature)
        self.deep_temporal = DeepTemporalBlock(num_features, hidden_dim, temporal_kernel)
        self.deep_spatial = DeepSpatialBlock(hidden_dim)
        self.scorer = SubgraphScorer(hidden_dim)
        self._k_override = None  # for k-annealing during training

        logger.info(
            "SelectThenAnalyzeV3: N=%d, k=%d, features=%d, hidden=%d",
            num_nodes, k, num_features, hidden_dim,
        )

    # -- helpers --

    def _batch_edge_index(self, edge_index: torch.Tensor, B: int) -> torch.Tensor:
        if B <= 1:
            return edge_index
        device = edge_index.device
        offsets = torch.arange(B, device=device) * self.num_nodes
        return (
            edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
        ).reshape(2, -1)

    # -- forward --

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, F_, W = x.shape
        k = self._k_override if self._k_override is not None else self.k
        device = x.device

        # ---- Stage 1: Light encoder (full graph) ----
        x_flat = x.view(B * N, F_, W)
        batch_ei = self._batch_edge_index(edge_index, B)
        h_light = self.light_encoder(x_flat, batch_ei)                      # [B*N, H]
        h_light_batched = h_light.view(B, N, -1)                            # [B, N, H]

        # ---- Stage 2: STE selection ----
        M_ste, M_soft, top_k_indices, selector_logits = self.selector(
            h_light_batched, k
        )

        # ---- Stage 3: Mask features and extract subgraph ----
        # Gate features by STE mask: creates gradient path selector <- task loss
        x_masked = x * M_ste.unsqueeze(-1).unsqueeze(-1)                    # [B, N, F, W]

        x_sub, sub_ei = extract_subgraph_vectorized(
            x_masked, edge_index, top_k_indices, self.num_nodes,
        )                                                                    # [B*k, F, W], [2, E_sub]

        # ---- Stage 4: Deep STGCN on subgraph (O(k^2)) ----
        h_deep = self.deep_temporal(x_sub)                                   # [B*k, H]
        h_deep = self.deep_spatial(h_deep, sub_ei)                           # [B*k, H]
        h_deep_batched = h_deep.view(B, k, -1)                              # [B, k, H]

        # ---- Stage 5: Score and scatter back ----
        scores_sub = self.scorer(h_deep_batched)                             # [B, k]

        logits_full = torch.full((B, N), float('-inf'), device=device)
        logits_full.scatter_(1, top_k_indices, scores_sub)

        return logits_full, M_soft, top_k_indices, selector_logits

    def set_temperature(self, t: float) -> None:
        self.selector.set_temperature(t)

    def set_k(self, k: int) -> None:
        """Override k for k-annealing. Pass None to reset to default."""
        self._k_override = k
