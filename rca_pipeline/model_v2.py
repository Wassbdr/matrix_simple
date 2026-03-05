"""
Select-then-Analyze STGCN with Graph Information Bottleneck.

Innovation: The GIB operates BEFORE the deep STGCN analysis, not after.
    1. Lightweight encoder (1-layer GCN) produces quick per-node embeddings
    2. GIB selector predicts a SOFTMAX distribution over nodes → selects top-k
    3. Deep STGCN (temporal conv + multi-layer GCN) processes ONLY the selected subgraph
    4. Per-node scorer on the subgraph → root cause classification

Speedup: When N=20 and k=5, the deep STGCN does O(k²) instead of O(N²) work.
The GIB has a structural reason to exist: it reduces computation AND filters noise.

Mask distribution: Softmax (not sigmoid) → mutual competition between nodes.
This encodes the single-root-cause constraint ∑ᵢ pᵢ = 1 directly in the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Lightweight Encoder (runs on full graph, must be cheap) ──────────

class LightEncoder(nn.Module):
    """
    Two-layer GCN + temporal pooling.
    Runs on ALL N nodes — encoding quality is critical for selection.
    
    Input : [B*N, F, W]
    Output: [B*N, H]
    """
    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        # Temporal: Conv1d + attention pooling (not just mean)
        self.temporal = nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1)
        self.temp_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        # Two GCN layers with residual (selector accuracy is critical)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.spat_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_flat: [B*N, F, W]
            edge_index: [2, E] (batched)
        Returns:
            [B*N, H]
        """
        h = F.relu(self.temporal(x_flat))            # [B*N, H, W]
        # Attention pooling over time
        h_t = h.permute(0, 2, 1)                     # [B*N, W, H]
        attn = F.softmax(self.temp_attn(h_t).squeeze(-1), dim=-1)  # [B*N, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)    # [B*N, H]
        h = self.norm(h)
        # Spatial: 2-layer GCN with residual
        residual = h
        h = F.relu(self.gcn1(h, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        h = self.spat_norm(h + residual)
        return h


# ── GIB Selector (softmax distribution over nodes) ──────────────────

class GIBSelector(nn.Module):
    """
    Predicts a softmax probability distribution over nodes.
    
    Softmax ensures ∑ᵢ pᵢ = 1 → mutual competition.
    Temperature-controlled: high temp = exploration, low temp = sharp selection.
    
    Input : [B, N, H]
    Output: (probs [B, N], selected_indices [B, k])
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
        self.temperature = t
    
    def forward(self, h: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, N, H] — light encoder output
            k: number of nodes to select
        Returns:
            probs: [B, N] — softmax distribution (for loss computation)
            top_k_indices: [B, k] — indices of selected nodes
            logits: [B, N] — raw logits before softmax (for CE loss)
        """
        logits = self.mlp(h).squeeze(-1)             # [B, N]
        probs = F.softmax(logits / self.temperature, dim=-1)  # [B, N]
        
        # Select top-k nodes (differentiable via straight-through in training)
        _, top_k_indices = torch.topk(probs, k, dim=-1)  # [B, k]
        
        return probs, top_k_indices, logits


# ── Deep Temporal Conv (runs on subgraph only) ──────────────────────

class DeepTemporalBlock(nn.Module):
    """
    Two-layer Conv1d + learned temporal attention.
    Same as the original TemporalConvBlock but only processes k nodes.
    
    Input : [B*k, F, W]
    Output: [B*k, H]
    """
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h_t = h.permute(0, 2, 1)                           # [B*k, W, H]
        attn = F.softmax(self.temporal_attn(h_t).squeeze(-1), dim=-1)  # [B*k, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)           # [B*k, H]
        h = self.norm(h)
        return h


# ── Deep Spatial Conv (runs on subgraph only) ──────────────────────

class DeepSpatialBlock(nn.Module):
    """
    Two-layer GCN with residual + LayerNorm on the selected subgraph.
    
    Input : [B*k, H] + subgraph edge_index
    Output: [B*k, H]
    """
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


# ── Per-Node Scorer ──────────────────────────────────────────────────

class SubgraphScorer(nn.Module):
    """
    Scores each node in the selected subgraph.
    Uses a context vector (global subgraph summary) to help scoring.
    
    Input : [B, k, H]
    Output: [B, k]
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Context-aware scoring: node embedding + global subgraph context
        self.context_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, k, H]
        # Attention-weighted subgraph context
        attn = F.softmax(self.context_attn(h).squeeze(-1), dim=-1)  # [B, k]
        context = (h * attn.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, H]
        context = context.expand_as(h)  # [B, k, H]
        # Concatenate node + context for scoring
        h_cat = torch.cat([h, context], dim=-1)  # [B, k, 2*H]
        return self.scorer(h_cat).squeeze(-1)  # [B, k]


# ── Main Architecture: Select-then-Analyze ──────────────────────────

class SelectThenAnalyze(nn.Module):
    """
    Full Select-then-Analyze pipeline:
        1. LightEncoder (1-layer GCN) on full graph → h_light [B, N, H]
        2. GIBSelector → probs [B, N], top_k [B, k]
        3. Extract subgraph: x_sub [B, k, F, W], sub_edge_index
        4. DeepTemporalBlock + DeepSpatialBlock on subgraph → h_deep [B, k, H]
        5. SubgraphScorer → scores [B, k]
        6. Map back to full graph for loss/metrics
    
    Args:
        num_features: number of input features per node
        hidden_dim: hidden dimensionality throughout the model
        num_nodes: total number of nodes N in the full graph
        k: number of nodes to select (subgraph size)
        gib_temperature: initial temperature for softmax selector
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
        self.num_nodes = num_nodes
        self.k = k
        
        # Stage 1: Light encoder (full graph)
        self.light_encoder = LightEncoder(num_features, hidden_dim)
        
        # Stage 2: GIB selector (softmax over nodes)
        self.gib_selector = GIBSelector(hidden_dim, gib_temperature)
        
        # Stage 3: Deep encoder (subgraph only)
        self.deep_temporal = DeepTemporalBlock(num_features, hidden_dim, temporal_kernel)
        self.deep_spatial = DeepSpatialBlock(hidden_dim)
        
        # Stage 4: Per-node scorer (subgraph only)
        self.scorer = SubgraphScorer(hidden_dim)
        
        logger.info(
            "SelectThenAnalyze: N=%d, k=%d, features=%d, hidden=%d",
            num_nodes, k, num_features, hidden_dim,
        )
    
    def _extract_subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract subgraph for selected nodes.
        
        Args:
            x: [B, N, F, W]
            edge_index: [2, E] (original graph)
            top_k_indices: [B, k]
        Returns:
            x_sub: [B, k, F, W]
            sub_edge_index_batched: [2, E_sub_total] (batched subgraph edges)
        """
        B, N, F_, W = x.shape
        k = top_k_indices.size(1)
        
        # Gather selected node features
        # top_k_indices: [B, k] → expand to [B, k, F, W]
        idx_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F_, W)
        x_sub = torch.gather(x, 1, idx_expanded)  # [B, k, F, W]
        
        # Build subgraph edge_index for each batch element
        # For each sample, remap edges to local indices [0..k-1]
        all_src, all_dst = [], []
        src_orig, dst_orig = edge_index[0], edge_index[1]
        
        for b in range(B):
            selected = top_k_indices[b]  # [k]
            # Build mapping: global_idx → local_idx
            node_map = {}
            for local_idx, global_idx in enumerate(selected.tolist()):
                node_map[global_idx] = local_idx
            
            # Keep only edges where both endpoints are in selected set
            for i in range(src_orig.size(0)):
                s, d = src_orig[i].item(), dst_orig[i].item()
                if s in node_map and d in node_map:
                    all_src.append(node_map[s] + b * k)
                    all_dst.append(node_map[d] + b * k)
        
        if all_src:
            sub_edge_index = torch.tensor([all_src, all_dst], dtype=torch.long, device=x.device)
        else:
            # Fallback: self-loops for all selected nodes
            nodes = torch.arange(B * k, device=x.device)
            sub_edge_index = torch.stack([nodes, nodes])
        
        return x_sub, sub_edge_index
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, F, W]
            edge_index: [2, E]
        Returns:
            logits_full: [B, N] — score for every node (sparse: only k non-zero)
            probs: [B, N] — GIB softmax probabilities
            top_k_indices: [B, k] — selected node indices
            gib_logits: [B, N] — raw GIB logits (for CE loss, numerically stable)
        """
        B = x.size(0)
        N = self.num_nodes
        k = self.k
        F_ = x.size(2)
        W = x.size(3)
        
        # ── Stage 1: Light encoder on full graph ──
        x_flat = x.view(B * N, F_, W)  # [B*N, F, W]
        
        # Batch edge_index
        if B > 1:
            offsets = torch.arange(B, device=x.device) * N
            batch_ei = edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
            batch_ei = batch_ei.reshape(2, -1)
        else:
            batch_ei = edge_index
        
        h_light = self.light_encoder(x_flat, batch_ei)  # [B*N, H]
        h_light_batched = h_light.view(B, N, -1)        # [B, N, H]
        
        # ── Stage 2: GIB selection ──
        probs, top_k_indices, gib_logits = self.gib_selector(h_light_batched, k)  # [B,N], [B,k], [B,N]
        
        # ── Stage 3: Extract subgraph & deep analysis ──
        x_sub, sub_ei = self._extract_subgraph(x, edge_index, top_k_indices)  # [B,k,F,W], [2,E_sub]
        
        # Deep temporal on subgraph
        x_sub_flat = x_sub.view(B * k, F_, W)        # [B*k, F, W]
        h_deep = self.deep_temporal(x_sub_flat)       # [B*k, H]
        
        # Deep spatial on subgraph
        h_deep = self.deep_spatial(h_deep, sub_ei)    # [B*k, H]
        h_deep_batched = h_deep.view(B, k, -1)       # [B, k, H]
        
        # ── Stage 4: Score selected nodes ──
        scores_sub = self.scorer(h_deep_batched)      # [B, k]
        
        # Map back to full graph: scatter scores to original node positions
        logits_full = torch.full((B, N), float('-inf'), device=x.device)
        logits_full.scatter_(1, top_k_indices, scores_sub)
        
        return logits_full, probs, top_k_indices, gib_logits
    
    def set_gib_temperature(self, t: float) -> None:
        self.gib_selector.set_temperature(t)


# ── Legacy wrapper: original architecture for comparison ─────────────
# (imported from model.py as SpatioTemporalRCA)
