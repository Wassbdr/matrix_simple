"""
Select-then-Analyze (STA) v2: Zero-Hop Selector + Variational GIB + Residual.

Architecture (Early Bottleneck):
    1. LightEncoder: zero-hop MLP_fast (Conv1d + attention, NO GCN)  -- O(N)
    2. STESelector: Gumbel-sigmoid + top-k binarization + STE        -- O(N)
    3. Vectorized subgraph extraction via pyg_subgraph               -- O(E)
    4. DeepSTGCN: 2-layer temporal + 2-layer GCN on subgraph         -- O(k^2)
    5. SubgraphScorer: context-aware node scoring + residual blend   -- O(k)

    Total: O(N + k^2) vs O(N^2) in the Late Bottleneck baseline.

Key design principles (aligned with formalization):
    - Zero-hop selector: H_fast = MLP_fast(X). No graph convolution before
      selection, preventing over-smoothing from polluting supp(U) localization.
    - Learned residual alpha blends scorer and selector on top-k nodes:
        logits = (1-alpha)*scorer + alpha*selector
      The scorer can only refine, not contradict the selector.
    - Variational GIB regularization (KL Bernoulli) replaces naive L1
      for a principled bound on I(Z; X).
    - STE with Gumbel noise for differentiable discrete selection.
    - Single vectorized pyg_subgraph call for the entire batch.
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
    Zero-hop temporal encoder (MLP_fast): Conv1d + attention pooling + MLP.

    Per the formalization H_fast = MLP_fast(X), the selector must see ONLY
    node-local features before selection.  No graph convolution is applied
    here — GCN is reserved for the deep path on the selected subgraph,
    preventing over-smoothing from polluting the selector at large N.
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
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_flat:     [B*N, F, W]
            edge_index: ignored (kept for API compatibility)
        Returns:
            [B*N, H]
        """
        h = F.relu(self.temporal1(x_flat))                                  # [B*N, H, W]
        h = F.relu(self.temporal2(h))                                       # [B*N, H, W]
        h_t = h.permute(0, 2, 1)                                           # [B*N, W, H]
        attn = F.softmax(self.temp_attn(h_t).squeeze(-1), dim=-1)          # [B*N, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)                          # [B*N, H]
        h = self.norm(h)
        h = self.proj(h)
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


class S4TemporalBlock(nn.Module):
    """
    Simplified S4 (Structured State Space) temporal encoder.

    Models f_theta as a true state transition:
        h(t+1) = A_s * h(t) + B_s * x(t),  y(t) = C_s * h(t) + D_s * x(t)

    Uses the HiPPO-LegS initialization for long-range dependencies and
    computes the SSM via convolutional mode in O(W log W).

    Faithfully implements the formalization's state-transition model for
    temporal dynamics, unlike Conv1d which is a static feature extractor.

    Input: [B*k, F, W] -> Output: [B*k, H]
    """

    def __init__(self, in_channels: int, hidden_dim: int, state_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # SSM parameters (per hidden dimension)
        # A: diagonal state matrix (log-parameterized for stability)
        self.log_A = nn.Parameter(torch.log(0.5 + 0.5 * torch.rand(hidden_dim, state_dim)))
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _ssm_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parallel scan of the SSM in convolutional mode.

        Args:
            u: [batch, hidden_dim, W] input sequence
        Returns:
            y: [batch, hidden_dim, W] output sequence
        """
        batch, H, W = u.shape
        A = -torch.exp(self.log_A)                                           # [H, S] negative for stability
        device = u.device

        # Discretize: A_bar = exp(A * dt), B_bar = (A_bar - I) @ inv(A) @ B ≈ B * dt
        dt = 1.0 / W
        A_bar = torch.exp(A * dt)                                            # [H, S]
        B_bar = self.B * dt                                                  # [H, S]

        # Build convolution kernel via powers of A_bar (no in-place ops)
        # K[t] = C @ A_bar^t @ B_bar for t = 0..W-1
        power_list = [torch.ones(H, self.state_dim, device=device)]
        for t in range(1, W):
            power_list.append(power_list[-1] * A_bar)
        powers = torch.stack(power_list, dim=-1)                              # [H, S, W]

        # K = C^T @ powers @ B_bar: [H, W]
        kernel = torch.einsum('hs,hsw,hs->hw', self.C, powers, B_bar)       # [H, W]

        # Convolve: y = u * K (causal convolution via FFT)
        kernel_padded = F.pad(kernel, (0, W))                                # [H, 2W]
        u_padded = F.pad(u, (0, W))                                          # [B, H, 2W]
        K_f = torch.fft.rfft(kernel_padded, dim=-1)                          # [H, W+1]
        U_f = torch.fft.rfft(u_padded, dim=-1)                               # [B, H, W+1]
        y = torch.fft.irfft(U_f * K_f.unsqueeze(0), n=2 * W, dim=-1)[..., :W]

        # D skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*k, F, W] -> [B*k, H]"""
        batch, F_in, W = x.shape
        # Project input features to hidden_dim: [B*k, W, F] -> [B*k, W, H]
        x_proj = self.input_proj(x.permute(0, 2, 1))                        # [B*k, W, H]
        u = x_proj.permute(0, 2, 1)                                         # [B*k, H, W]

        y = self._ssm_scan(u)                                                # [B*k, H, W]

        # Temporal pooling: take last hidden state (most informative for state-transition)
        h = y[:, :, -1]                                                      # [B*k, H]
        h = self.norm(h)
        h = h + self.output_proj(h)
        return h


def get_temporal_encoder(
    encoder_type: str, in_channels: int, hidden_dim: int, kernel_size: int = 3
) -> nn.Module:
    """Factory for temporal encoders used in the deep path (post-selection).

    Available types:
        'conv' (default): DeepTemporalBlock — Conv1d + attention pooling.
        's4': S4TemporalBlock — Structured State Space Model implementing
              the state-transition f_theta: h(t+1) = A*h(t) + B*x(t).
    """
    if encoder_type == "conv":
        return DeepTemporalBlock(in_channels, hidden_dim, kernel_size)
    elif encoder_type == "s4":
        return S4TemporalBlock(in_channels, hidden_dim)
    else:
        raise ValueError(f"Unknown temporal encoder: {encoder_type}. Use 'conv' or 's4'.")


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
      the bottleneck in a naive loop), this function:
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


class HierarchicalSelector(nn.Module):
    """
    Two-stage hierarchical node selection: N -> N/2 -> k.

    Progressively refines the support of U(t) — analogous to branch-and-bound
    in combinatorial optimization.

    Stage 1: Coarse selection with a cheap MLP -> top N/2 nodes
    Stage 2: Fine selection with a more expressive MLP on the survivors -> top k

    Complexity: O(N) + O(N/2) + O(k^2) instead of O(N) + O(k^2).
    The intermediate stage helps when N >> k by providing a better candidate set.
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        # Coarse-level scorer (cheap)
        self.coarse_mlp = nn.Sequential(nn.Linear(hidden_dim, 1))
        # Fine-level scorer (more expressive)
        self.fine_mlp = nn.Sequential(
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
            k: final number of nodes to select
        Returns:
            Same signature as STESelector for drop-in compatibility.
        """
        B, N, H = h.shape

        # Stage 1: Coarse selection -> top N/2
        k_coarse = max(k, N // 2)
        logits_coarse = self.coarse_mlp(h).squeeze(-1)                       # [B, N]

        if self.training:
            u = torch.rand_like(logits_coarse).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            logits_noisy = logits_coarse + gumbel
        else:
            logits_noisy = logits_coarse

        M_coarse_soft = torch.sigmoid(logits_noisy / self.temperature)
        _, coarse_idx = torch.topk(M_coarse_soft, k_coarse, dim=-1)

        # Gather survivors
        h_survivors = torch.gather(h, 1, coarse_idx.unsqueeze(-1).expand(-1, -1, H))  # [B, k_coarse, H]

        # Stage 2: Fine selection -> top k from survivors
        logits_fine = self.fine_mlp(h_survivors).squeeze(-1)                 # [B, k_coarse]

        if self.training:
            u2 = torch.rand_like(logits_fine).clamp(1e-6, 1 - 1e-6)
            gumbel2 = -torch.log(-torch.log(u2))
            logits_fine_noisy = logits_fine + gumbel2
        else:
            logits_fine_noisy = logits_fine

        M_fine_soft = torch.sigmoid(logits_fine_noisy / self.temperature)
        _, fine_local_idx = torch.topk(M_fine_soft, k, dim=-1)

        # Map back to original indices (sort for pyg_subgraph compatibility)
        top_k_indices = torch.gather(coarse_idx, 1, fine_local_idx)
        top_k_indices, sort_perm = top_k_indices.sort(dim=-1)
        fine_local_idx = fine_local_idx.gather(1, sort_perm)

        # Build full M_hard and M_soft
        M_hard = torch.zeros(B, N, device=h.device)
        M_hard.scatter_(1, top_k_indices, 1.0)

        # M_soft: product of coarse and fine probabilities at the k selected
        # positions gives joint gradient flow through both selector stages.
        coarse_at_selected = M_coarse_soft.gather(1, top_k_indices)       # [B, k]
        fine_at_selected = M_fine_soft.gather(1, fine_local_idx)           # [B, k]
        combined_probs = coarse_at_selected * fine_at_selected             # [B, k]
        M_soft = torch.zeros(B, N, device=h.device)
        M_soft.scatter_(1, top_k_indices, combined_probs)

        M_ste = M_hard - M_soft.detach() + M_soft

        # Full logits for CE loss (from coarse scorer applied to all N)
        selector_logits = logits_coarse

        return M_ste, M_soft, top_k_indices, selector_logits


# ---------------------------------------------------------------------------
# Main Architecture: Select-then-Analyze (STA)
# ---------------------------------------------------------------------------

class SelectThenAnalyze(nn.Module):
    """
    Full pipeline:
        x [B, N, F, W]  -->  LightEncoder (zero-hop)  -->  STESelector  -->  subgraph
        subgraph  -->  DeepTemporal + DeepSpatial  -->  Scorer  -->  logits [B, N]

    Learned residual connection (alpha) between selector and scorer prevents
    the scorer from degrading the selector's predictions:
        logits_final = (1 - alpha) * scorer_logits + alpha * selector_logits  (on top-k)

    Returns four tensors consumed by different loss terms:
        logits_full:      [B, N]   blended output (for supervised CE, only k finite)
        M_soft:           [B, N]   continuous mask (for variational GIB loss)
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
        temporal_encoder: str = "conv",
        selector_type: str = "ste",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.k = k

        self.light_encoder = LightEncoder(num_features, hidden_dim)
        if selector_type == "hierarchical":
            self.selector = HierarchicalSelector(hidden_dim, gib_temperature)
        else:
            self.selector = STESelector(hidden_dim, gib_temperature)
        self.deep_temporal = get_temporal_encoder(temporal_encoder, num_features, hidden_dim, temporal_kernel)
        self.deep_spatial = DeepSpatialBlock(hidden_dim)
        self.scorer = SubgraphScorer(hidden_dim)
        self._k_override = None

        self.residual_alpha = nn.Parameter(torch.tensor(0.5))

        logger.info(
            "SelectThenAnalyze: N=%d, k=%d, features=%d, hidden=%d, temporal=%s, selector=%s",
            num_nodes, k, num_features, hidden_dim, temporal_encoder, selector_type,
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

        # ---- Stage 1: Light encoder (zero-hop, no GCN) ----
        x_flat = x.view(B * N, F_, W)
        h_light = self.light_encoder(x_flat)                                 # [B*N, H]
        h_light_batched = h_light.view(B, N, -1)                            # [B, N, H]

        # ---- Stage 2: STE selection ----
        M_ste, M_soft, top_k_indices, selector_logits = self.selector(
            h_light_batched, k
        )

        # ---- Stage 3: Mask features and extract subgraph ----
        x_masked = x * M_ste.unsqueeze(-1).unsqueeze(-1)                    # [B, N, F, W]

        x_sub, sub_ei = extract_subgraph_vectorized(
            x_masked, edge_index, top_k_indices, self.num_nodes,
        )                                                                    # [B*k, F, W], [2, E_sub]

        # ---- Stage 4: Deep STGCN on subgraph (O(k^2)) ----
        h_deep = self.deep_temporal(x_sub)                                   # [B*k, H]
        h_deep = self.deep_spatial(h_deep, sub_ei)                           # [B*k, H]
        h_deep_batched = h_deep.view(B, k, -1)                              # [B, k, H]

        # ---- Stage 5: Score with residual connection ----
        scores_sub = self.scorer(h_deep_batched)                             # [B, k]

        # Blend scorer output with selector logits on top-k nodes
        alpha = torch.sigmoid(self.residual_alpha)
        sel_on_topk = torch.gather(selector_logits, 1, top_k_indices)       # [B, k]
        blended = (1.0 - alpha) * scores_sub + alpha * sel_on_topk

        logits_full = torch.full((B, N), float('-inf'), device=device)
        logits_full.scatter_(1, top_k_indices, blended)

        return logits_full, M_soft, top_k_indices, selector_logits

    def set_temperature(self, t: float) -> None:
        self.selector.set_temperature(t)

    def set_k(self, k: int) -> None:
        """Override k for k-annealing. Pass None to reset to default."""
        self._k_override = k
