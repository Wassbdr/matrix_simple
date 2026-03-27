"""
cascade_aware_loss.py — Cascade-structure regularization for RCA.

Implements two cascade-aware regularization terms:

1. CascadeAwareLoss (original):
    Hinge loss on structural hop distance — inspired by Chain-of-Event (FSE 2024).
    s[rc] > s[1hop] + margin_1, s[1hop] > s[2hop] + margin_2.

2. CausalCascadeAwareLoss (new):
    Hinge loss with causal edge weights from causal_graph_builder.py.
    Margins are proportional to causal weight (stronger causal link → larger margin).

Usage:
    from cascade_aware_loss import CascadeAwareLoss, CausalCascadeAwareLoss

    # Structural version
    loss = criterion(logits, y) + 0.2 * CascadeAwareLoss()(logits, y, edge_index)

    # Causal version (edges + weights from CausalGraphBuilder)
    loss = criterion(logits, y) + 0.2 * CausalCascadeAwareLoss()(
        logits, y, edge_index, edge_weights
    )

References:
    Chain-of-Event (FSE 2024), CKG (§4.3), PC/FCI (§6.2)
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph hop utilities
# ---------------------------------------------------------------------------

def get_neighbors(node_idx: int, edge_index: torch.Tensor) -> torch.Tensor:
    """Get direct neighbors of node_idx (outgoing edges: node -> neighbor)."""
    src = edge_index[0]
    dst = edge_index[1]
    mask = src == node_idx
    return dst[mask]


def get_k_hop_neighbors(
    node_idx: int,
    edge_index: torch.Tensor,
    k: int,
    exclude: Optional[set] = None,
) -> torch.Tensor:
    """
    Get all nodes reachable from node_idx in exactly k hops.

    Args:
        node_idx:   source node
        edge_index: [2, E] directed adjacency
        k:          number of hops
        exclude:    node indices to exclude from output (e.g., already seen nodes)
    Returns:
        tensor of reachable node indices (may be empty)
    """
    if exclude is None:
        exclude = set()
    current = {node_idx}
    seen = {node_idx}
    for _ in range(k):
        next_nodes = set()
        for n in current:
            nbrs = get_neighbors(n, edge_index)
            for nb in nbrs.tolist():
                if nb not in seen:
                    next_nodes.add(nb)
                    seen.add(nb)
        current = next_nodes
    result = [n for n in current if n not in exclude]
    if not result:
        return torch.tensor([], dtype=torch.long, device=edge_index.device)
    return torch.tensor(result, dtype=torch.long, device=edge_index.device)


# ---------------------------------------------------------------------------
# Cascade-aware loss
# ---------------------------------------------------------------------------

class CascadeAwareLoss(torch.nn.Module):
    """
    Cascade-structure regularization loss.

    For each sample in the batch with ground-truth root cause rc:
        1. Get 1-hop neighbors of rc from directed edge_index.
        2. Compute hinge: max(0, s[1hop] - s[rc] + margin_1hop)
        3. Get 2-hop neighbors.
        4. Compute hinge: max(0, s[2hop] - s[1hop_min] + margin_2hop)
           (use min score among 1-hoppers as the "cause" proxy)

    Total: mean of all violated constraints.

    Note: Only works when the ground-truth RC is known (training time).
    At inference, the loss is undefined — do not use for eval.

    Args:
        margin_1hop: required score gap between RC and 1-hop nodes (default 0.5)
        margin_2hop: required score gap between 1-hop and 2-hop nodes (default 0.3)
        max_neighbors: cap on neighbors per hop to avoid O(N^2) in dense graphs
    """

    def __init__(
        self,
        margin_1hop: float = 0.5,
        margin_2hop: float = 0.3,
        max_neighbors: int = 20,
    ) -> None:
        super().__init__()
        self.margin_1hop = margin_1hop
        self.margin_2hop = margin_2hop
        self.max_neighbors = max_neighbors

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cascade regularization loss.

        Args:
            logits:     [B, N] — per-node scores
            y:          [B]    — ground-truth root cause indices
            edge_index: [2, E] — directed dependency graph (shared across batch)
        Returns:
            Scalar loss (mean violation).
        """
        B, N = logits.shape
        device = logits.device
        total_violation = torch.tensor(0.0, device=device)
        n_constraints = 0

        for b in range(B):
            rc = y[b].item()
            s = logits[b]  # [N]
            s_rc = s[rc]

            # --- 1-hop constraints: s[rc] > s[1hop] + margin ---
            hop1 = get_k_hop_neighbors(rc, edge_index, k=1)
            if hop1.numel() > 0:
                hop1 = hop1[:self.max_neighbors]
                violations_1 = F.relu(s[hop1] - s_rc + self.margin_1hop)
                total_violation = total_violation + violations_1.sum()
                n_constraints += hop1.numel()

                # --- 2-hop constraints: s[1hop_min] > s[2hop] + margin ---
                hop1_scores = s[hop1]
                if hop1_scores.numel() > 0:
                    s_1hop_min = hop1_scores.min()
                    hop2 = get_k_hop_neighbors(
                        rc, edge_index, k=2, exclude={rc} | set(hop1.tolist())
                    )
                    if hop2.numel() > 0:
                        hop2 = hop2[:self.max_neighbors]
                        violations_2 = F.relu(s[hop2] - s_1hop_min + self.margin_2hop)
                        total_violation = total_violation + violations_2.sum()
                        n_constraints += hop2.numel()

        if n_constraints == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_violation / n_constraints


class CausalCascadeAwareLoss(torch.nn.Module):
    """
    Cascade regularization with causal edge weights.

    Instead of fixed margins, the margin between connected nodes is proportional
    to the causal weight of the edge:
        margin(i,j) = base_margin * edge_weight(i,j)

    This encourages the model to respect the causal structure more strongly
    on edges that the causal discovery algorithm found significant.
    """

    def __init__(self, base_margin: float = 0.3, max_neighbors: int = 20) -> None:
        super().__init__()
        self.base_margin = base_margin
        self.max_neighbors = max_neighbors

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:       [B, N]
            y:            [B]
            edge_index:   [2, E]
            edge_weights: [E] — causal weights (default 1.0 if None)
        """
        B, N = logits.shape
        device = logits.device
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), device=device)
        edge_weights = edge_weights.to(device)

        total_violation = torch.tensor(0.0, device=device)
        n_constraints = 0

        for b in range(B):
            rc = y[b].item()
            s = logits[b]
            s_rc = s[rc]

            mask = edge_index[0] == rc
            dst = edge_index[1, mask][:self.max_neighbors]
            w = edge_weights[mask][:self.max_neighbors]

            if dst.numel() == 0:
                continue

            margins = self.base_margin * w
            violations = F.relu(s[dst] - s_rc + margins)
            total_violation = total_violation + violations.sum()
            n_constraints += dst.numel()

        if n_constraints == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_violation / n_constraints


class LearnedPropagationCascadeLoss(torch.nn.Module):
    """
    Cascade regularization with a learnable propagation matrix B.

    Formalization: X(t+1) = f_theta(A, X(t)) + B * U(t)
    B encodes how perturbations propagate through the graph.

    Instead of fixed hinge margins, B is a learnable nn.Parameter initialized
    from the causal graph weights. The loss enforces that the root cause
    score exceeds downstream nodes by a margin proportional to B_ij:
        L = sum_{(i,j)} B_ij * max(0, s_j - s_i + delta)

    Args:
        num_nodes: N
        edge_index: [2, E] initial graph topology
        edge_weights: [E] initial causal weights (default 1.0)
        base_margin: initial learned margin
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        base_margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # Build dense B matrix from sparse edge_index
        B_init = torch.zeros(num_nodes, num_nodes)
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        for e in range(edge_index.size(1)):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            if i < num_nodes and j < num_nodes:
                B_init[i, j] = max(B_init[i, j].item(), float(edge_weights[e]))

        self.B = torch.nn.Parameter(B_init)
        self.delta = torch.nn.Parameter(torch.tensor(base_margin))

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, N] per-node scores
            y:      [B]    ground-truth root cause indices
        Returns:
            Scalar cascade propagation loss.
        """
        batch_size = logits.size(0)
        device = logits.device
        B_abs = self.B.abs()
        delta = F.softplus(self.delta)
        total_loss = torch.tensor(0.0, device=device)
        n_constraints = 0

        for b in range(batch_size):
            rc = y[b].item()
            s_rc = logits[b, rc]

            # All downstream nodes from rc (non-zero B entries)
            b_row = B_abs[rc]
            downstream = (b_row > 1e-6).nonzero(as_tuple=True)[0]

            if downstream.numel() == 0:
                continue

            margins = b_row[downstream] * delta
            violations = F.relu(logits[b, downstream] - s_rc + margins)
            total_loss = total_loss + violations.sum()
            n_constraints += downstream.numel()

        if n_constraints == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / n_constraints


def cascade_aware_total_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    lambda_cascade: float = 0.2,
    margin_1hop: float = 0.5,
    margin_2hop: float = 0.3,
) -> torch.Tensor:
    """
    Combined CE + cascade regularization loss.

    Args:
        logits:         [B, N] — per-node scores
        y:              [B]    — ground-truth root cause indices
        edge_index:     [2, E] — directed dependency graph
        lambda_cascade: weight for the cascade regularization term
        margin_1hop:    hinge margin for 1-hop constraint
        margin_2hop:    hinge margin for 2-hop constraint
    Returns:
        Scalar total loss.
    """
    ce_loss = F.cross_entropy(logits, y)
    cascade_loss_fn = CascadeAwareLoss(margin_1hop=margin_1hop, margin_2hop=margin_2hop)
    l_cascade = cascade_loss_fn(logits, y, edge_index)
    return ce_loss + lambda_cascade * l_cascade


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=== CascadeAwareLoss unit tests ===")

    N = 10
    # Simple chain: 0 -> 1 -> 2 -> 3 -> 4
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
    ])

    # Test 1: No violation — RC scores highest
    logits_good = torch.zeros(1, N)
    logits_good[0, 0] = 3.0   # RC
    logits_good[0, 1] = 1.0   # 1-hop
    logits_good[0, 2] = 0.5   # 2-hop
    y = torch.tensor([0])

    loss_fn = CascadeAwareLoss(margin_1hop=0.5, margin_2hop=0.3)
    loss_good = loss_fn(logits_good, y, edge_index)
    assert loss_good.item() < 0.01, f"Expected ~0 loss, got {loss_good.item()}"
    print(f"  Test 1 PASS: No constraint violation (loss={loss_good.item():.4f})")

    # Test 2: Violation — 1-hop scores higher than RC
    logits_bad = torch.zeros(1, N)
    logits_bad[0, 0] = 0.5    # RC (too low)
    logits_bad[0, 1] = 2.0    # 1-hop (too high)
    loss_bad = loss_fn(logits_bad, y, edge_index)
    assert loss_bad.item() > 0.5, f"Expected violation, got {loss_bad.item()}"
    print(f"  Test 2 PASS: Violation detected (loss={loss_bad.item():.4f})")

    # Test 3: Gradient flows
    logits_grad = torch.randn(2, N, requires_grad=True)
    y2 = torch.tensor([0, 3])
    total = cascade_aware_total_loss(logits_grad, y2, edge_index)
    total.backward()
    assert logits_grad.grad is not None
    print(f"  Test 3 PASS: Gradient flows (total_loss={total.item():.4f})")

    print("\n=== All tests passed ===")
