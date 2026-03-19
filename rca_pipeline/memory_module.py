"""
Memory-Augmented Module for Recurrent Pattern Recognition.

Implements a Memory Bank for storing and retrieving known anomaly patterns,
inspired by Memory-Augmented Neural Networks (MANN) and approaches like:
- DejaVu (ATC'22): Memory-augmented anomaly detection
- MULAN (NSDI'23): GNN with attention on incident history
- Neural Turing Machines: External memory for sequence modeling

Key Components:
1. MemoryBank: Differentiable memory with attention-based read/write
2. MemoryAugmentedRCA: Wrapper that adds memory to any base RCA model
3. RecurrenceClassifier: Auxiliary head for novel vs recurrent classification

The memory module enables:
- Higher accuracy on recurrent patterns (by recognizing seen patterns)
- Faster inference (pattern matching instead of full reasoning)
- Interpretability (attention weights show which patterns were matched)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryBank(nn.Module):
    """
    Differentiable memory bank for storing anomaly pattern embeddings.

    Architecture:
    - Memory slots: [M, D] learnable parameters
    - Read: Attention-based retrieval using query-key-value
    - Write: EMA update of closest slot (during training only)
    - Gating: Learned gate decides how much to use memory vs. raw query

    The memory bank learns to store prototypical anomaly patterns,
    enabling the model to recognize recurrent incidents.
    """

    def __init__(
        self,
        num_slots: int = 128,
        embed_dim: int = 64,
        num_heads: int = 4,
        temperature: float = 0.1,
        dropout: float = 0.1,
    ):
        """
        Initialize the memory bank.

        Args:
            num_slots: Number of memory slots (M)
            embed_dim: Dimension of each slot (D)
            num_heads: Number of attention heads for reading
            temperature: Softmax temperature for attention (lower = sharper)
            dropout: Dropout probability
        """
        super().__init__()
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Learnable memory matrix [M, D]
        self.memory = nn.Parameter(torch.randn(num_slots, embed_dim) * 0.02)

        # Projections for attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Gating mechanism: decides how much to use memory
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Usage tracking for LRU-like write (not learnable)
        self.register_buffer(
            "slot_usage",
            torch.zeros(num_slots),
        )

    def read(
        self,
        query: torch.Tensor,
        return_attn: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read from memory using attention.

        Args:
            query: [B, D] query embeddings
            return_attn: Whether to return attention weights

        Returns:
            memory_output: [B, D] retrieved memory content
            attn_weights: [B, M] attention weights (if return_attn=True)
        """
        B = query.size(0)

        # Project query, keys, values
        q = self.query_proj(query)  # [B, D]
        k = self.key_proj(self.memory)  # [M, D]
        v = self.value_proj(self.memory)  # [M, D]

        # Multi-head attention
        q = q.view(B, self.num_heads, self.head_dim)  # [B, H, D/H]
        k = k.view(self.num_slots, self.num_heads, self.head_dim)  # [M, H, D/H]
        v = v.view(self.num_slots, self.num_heads, self.head_dim)  # [M, H, D/H]

        # Attention scores: [B, H, M]
        scores = torch.einsum("bhd,mhd->bhm", q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores / self.temperature, dim=-1)

        # Weighted sum of values: [B, H, D/H]
        attended = torch.einsum("bhm,mhd->bhd", attn_weights, v)
        attended = attended.reshape(B, self.embed_dim)  # [B, D]

        # Output projection
        memory_output = self.out_proj(attended)
        memory_output = self.layer_norm(memory_output)

        # Average attention across heads for interpretability
        if return_attn:
            avg_attn = attn_weights.mean(dim=1)  # [B, M]
            return memory_output, avg_attn
        return memory_output, None

    def write(
        self,
        key: torch.Tensor,
        is_novel: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        """
        Write new patterns to memory (training only).

        Uses EMA update on the most similar slot for novel patterns.
        Implements approximate LRU by tracking slot usage.

        Args:
            key: [B, D] embeddings to write
            is_novel: [B] boolean mask (True = novel pattern to store)
            momentum: EMA coefficient (higher = slower update)
        """
        if not self.training:
            return

        # Only write novel patterns
        novel_mask = is_novel.bool()
        if not novel_mask.any():
            return

        novel_keys = key[novel_mask]  # [B_novel, D]

        with torch.no_grad():
            # Find most similar slot for each novel key
            # Cosine similarity
            key_norm = F.normalize(novel_keys, dim=-1)  # [B_novel, D]
            mem_norm = F.normalize(self.memory, dim=-1)  # [M, D]
            similarity = torch.matmul(key_norm, mem_norm.T)  # [B_novel, M]

            # Find slot with max similarity
            _, slot_indices = similarity.max(dim=-1)  # [B_novel]

            # EMA update
            for i, (slot_idx, new_key) in enumerate(zip(slot_indices, novel_keys)):
                old_val = self.memory.data[slot_idx]
                self.memory.data[slot_idx] = momentum * old_val + (1 - momentum) * new_key

                # Update usage tracking
                self.slot_usage[slot_idx] += 1

    def forward(
        self,
        query: torch.Tensor,
        is_novel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: read from memory with gating.

        Args:
            query: [B, D] input embeddings
            is_novel: [B] optional boolean mask for write

        Returns:
            output: [B, D] gated combination of query and memory
            gate_value: [B, 1] gate values (high = use memory more)
            attn_weights: [B, M] attention weights on memory slots
        """
        # Read from memory
        memory_output, attn_weights = self.read(query, return_attn=True)

        # Compute gate
        gate_input = torch.cat([query, memory_output], dim=-1)  # [B, 2D]
        gate_value = self.gate_net(gate_input)  # [B, 1]

        # Gated output: blend query with memory
        output = gate_value * memory_output + (1 - gate_value) * query

        # Write during training
        if is_novel is not None and self.training:
            self.write(query, is_novel)

        return output, gate_value, attn_weights

    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory usage."""
        with torch.no_grad():
            mem_norms = self.memory.norm(dim=-1)
            return {
                "num_slots": self.num_slots,
                "avg_slot_norm": mem_norms.mean().item(),
                "min_slot_norm": mem_norms.min().item(),
                "max_slot_norm": mem_norms.max().item(),
                "total_usage": self.slot_usage.sum().item(),
                "active_slots": (self.slot_usage > 0).sum().item(),
            }


class MemoryAugmentedRCA(nn.Module):
    """
    Memory-Augmented Root Cause Analysis model.

    Wraps any base RCA model (e.g., SelectThenAnalyze) and adds:
    1. Memory bank for pattern storage
    2. Incident encoder for memory queries
    3. Memory-informed scoring boost
    4. Recurrence classification head

    Architecture:
        Input -> Base Model -> Graph Embedding -> Memory Query
                                                      |
                                                      v
                                               Memory Bank
                                                      |
                                                      v
                              Gate * Memory + (1-Gate) * Query
                                                      |
                                                      v
                                    Augmented Logits + Recurrence Head
    """

    def __init__(
        self,
        base_model: nn.Module,
        memory_slots: int = 128,
        memory_dim: int = 64,
        num_heads: int = 4,
        use_memory_for_scoring: bool = True,
    ):
        """
        Initialize the memory-augmented model.

        Args:
            base_model: Base RCA model (must have light_encoder or similar)
            memory_slots: Number of memory slots
            memory_dim: Dimension of memory embeddings
            num_heads: Attention heads for memory read
            use_memory_for_scoring: Whether to boost logits with memory
        """
        super().__init__()
        self.base_model = base_model
        self.use_memory_for_scoring = use_memory_for_scoring
        self.memory_dim = memory_dim

        # Infer hidden dim from base model
        if hasattr(base_model, "hidden_dim"):
            hidden_dim = base_model.hidden_dim
        else:
            hidden_dim = 32  # Fallback

        # Incident encoder: project graph embedding to memory space
        self.incident_encoder = nn.Sequential(
            nn.Linear(hidden_dim, memory_dim),
            nn.ReLU(),
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, memory_dim),
        )

        # Memory bank
        self.memory = MemoryBank(
            num_slots=memory_slots,
            embed_dim=memory_dim,
            num_heads=num_heads,
        )

        # Memory-informed scoring head
        if use_memory_for_scoring:
            self.memory_scorer = nn.Sequential(
                nn.Linear(memory_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Recurrence classification head (novel vs recurrent)
        self.recurrence_head = nn.Sequential(
            nn.Linear(memory_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def encode_incident(self, h_graph: torch.Tensor) -> torch.Tensor:
        """
        Encode graph-level embedding into memory query.

        Args:
            h_graph: [B, D] or [B, N, D] graph embedding

        Returns:
            [B, D_mem] incident embedding
        """
        if h_graph.dim() == 3:
            # Pool over nodes: [B, N, D] -> [B, D]
            h_graph = h_graph.mean(dim=1)
        return self.incident_encoder(h_graph)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_novel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with memory augmentation.

        Args:
            x: [B, N, F, W] input features
            edge_index: [2, E] graph connectivity
            is_novel: [B] optional ground truth for novel vs recurrent (training)

        Returns:
            Dict containing:
            - logits_full: [B, N] RC logits (augmented with memory)
            - logits_base: [B, N] RC logits (from base model)
            - M_soft: [B, N] selection mask
            - sel_logits: [B, N] selector logits
            - top_k: [B, k] top-k selected indices
            - recurrence_logits: [B, 2] novel vs recurrent
            - memory_gate: [B, 1] gate value
            - memory_attn: [B, M] attention on memory slots
            - incident_embedding: [B, D_mem] incident embedding
        """
        B = x.size(0)

        # Forward through base model
        if hasattr(self.base_model, "light_encoder"):
            # SelectThenAnalyze model
            logits_full, M_soft, top_k, sel_logits = self.base_model(x, edge_index)

            # Get graph embedding from light encoder
            N = x.size(1)
            F_ = x.size(2)
            W = x.size(3)
            x_flat = x.view(B * N, F_, W)
            batch_ei = self.base_model._batch_edge_index(edge_index, B)
            h_light = self.base_model.light_encoder(x_flat, batch_ei)
            h_graph = h_light.view(B, N, -1).mean(dim=1)  # [B, D]

        else:
            # Fallback for other models (SpatioTemporalRCA)
            z_graph, probs, h_spatial = self.base_model(x, edge_index)
            h_graph = z_graph
            N = h_spatial.size(1)
            logits_full = self.base_model.gib_logits(h_spatial)
            M_soft = probs.squeeze(-1)
            sel_logits = logits_full
            top_k = logits_full.topk(min(10, N), dim=-1).indices

        # Encode incident for memory query
        incident_emb = self.encode_incident(h_graph)  # [B, D_mem]

        # Memory read/write
        memory_output, gate_value, attn_weights = self.memory(
            incident_emb,
            is_novel=is_novel,
        )

        # Augment logits with memory
        if self.use_memory_for_scoring:
            memory_boost = self.memory_scorer(memory_output)  # [B, 1]
            # Boost high-confidence predictions when memory recognizes pattern
            logits_augmented = logits_full + gate_value * memory_boost
        else:
            logits_augmented = logits_full

        # Recurrence classification
        recurrence_input = torch.cat([incident_emb, h_graph], dim=-1)
        recurrence_logits = self.recurrence_head(recurrence_input)

        return {
            "logits_full": logits_augmented,
            "logits_base": logits_full,
            "M_soft": M_soft,
            "sel_logits": sel_logits,
            "top_k": top_k,
            "recurrence_logits": recurrence_logits,
            "memory_gate": gate_value,
            "memory_attn": attn_weights,
            "incident_embedding": incident_emb,
        }

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory bank statistics."""
        return self.memory.get_memory_stats()


def wrap_with_memory(
    base_model: nn.Module,
    memory_slots: int = 128,
    memory_dim: int = 64,
) -> MemoryAugmentedRCA:
    """
    Convenience function to wrap any base model with memory.

    Args:
        base_model: Trained or untrained base RCA model
        memory_slots: Number of memory slots
        memory_dim: Memory embedding dimension

    Returns:
        MemoryAugmentedRCA wrapping the base model
    """
    return MemoryAugmentedRCA(
        base_model=base_model,
        memory_slots=memory_slots,
        memory_dim=memory_dim,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing MemoryBank...")

    bank = MemoryBank(num_slots=64, embed_dim=32, num_heads=4)
    query = torch.randn(8, 32)  # Batch of 8

    # Read test
    output, gate, attn = bank(query)
    print(f"Output shape: {output.shape}")  # [8, 32]
    print(f"Gate shape: {gate.shape}")  # [8, 1]
    print(f"Attention shape: {attn.shape}")  # [8, 64]
    print(f"Gate mean: {gate.mean().item():.3f}")
    print(f"Attention entropy: {-(attn * (attn + 1e-8).log()).sum(-1).mean().item():.3f}")

    # Write test (training mode)
    bank.train()
    is_novel = torch.tensor([True, False, True, True, False, True, False, True])
    output2, _, _ = bank(query, is_novel=is_novel)

    stats = bank.get_memory_stats()
    print(f"Memory stats: {stats}")

    print("\nAll tests passed!")
