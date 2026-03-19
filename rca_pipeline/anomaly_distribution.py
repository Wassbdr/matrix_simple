"""
Power-law distribution for root cause selection with recurrence support.

In production systems, incident patterns follow Pareto distributions:
- ~10% of services cause ~80% of incidents
- ~80% of incidents are recurrences of known patterns

This module provides:
1. PowerLawRootCauseDistribution: Generates RC indices following Zipf's law
2. RecurrentPatternTracker: Tracks historical patterns for recurrence sampling
3. Integration utilities for simulated_data_scalable.py

References:
- Google SRE Handbook: "Most incidents are repeats"
- Microsoft AIOps 2024: "80% recurrence rate in cloud incidents"
- DejaVu (ATC'22): Memory-augmented anomaly detection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IncidentRecord:
    """Record of a generated incident for recurrence tracking."""
    root_cause: int
    pattern_type: str
    affected_features: List[int]
    signal_strength: float
    timestamp: int  # Sample index when generated


@dataclass
class PatternSignature:
    """Signature of a recurring pattern."""
    pattern_type: str
    affected_features: Tuple[int, ...]
    avg_strength: float
    occurrences: int = 1

    def matches(self, other: 'PatternSignature', tolerance: float = 0.3) -> bool:
        """Check if two signatures are similar enough to be considered the same pattern."""
        if self.pattern_type != other.pattern_type:
            return False
        # Check feature overlap (at least 50% common features)
        common = set(self.affected_features) & set(other.affected_features)
        union = set(self.affected_features) | set(other.affected_features)
        if len(common) / max(len(union), 1) < 0.5:
            return False
        # Check strength similarity
        if abs(self.avg_strength - other.avg_strength) / max(self.avg_strength, 0.1) > tolerance:
            return False
        return True


class PowerLawRootCauseDistribution:
    """
    Power-law (Zipf) distribution for root cause sampling.

    Implements the Pareto principle: top 10% of nodes cause 80% of incidents.
    Supports both novel incidents (following power-law) and recurrent incidents
    (sampling from historical patterns).

    Attributes:
        num_nodes: Total number of nodes in the system
        alpha: Zipf exponent (typically 1.5-2.5)
        probabilities: Sampling probabilities for each node
        historical_incidents: List of past incidents for recurrence
        pattern_registry: Grouped patterns by root cause
    """

    def __init__(
        self,
        num_nodes: int,
        alpha: float = 2.0,
        top_k_weight: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize the power-law distribution.

        Args:
            num_nodes: Total number of nodes (N)
            alpha: Zipf exponent. Higher = more concentrated on top nodes
            top_k_weight: Target weight for top 10% of nodes (default 0.8 = 80%)
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.top_k_weight = top_k_weight
        self.rng = np.random.RandomState(seed)

        # Generate and normalize probabilities
        self.probabilities = self._generate_power_law()
        self._normalize_for_pareto()

        # Rank nodes by probability (for logging)
        self.node_ranks = np.argsort(self.probabilities)[::-1]

        # Recurrence tracking
        self.historical_incidents: List[IncidentRecord] = []
        self.pattern_registry: Dict[int, List[PatternSignature]] = defaultdict(list)
        self.rc_frequency: Dict[int, int] = defaultdict(int)

        logger.debug(
            "Initialized PowerLawRootCauseDistribution: N=%d, alpha=%.2f, "
            "top_10%%_weight=%.2f (target=%.2f)",
            num_nodes, alpha, self.probabilities[self.node_ranks[:max(1, num_nodes // 10)]].sum(),
            top_k_weight,
        )

    def _generate_power_law(self) -> np.ndarray:
        """Generate raw Zipf probabilities: p(k) ~ k^(-alpha)."""
        ranks = np.arange(1, self.num_nodes + 1, dtype=float)
        probs = ranks ** (-self.alpha)
        return probs / probs.sum()

    def _normalize_for_pareto(self) -> None:
        """
        Adjust probabilities so top 10% of nodes get target weight.

        This ensures the Pareto principle (80/20 rule) is enforced
        regardless of the alpha parameter.
        """
        top_k = max(1, self.num_nodes // 10)
        sorted_probs = np.sort(self.probabilities)[::-1]
        current_top_weight = sorted_probs[:top_k].sum()

        if current_top_weight < 1e-8:
            return

        # Scale factors
        scale_top = self.top_k_weight / current_top_weight
        remaining_weight = 1.0 - self.top_k_weight
        current_rest_weight = 1.0 - current_top_weight
        scale_rest = remaining_weight / max(current_rest_weight, 1e-8)

        # Apply scaling
        threshold = sorted_probs[top_k - 1]
        for i in range(self.num_nodes):
            if self.probabilities[i] >= threshold:
                self.probabilities[i] *= scale_top
            else:
                self.probabilities[i] *= scale_rest

        # Re-normalize to sum to 1
        self.probabilities /= self.probabilities.sum()

    def sample_root_cause(
        self,
        is_recurrent: Optional[bool] = None,
        recurrence_prob: float = 0.8,
    ) -> Tuple[int, bool]:
        """
        Sample a root cause index.

        Args:
            is_recurrent: Force recurrent (True) or novel (False).
                         If None, determined by recurrence_prob.
            recurrence_prob: Probability of sampling a recurrent pattern

        Returns:
            Tuple of (root_cause_index, is_actually_recurrent)
        """
        # Determine if this should be a recurrence
        if is_recurrent is None:
            has_history = len(self.historical_incidents) > 0
            is_recurrent = has_history and (self.rng.rand() < recurrence_prob)

        if is_recurrent and len(self.historical_incidents) > 0:
            # Sample from historical root causes, weighted by frequency
            unique_rcs = list(self.rc_frequency.keys())
            weights = np.array([self.rc_frequency[rc] for rc in unique_rcs], dtype=float)
            weights /= weights.sum()
            root_cause = self.rng.choice(unique_rcs, p=weights)
            return root_cause, True
        else:
            # Novel incident: sample from power-law distribution
            root_cause = self.rng.choice(self.num_nodes, p=self.probabilities)
            return root_cause, False

    def sample_pattern_for_rc(
        self,
        root_cause: int,
        is_recurrent: bool,
        available_patterns: List[str],
        pattern_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, bool]:
        """
        Sample a pattern type for a given root cause.

        If recurrent and this RC has known patterns, reuse one.
        Otherwise, sample a new pattern type (optionally weighted).

        Args:
            root_cause: The root cause node index
            is_recurrent: Whether this is a recurrence
            available_patterns: List of available pattern type names
            pattern_weights: Optional dict mapping pattern names to weights
                            (e.g., {"gray_failure": 0.39, ...} for SOTA distribution)

        Returns:
            Tuple of (pattern_type, is_pattern_reused)
        """
        if is_recurrent and root_cause in self.pattern_registry:
            known = self.pattern_registry[root_cause]
            if known:
                # Weighted by occurrence count
                weights = np.array([p.occurrences for p in known], dtype=float)
                weights /= weights.sum()
                idx = self.rng.choice(len(known), p=weights)
                return known[idx].pattern_type, True

        # Novel pattern - use weights if provided (SOTA: 39% gray failures)
        if pattern_weights:
            weights = np.array([pattern_weights.get(p, 1.0) for p in available_patterns])
            weights /= weights.sum()
            return self.rng.choice(available_patterns, p=weights), False

        # Uniform sampling (fallback)
        return self.rng.choice(available_patterns), False

    def register_incident(
        self,
        root_cause: int,
        pattern_type: str,
        affected_features: List[int],
        signal_strength: float,
        sample_idx: int,
    ) -> None:
        """
        Register an incident in the history for future recurrence sampling.

        Args:
            root_cause: Root cause node index
            pattern_type: Type of temporal pattern used
            affected_features: List of affected feature indices
            signal_strength: Amplitude of the anomaly signal
            sample_idx: Index of this sample in the dataset
        """
        record = IncidentRecord(
            root_cause=root_cause,
            pattern_type=pattern_type,
            affected_features=affected_features,
            signal_strength=signal_strength,
            timestamp=sample_idx,
        )
        self.historical_incidents.append(record)
        self.rc_frequency[root_cause] += 1

        # Update pattern registry
        new_sig = PatternSignature(
            pattern_type=pattern_type,
            affected_features=tuple(sorted(affected_features)),
            avg_strength=signal_strength,
            occurrences=1,
        )

        # Check if this pattern already exists for this RC
        existing = self.pattern_registry[root_cause]
        for sig in existing:
            if sig.matches(new_sig):
                # Update existing pattern
                sig.avg_strength = (sig.avg_strength * sig.occurrences + signal_strength) / (sig.occurrences + 1)
                sig.occurrences += 1
                return

        # New pattern for this RC
        self.pattern_registry[root_cause].append(new_sig)

    def get_statistics(self) -> Dict[str, Any]:
        """Get distribution statistics for logging/validation."""
        top_k = max(1, self.num_nodes // 10)
        top_k_indices = self.node_ranks[:top_k]

        stats = {
            "num_nodes": self.num_nodes,
            "alpha": self.alpha,
            "top_10pct_theoretical_weight": self.probabilities[top_k_indices].sum(),
            "total_incidents": len(self.historical_incidents),
            "unique_root_causes": len(self.rc_frequency),
            "patterns_per_rc": {
                rc: len(patterns) for rc, patterns in self.pattern_registry.items()
            },
        }

        if self.historical_incidents:
            actual_top_k_incidents = sum(
                1 for inc in self.historical_incidents
                if inc.root_cause in top_k_indices
            )
            stats["top_10pct_actual_weight"] = actual_top_k_incidents / len(self.historical_incidents)

        return stats

    def reset_history(self) -> None:
        """Clear historical incidents (e.g., between dataset generations)."""
        self.historical_incidents.clear()
        self.pattern_registry.clear()
        self.rc_frequency.clear()


def validate_distribution(
    dist: PowerLawRootCauseDistribution,
    num_samples: int = 10000,
    expected_top_weight: float = 0.8,
    tolerance: float = 0.05,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that the distribution follows expected Pareto properties.

    Args:
        dist: Distribution to validate
        num_samples: Number of samples to draw
        expected_top_weight: Expected weight for top 10% (default 0.8)
        tolerance: Acceptable deviation from expected weight

    Returns:
        Tuple of (is_valid, metrics_dict)
    """
    samples = [dist.sample_root_cause(is_recurrent=False)[0] for _ in range(num_samples)]

    # Count by node
    counts = np.zeros(dist.num_nodes)
    for s in samples:
        counts[s] += 1

    # Top 10% weight
    top_k = max(1, dist.num_nodes // 10)
    top_k_indices = np.argsort(counts)[::-1][:top_k]
    top_weight = counts[top_k_indices].sum() / num_samples

    metrics = {
        "top_10pct_weight": top_weight,
        "expected_weight": expected_top_weight,
        "deviation": abs(top_weight - expected_top_weight),
        "unique_sampled": (counts > 0).sum(),
    }

    is_valid = metrics["deviation"] <= tolerance

    return is_valid, metrics


if __name__ == "__main__":
    # Quick validation
    import sys

    print("Testing PowerLawRootCauseDistribution...")

    for N in [50, 100, 200]:
        dist = PowerLawRootCauseDistribution(num_nodes=N, alpha=2.0, seed=42)
        is_valid, metrics = validate_distribution(dist, num_samples=10000)

        status = "PASS" if is_valid else "FAIL"
        print(f"N={N}: [{status}] top_10%%={metrics['top_10pct_weight']:.3f} "
              f"(expected={metrics['expected_weight']:.3f}, dev={metrics['deviation']:.3f})")

    # Test recurrence
    print("\nTesting recurrence sampling...")
    dist = PowerLawRootCauseDistribution(num_nodes=100, seed=42)

    # Generate some incidents
    for i in range(100):
        rc, is_rec = dist.sample_root_cause(recurrence_prob=0.8)
        dist.register_incident(
            root_cause=rc,
            pattern_type="exponential_decay",
            affected_features=[0, 1],
            signal_strength=3.0,
            sample_idx=i,
        )

    # Check recurrence rate
    recurrent_count = 0
    for i in range(1000):
        _, is_rec = dist.sample_root_cause(recurrence_prob=0.8)
        if is_rec:
            recurrent_count += 1

    print(f"Recurrence rate: {recurrent_count / 1000:.2%} (expected ~80%)")

    stats = dist.get_statistics()
    print(f"Unique root causes: {stats['unique_root_causes']}")
    print(f"Top 10% actual weight: {stats.get('top_10pct_actual_weight', 'N/A'):.3f}")
