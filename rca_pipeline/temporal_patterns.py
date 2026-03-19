"""
Temporal anomaly patterns for realistic RCA simulation.

Implements diverse temporal signatures observed in production systems:
1. ExponentialDecay - Classic transient error (spike then decay)
2. RetryStorm - Exponential backoff retries (periodic spikes)
3. MemoryLeak - Gradual increase then OOM crash
4. ColdStart - High latency at startup, rapid normalization
5. ConnectionPoolExhaustion - Gradual saturation then plateau

Each pattern generates a [W, F] tensor where W=window_size, F=num_features.
Patterns affect only the specified feature subset (affected_feats).

Usage:
    from temporal_patterns import PATTERN_REGISTRY, inject_anomaly_pattern

    pattern = PATTERN_REGISTRY['retry_storm']
    signal = pattern.generate(window_size=12, num_features=5, affected_feats=[0, 1], rng=rng)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TemporalPattern(ABC):
    """
    Abstract base class for temporal anomaly patterns.

    Each pattern generates a characteristic temporal signature that
    can be recognized by the RCA model. Patterns should be:
    - Distinguishable: Different patterns produce visibly different signatures
    - Realistic: Based on actual production incident patterns
    - Configurable: Support randomization for diversity
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this pattern type."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the pattern."""
        pass

    @abstractmethod
    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate the temporal signal for this pattern.

        Args:
            window_size: Number of timesteps (W)
            num_features: Total number of features (F)
            affected_feats: Indices of features to affect
            rng: Random state for reproducibility
            **kwargs: Pattern-specific parameters (strength, etc.)

        Returns:
            Tensor of shape [W, F] with the anomaly signal
        """
        pass

    def get_default_params(self) -> Dict:
        """Return default parameters for this pattern."""
        return {}


class ExponentialDecay(TemporalPattern):
    """
    Classic transient error pattern: spike followed by exponential decay.

    Common in: Transient network errors, brief resource contention,
    single request failures, timeout events.

    Shape: High initial spike that decays exponentially over time.
    """

    @property
    def name(self) -> str:
        return "exponential_decay"

    @property
    def description(self) -> str:
        return "Spike followed by exponential decay (transient error)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        strength: float = 3.0,
        decay_rate: float = 0.5,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        # Randomize start time and parameters
        t_start = rng.randint(0, max(1, W - 4))
        actual_strength = strength * (0.8 + rng.rand() * 0.4)  # +/- 20%
        actual_decay = decay_rate * (0.8 + rng.rand() * 0.4)

        # Generate decaying signal
        for dt in range(W - t_start):
            t = t_start + dt
            value = actual_strength * (actual_decay ** dt)
            if value < 0.05:  # Stop when signal is negligible
                break
            # Add small noise
            noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.1
            signal[t, affected_feats] = value + noise

        return signal

    def get_default_params(self) -> Dict:
        return {"strength": 3.0, "decay_rate": 0.5}


class RetryStorm(TemporalPattern):
    """
    Exponential backoff retry pattern: periodic spikes with increasing intervals.

    Common in: Circuit breaker activation, service unavailability with retries,
    distributed system cascading failures.

    Shape: Spikes at t=0, t=1, t=3, t=7... (exponential backoff intervals)
    """

    @property
    def name(self) -> str:
        return "retry_storm"

    @property
    def description(self) -> str:
        return "Exponential backoff retries (periodic spikes)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        strength: float = 2.5,
        max_retries: int = 6,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        t_start = rng.randint(0, max(1, W // 3))
        actual_strength = strength * (0.8 + rng.rand() * 0.4)

        # Calculate retry times with exponential backoff
        retry_times = [t_start]
        interval = 1
        for _ in range(max_retries):
            next_t = retry_times[-1] + interval
            if next_t >= W:
                break
            retry_times.append(next_t)
            interval = min(interval * 2, W // 2)  # Cap interval growth

        # Generate spikes at retry times
        for i, t in enumerate(retry_times):
            if t >= W:
                continue
            # Each retry is slightly weaker (service degradation)
            spike_strength = actual_strength * (0.85 ** i)
            noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.15
            signal[t, affected_feats] = spike_strength + noise

        return signal

    def get_default_params(self) -> Dict:
        return {"strength": 2.5, "max_retries": 6}


class MemoryLeak(TemporalPattern):
    """
    Gradual memory leak pattern: slow increase followed by OOM crash.

    Common in: Memory leaks, resource exhaustion, unbounded queue growth,
    connection/handle leaks.

    Shape: Linear/exponential ramp up, then sudden spike at crash.
    """

    @property
    def name(self) -> str:
        return "memory_leak"

    @property
    def description(self) -> str:
        return "Gradual increase then OOM crash (memory leak)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        leak_rate: float = 0.15,
        crash_strength: float = 4.0,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_leak_rate = leak_rate * (0.8 + rng.rand() * 0.4)
        actual_crash = crash_strength * (0.9 + rng.rand() * 0.2)

        # Gradual increase (exponential growth)
        for t in range(W):
            # Slow exponential growth: (1 + rate)^t - 1
            level = (1 + actual_leak_rate) ** t - 1
            noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.1
            signal[t, affected_feats] = level * 0.3 + noise

        # OOM crash spike at the end
        if W > 2:
            crash_noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.2
            signal[-1, affected_feats] += actual_crash + crash_noise
            # Sometimes the crash affects one timestep before
            if rng.rand() < 0.3 and W > 3:
                signal[-2, affected_feats] += actual_crash * 0.5

        return signal

    def get_default_params(self) -> Dict:
        return {"leak_rate": 0.15, "crash_strength": 4.0}


class ColdStart(TemporalPattern):
    """
    Cold start latency pattern: high initial latency, rapid normalization.

    Common in: Container cold starts, JIT compilation warmup, cache misses
    after restart, connection pool initialization.

    Shape: High spike at start, rapid exponential decay to baseline.
    """

    @property
    def name(self) -> str:
        return "cold_start"

    @property
    def description(self) -> str:
        return "High initial latency, rapid normalization (cold start)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        peak_strength: float = 5.0,
        warmup_duration: int = 3,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_peak = peak_strength * (0.8 + rng.rand() * 0.4)
        actual_duration = min(max(1, warmup_duration + rng.randint(-1, 2)), W)

        # Rapid decay during warmup period
        for t in range(actual_duration):
            # Fast exponential decay: peak * 0.4^t
            value = actual_peak * (0.4 ** t)
            noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.15
            signal[t, affected_feats] = value + noise

        return signal

    def get_default_params(self) -> Dict:
        return {"peak_strength": 5.0, "warmup_duration": 3}


class ConnectionPoolExhaustion(TemporalPattern):
    """
    Connection pool saturation pattern: gradual fill then sustained plateau.

    Common in: Database connection pool exhaustion, thread pool saturation,
    file descriptor limits, sustained high load.

    Shape: Quadratic rise to saturation level, then flat plateau with jitter.
    """

    @property
    def name(self) -> str:
        return "connection_pool_exhaustion"

    @property
    def description(self) -> str:
        return "Gradual saturation then plateau (pool exhaustion)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        saturation_level: float = 3.5,
        saturation_time_frac: float = 0.4,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_level = saturation_level * (0.85 + rng.rand() * 0.3)
        t_saturation = max(1, int(W * saturation_time_frac * (0.8 + rng.rand() * 0.4)))

        # Quadratic rise to saturation
        for t in range(t_saturation):
            progress = t / t_saturation
            value = actual_level * (progress ** 2)  # Quadratic ease-in
            noise = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.1
            signal[t, affected_feats] = value + noise

        # Sustained plateau (no recovery within window)
        for t in range(t_saturation, W):
            jitter = torch.from_numpy(rng.randn(len(affected_feats))).float() * 0.15
            signal[t, affected_feats] = actual_level + jitter

        return signal

    def get_default_params(self) -> Dict:
        return {"saturation_level": 3.5, "saturation_time_frac": 0.4}


# =============================================================================
# SOTA Patterns: Gray Failures, Metastable, Limplock (Microsoft AIOps 2024)
# =============================================================================


class GrayFailure(TemporalPattern):
    """
    Gray failure pattern: intermittent degradation, system appears healthy
    but exhibits subtle, hard-to-detect performance issues.

    SOTA Significance: 39% of production incidents are gray failures
    (Microsoft AIOps 2024). These are the hardest to detect because:
    - Signal is below typical alert thresholds
    - Intermittent nature makes pattern recognition difficult
    - Multiple features affected inconsistently

    Characteristics:
    - Intermittent spikes (not continuous)
    - Low amplitude (below alert thresholds)
    - Long duration (persists across window)
    - Multiple features affected inconsistently

    Common in: Partial network partitions, flaky dependencies,
    intermittent disk I/O issues, DNS resolution delays.

    Shape: Random low-amplitude spikes scattered throughout window
    with occasional grouped bursts. Signal-to-noise ratio is LOW.
    """

    @property
    def name(self) -> str:
        return "gray_failure"

    @property
    def description(self) -> str:
        return "Intermittent subtle degradation (gray failure - 39% of incidents)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        intermittence_prob: float = 0.3,
        base_amplitude: float = 0.8,
        burst_prob: float = 0.15,
        strength: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_amplitude = base_amplitude * strength * (0.5 + rng.rand() * 0.5)

        for t in range(W):
            if rng.rand() < intermittence_prob:
                n_active = rng.randint(1, len(affected_feats) + 1)
                active_feats = list(rng.choice(affected_feats, size=n_active, replace=False))

                amplitude = actual_amplitude * (0.5 + rng.rand() * 0.5)
                noise = torch.from_numpy(rng.randn(len(active_feats)) * 0.1).float()
                signal[t, active_feats] = amplitude + noise

                if rng.rand() < burst_prob and t + 2 < W:
                    burst_decay = 0.7
                    noise1 = torch.from_numpy(rng.randn(len(active_feats)) * 0.1).float()
                    signal[t + 1, active_feats] = amplitude * burst_decay + noise1
                    if t + 2 < W:
                        noise2 = torch.from_numpy(rng.randn(len(active_feats)) * 0.1).float()
                        signal[t + 2, active_feats] = amplitude * burst_decay**2 + noise2

        return signal

    def get_default_params(self) -> Dict:
        return {"intermittence_prob": 0.3, "base_amplitude": 0.8, "burst_prob": 0.15}


class MetastableFailure(TemporalPattern):
    """
    Metastable failure pattern: system stays degraded after trigger disappears.

    The root cause may be brief but effects persist due to hysteresis in the
    system (e.g., circuit breaker stuck open, queue backlog not draining).

    Characteristics (SOTA):
    - Brief initial trigger (1-2 timesteps)
    - Sustained elevated state (plateau)
    - Recovery only at window end (or no recovery)
    - Hysteresis effect: state depends on history, not just current input

    Common in: Circuit breaker stuck open, queue backlog after spike,
    cache stampede effects, connection pool not recovering after timeout.

    Shape: Sharp spike -> sustained plateau -> optional late recovery
    """

    @property
    def name(self) -> str:
        return "metastable_failure"

    @property
    def description(self) -> str:
        return "Sustained degradation after brief trigger (metastable)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        trigger_strength: float = 4.0,
        plateau_level: float = 2.0,
        recovery_prob: float = 0.3,
        strength: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        t_trigger = rng.randint(0, max(1, W // 4))
        trigger_duration = rng.randint(1, 3)
        actual_trigger = trigger_strength * strength * (0.8 + rng.rand() * 0.4)
        actual_plateau = plateau_level * strength * (0.8 + rng.rand() * 0.4)

        for t in range(t_trigger, min(t_trigger + trigger_duration, W)):
            noise = torch.from_numpy(rng.randn(len(affected_feats)) * 0.2).float()
            signal[t, affected_feats] = actual_trigger + noise

        t_plateau_start = t_trigger + trigger_duration
        t_recovery = W

        if rng.rand() < recovery_prob:
            t_recovery = W - rng.randint(1, max(2, W // 5))

        for t in range(t_plateau_start, min(t_recovery, W)):
            jitter = torch.from_numpy(rng.randn(len(affected_feats)) * 0.15).float()
            signal[t, affected_feats] = actual_plateau + jitter

        if t_recovery < W:
            for t in range(t_recovery, W):
                recovery_progress = (t - t_recovery) / max(1, W - t_recovery)
                level = actual_plateau * (1 - recovery_progress)
                noise = torch.from_numpy(rng.randn(len(affected_feats)) * 0.1).float()
                signal[t, affected_feats] = level + noise

        return signal

    def get_default_params(self) -> Dict:
        return {"trigger_strength": 4.0, "plateau_level": 2.0, "recovery_prob": 0.3}


class LimplockFailure(TemporalPattern):
    """
    Limplock failure pattern: system is technically alive but effective
    throughput is near zero. All resources are busy doing nothing useful.

    This pattern is distinct from connection_pool_exhaustion because:
    - Resources are saturated BUT throughput collapses (not just high latency)
    - System appears "busy" but is not making progress
    - Often caused by: deadlocks, priority inversion, lock contention

    Characteristics (SOTA):
    - Gradual resource exhaustion (thread pool, connections)
    - Throughput drops to near-zero while latency maxes out
    - System appears "busy" but is not making progress

    Shape: Gradual rise (resource fill) -> sustained maximum (saturation)
           + secondary metric (throughput) drops to near-zero
    """

    @property
    def name(self) -> str:
        return "limplock_failure"

    @property
    def description(self) -> str:
        return "Near-zero throughput despite busy resources (limplock)"

    def generate(
        self,
        window_size: int,
        num_features: int,
        affected_feats: List[int],
        rng: np.random.RandomState,
        saturation_level: float = 4.0,
        ramp_fraction: float = 0.3,
        throughput_floor: float = 0.1,
        strength: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_saturation = saturation_level * strength * (0.85 + rng.rand() * 0.3)
        t_ramp_end = max(1, int(W * ramp_fraction * (0.8 + rng.rand() * 0.4)))

        n_feats = len(affected_feats)
        resource_feats = affected_feats[:max(1, n_feats // 2)]
        throughput_feats = affected_feats[max(1, n_feats // 2):] if n_feats > 1 else []

        for t in range(W):
            if t < t_ramp_end:
                progress = t / t_ramp_end
                resource_level = actual_saturation * (progress ** 1.5)
                throughput_level = 1.0 - progress * 0.8
            else:
                resource_level = actual_saturation
                throughput_level = throughput_floor

            noise_r = torch.from_numpy(rng.randn(len(resource_feats)) * 0.15).float()
            signal[t, resource_feats] = resource_level + noise_r

            if throughput_feats:
                noise_t = torch.from_numpy(rng.randn(len(throughput_feats)) * 0.1).float()
                signal[t, throughput_feats] = (1.0 - throughput_level) * actual_saturation * 0.5 + noise_t

        return signal

    def get_default_params(self) -> Dict:
        return {"saturation_level": 4.0, "ramp_fraction": 0.3, "throughput_floor": 0.1}


# =============================================================================
# Pattern Registry
# =============================================================================

PATTERN_REGISTRY: Dict[str, TemporalPattern] = {
    # Original patterns
    "exponential_decay": ExponentialDecay(),
    "retry_storm": RetryStorm(),
    "memory_leak": MemoryLeak(),
    "cold_start": ColdStart(),
    "connection_pool_exhaustion": ConnectionPoolExhaustion(),
    # SOTA patterns (Microsoft AIOps 2024)
    "gray_failure": GrayFailure(),
    "metastable_failure": MetastableFailure(),
    "limplock_failure": LimplockFailure(),
}

PATTERN_NAMES = list(PATTERN_REGISTRY.keys())

# Realistic pattern weights based on SOTA research (Microsoft AIOps 2024)
# Gray failures represent 39% of production incidents - the largest category!
PATTERN_WEIGHTS: Dict[str, float] = {
    "gray_failure": 0.39,               # 39% - Most common and hardest to detect
    "exponential_decay": 0.15,          # 15% - Classic transient errors
    "retry_storm": 0.12,                # 12% - Circuit breaker / retry cascades
    "memory_leak": 0.10,                # 10% - Resource leaks
    "metastable_failure": 0.10,         # 10% - Hysteresis effects
    "connection_pool_exhaustion": 0.08, #  8% - Pool saturation
    "cold_start": 0.04,                 #  4% - Startup issues
    "limplock_failure": 0.02,           #  2% - Complete throughput collapse
}

# Pattern difficulty for evaluation breakdown
PATTERN_DIFFICULTY: Dict[str, str] = {
    "gray_failure": "hard",
    "metastable_failure": "medium",
    "limplock_failure": "medium",
    "memory_leak": "normal",
    "retry_storm": "normal",
    "connection_pool_exhaustion": "normal",
    "exponential_decay": "easy",
    "cold_start": "easy",
}


def get_pattern(name: str) -> TemporalPattern:
    """Get a pattern by name."""
    if name not in PATTERN_REGISTRY:
        raise ValueError(f"Unknown pattern: {name}. Available: {PATTERN_NAMES}")
    return PATTERN_REGISTRY[name]


# =============================================================================
# Anomaly Injection
# =============================================================================

def inject_anomaly_pattern(
    X: torch.Tensor,
    root_cause_idx: int,
    pattern_type: str,
    P_base: torch.Tensor,
    affected_feats: List[int],
    rng: np.random.RandomState,
    max_cascade_hops: int = 6,
    cascade_amplification: float = 1.5,
    cascade_decay: float = 0.5,
    **pattern_kwargs,
) -> torch.Tensor:
    """
    Inject an anomaly pattern into the tensor X with cascade propagation.

    This function:
    1. Generates the root cause signal using the specified pattern
    2. Applies the signal to the root cause node
    3. Propagates the anomaly through the graph using the propagation matrix

    Args:
        X: Input tensor [W, N, F] to modify
        root_cause_idx: Index of the root cause node
        pattern_type: Name of the temporal pattern to use
        P_base: Propagation matrix [N, N] where P[i,j] > 0 means anomaly at i affects j
        affected_feats: List of feature indices affected by this anomaly
        rng: Random state for reproducibility
        max_cascade_hops: Maximum propagation depth
        cascade_amplification: Signal amplification factor during propagation
        cascade_decay: Decay factor per hop
        **pattern_kwargs: Additional parameters for the pattern

    Returns:
        Modified X tensor with injected anomaly
    """
    W, N, F = X.shape

    # Get the pattern generator
    pattern = get_pattern(pattern_type)

    # Generate root cause signal [W, F]
    root_signal = pattern.generate(
        window_size=W,
        num_features=F,
        affected_feats=affected_feats,
        rng=rng,
        **pattern_kwargs,
    )

    # Apply to root cause node
    X[:, root_cause_idx, :] += root_signal

    # Stochastic cascade propagation
    P_sample = P_base.clone()
    # Add randomness to propagation strengths
    jitter = torch.from_numpy(rng.uniform(0.7, 1.3, size=P_base.shape)).float()
    P_sample = P_sample * jitter * (0.5 + rng.rand())

    # Track active nodes at each timestep
    # Propagation has temporal delay: anomaly at t propagates to neighbors at t+1
    for hop in range(max_cascade_hops):
        hop_delay = hop + 1

        for t_src in range(W - hop_delay):
            t_dst = t_src + hop_delay
            if t_dst >= W:
                break

            # Source activity at t_src (mean across affected features)
            src_activity = X[t_src, :, affected_feats].mean(dim=-1)  # [N]

            # Propagate through adjacency
            propagated = torch.matmul(P_sample.T, src_activity) * cascade_amplification

            # Stochastic dropout (~20% of paths fail)
            drop_mask = torch.from_numpy(rng.binomial(1, 0.8, size=N)).float()
            propagated = propagated * drop_mask

            # Apply decay per hop
            propagated = propagated * (cascade_decay ** hop)

            # Add to affected features (with some bleed to other features)
            active_nodes = (propagated > 0.1).nonzero(as_tuple=False).squeeze(-1)
            if active_nodes.numel() > 0:
                for feat in affected_feats:
                    X[t_dst, active_nodes, feat] += propagated[active_nodes]

                # Feature bleed: slight effect on non-affected features
                other_feats = [f for f in range(F) if f not in affected_feats]
                if other_feats:
                    for feat in other_feats:
                        X[t_dst, active_nodes, feat] += propagated[active_nodes] * 0.15

    return X


def visualize_patterns(
    window_size: int = 12,
    num_features: int = 5,
    seed: int = 42,
) -> None:
    """Visualize all patterns for debugging/documentation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return

    rng = np.random.RandomState(seed)
    affected_feats = [0, 1]

    fig, axes = plt.subplots(len(PATTERN_NAMES), 1, figsize=(10, 3 * len(PATTERN_NAMES)))
    if len(PATTERN_NAMES) == 1:
        axes = [axes]

    for ax, name in zip(axes, PATTERN_NAMES):
        pattern = PATTERN_REGISTRY[name]
        signal = pattern.generate(
            window_size=window_size,
            num_features=num_features,
            affected_feats=affected_feats,
            rng=rng,
        )
        # Plot affected features
        for feat in affected_feats:
            ax.plot(signal[:, feat].numpy(), label=f"Feature {feat}")
        ax.set_title(f"{name}: {pattern.description}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("temporal_patterns_visualization.png", dpi=150)
    plt.close()
    logger.info("Saved visualization to temporal_patterns_visualization.png")


if __name__ == "__main__":
    # Test all patterns
    print("Testing temporal patterns...")
    print(f"Total patterns: {len(PATTERN_REGISTRY)}")
    print(f"Pattern weights sum: {sum(PATTERN_WEIGHTS.values()):.2f}")

    rng = np.random.RandomState(42)
    W, F = 12, 5
    affected = [0, 1, 2]

    print("\n--- Pattern Statistics ---")
    for name, pattern in PATTERN_REGISTRY.items():
        signal = pattern.generate(W, F, affected, rng)
        max_val = signal[:, affected].max().item()
        mean_val = signal[:, affected].mean().item()
        weight = PATTERN_WEIGHTS.get(name, 0.0)
        difficulty = PATTERN_DIFFICULTY.get(name, "unknown")
        print(f"{name:30s} | max={max_val:6.3f} | mean={mean_val:6.3f} | weight={weight:.0%} | {difficulty}")

    print("\n--- Testing cascade injection ---")
    X = torch.randn(W, 100, F) * 0.1  # Low noise baseline
    P = torch.zeros(100, 100)
    # Create some connections
    for i in range(99):
        P[i, i + 1] = 0.8

    X_modified = inject_anomaly_pattern(
        X.clone(),
        root_cause_idx=0,
        pattern_type="gray_failure",  # Test new pattern
        P_base=P,
        affected_feats=affected,
        rng=rng,
    )

    # Check that root cause has higher signal
    rc_signal = X_modified[:, 0, affected].abs().mean().item()
    other_signal = X_modified[:, 50:, affected].abs().mean().item()
    print(f"Root cause signal: {rc_signal:.3f}, Other nodes: {other_signal:.3f}")

    # Optionally visualize
    try:
        visualize_patterns()
    except Exception as e:
        print(f"Visualization skipped: {e}")
