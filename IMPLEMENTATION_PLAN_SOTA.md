# Plan d'Implementation SOTA - Anomalies Recurrentes

## Contexte

Le pipeline actuel atteint **7/10** de conformite SOTA. Ce plan detaille les etapes pour atteindre **9-10/10**.

### Etat Actuel (Deja Implemente)
- [x] Distribution Power-Law des root causes (Pareto 80/20)
- [x] Ratio de recurrence 80% (anomalies repetitives)
- [x] 5 patterns temporels (exponential_decay, retry_storm, memory_leak, cold_start, connection_pool_exhaustion)
- [x] Module de memoire (MemoryBank + MemoryAugmentedRCA)
- [x] Metriques SOTA (MRR, Top@k, P@k, SW-MRR, breakdown)
- [x] Correction data leakage (same-window augmentation)
- [x] Losses multi-root et recurrence

### Gaps Identifies (A Implementer)
- [ ] Gray Failures (39% des incidents en production!)
- [ ] Metastable Failures
- [ ] Limplock Failures
- [ ] Scoring de severite base sur SLI impact
- [ ] Support multi-modal (logs + traces) - Roadmap

---

## Phase 1: Patterns de Defaillance Manquants (Priorite P0)

### 1.1 Gray Failure Pattern

**Source SOTA**: 39% des incidents de production sont des gray failures (Microsoft AIOps 2024)

**Definition**: Defaillance partielle ou non-deterministe ou le systeme continue de fonctionner mais avec une degradation subtile des performances. Le systeme n'est ni "up" ni "down" - il est "gray".

**Fichier**: `rca_pipeline/temporal_patterns.py`

```python
class GrayFailure(TemporalPattern):
    """
    Gray failure pattern: intermittent degradation, system appears healthy
    but exhibits subtle, hard-to-detect performance issues.

    Characteristics (SOTA):
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
        intermittence_prob: float = 0.3,  # Only 30% of timesteps show anomaly
        base_amplitude: float = 0.8,       # LOW amplitude (below typical thresholds)
        burst_prob: float = 0.15,          # 15% chance of burst
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        # Intermittent degradation pattern
        for t in range(W):
            if rng.rand() < intermittence_prob:
                # Random subset of affected features show degradation
                n_active = rng.randint(1, len(affected_feats) + 1)
                active_feats = rng.choice(affected_feats, size=n_active, replace=False)

                # Low amplitude spike (hard to detect)
                amplitude = base_amplitude * (0.5 + rng.rand() * 0.5)
                noise = rng.randn(len(active_feats)) * 0.1
                signal[t, active_feats] = amplitude + noise

                # Occasional burst (correlated spikes)
                if rng.rand() < burst_prob and t + 2 < W:
                    burst_decay = 0.7
                    signal[t + 1, active_feats] = amplitude * burst_decay + rng.randn(len(active_feats)) * 0.1
                    signal[t + 2, active_feats] = amplitude * burst_decay**2 + rng.randn(len(active_feats)) * 0.1

        return signal
```

**Validation**:
- Test: signal.max() < 2.0 (amplitude basse)
- Test: (signal > 0.1).sum() / signal.numel() < 0.4 (intermittent)
- Metrique: Le modele doit avoir MRR plus bas sur gray failures (plus difficile)

---

### 1.2 Metastable Failure Pattern

**Definition**: Le systeme reste degrade meme apres que le trigger initial a disparu. La cause racine peut etre breve mais les effets persistent.

**Fichier**: `rca_pipeline/temporal_patterns.py`

```python
class MetastableFailure(TemporalPattern):
    """
    Metastable failure pattern: system stays degraded after trigger disappears.

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
        recovery_prob: float = 0.3,  # Only 30% recover within window
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        # Brief trigger (1-2 timesteps at start)
        t_trigger = rng.randint(0, max(1, W // 4))
        trigger_duration = rng.randint(1, 3)
        actual_trigger = trigger_strength * (0.8 + rng.rand() * 0.4)
        actual_plateau = plateau_level * (0.8 + rng.rand() * 0.4)

        # Trigger phase
        for t in range(t_trigger, min(t_trigger + trigger_duration, W)):
            noise = rng.randn(len(affected_feats)) * 0.2
            signal[t, affected_feats] = actual_trigger + noise

        # Sustained plateau phase
        t_plateau_start = t_trigger + trigger_duration
        t_recovery = W  # Default: no recovery

        if rng.rand() < recovery_prob:
            # Recovery in last 20% of window
            t_recovery = W - rng.randint(1, max(2, W // 5))

        for t in range(t_plateau_start, t_recovery):
            if t >= W:
                break
            jitter = rng.randn(len(affected_feats)) * 0.15
            signal[t, affected_feats] = actual_plateau + jitter

        # Recovery phase (if applicable)
        if t_recovery < W:
            for t in range(t_recovery, W):
                # Exponential recovery
                recovery_progress = (t - t_recovery) / max(1, W - t_recovery)
                level = actual_plateau * (1 - recovery_progress)
                signal[t, affected_feats] = level + rng.randn(len(affected_feats)) * 0.1

        return signal
```

**Validation**:
- Test: Plateau detecte (ecart-type faible sur t > trigger)
- Test: Trigger bref (< 20% de la fenetre)
- Metrique: Correlation temporelle elevee (signal persistant)

---

### 1.3 Limplock Failure Pattern

**Definition**: Throughput effectif proche de zero malgre un systeme techniquement "up". Toutes les ressources sont occupees mais aucun travail utile n'est accompli.

**Fichier**: `rca_pipeline/temporal_patterns.py`

```python
class LimplockFailure(TemporalPattern):
    """
    Limplock failure pattern: system is technically alive but effective
    throughput is near zero. All resources are busy doing nothing useful.

    Characteristics (SOTA):
    - Gradual resource exhaustion (thread pool, connections)
    - Throughput drops to near-zero while latency maxes out
    - System appears "busy" but is not making progress
    - Often caused by: deadlocks, priority inversion, lock contention

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
        ramp_fraction: float = 0.3,  # 30% of window for ramp
        throughput_floor: float = 0.1,  # Near-zero throughput
        **kwargs,
    ) -> torch.Tensor:
        W, F = window_size, num_features
        signal = torch.zeros(W, F)

        actual_saturation = saturation_level * (0.85 + rng.rand() * 0.3)
        t_ramp_end = max(1, int(W * ramp_fraction * (0.8 + rng.rand() * 0.4)))

        # Split affected features: half for "resource usage", half for "throughput"
        n_feats = len(affected_feats)
        resource_feats = affected_feats[:max(1, n_feats // 2)]
        throughput_feats = affected_feats[max(1, n_feats // 2):] if n_feats > 1 else []

        for t in range(W):
            if t < t_ramp_end:
                # Ramp phase: resources filling up
                progress = t / t_ramp_end
                resource_level = actual_saturation * (progress ** 1.5)  # Slightly convex
                throughput_level = 1.0 - progress * 0.8  # Throughput dropping
            else:
                # Saturation phase: maxed out, no progress
                resource_level = actual_saturation
                throughput_level = throughput_floor

            # Resource features (high)
            noise_r = rng.randn(len(resource_feats)) * 0.15
            signal[t, resource_feats] = resource_level + noise_r

            # Throughput features (low) - represented as negative deviation
            if throughput_feats:
                noise_t = rng.randn(len(throughput_feats)) * 0.1
                # Invert: low throughput = high anomaly signal
                signal[t, throughput_feats] = (1.0 - throughput_level) * actual_saturation * 0.5 + noise_t

        return signal
```

**Validation**:
- Test: resource_feats.mean() > 3.0 en phase saturation
- Test: throughput inverse au resource usage
- Metrique: Pattern distinctif de connection_pool_exhaustion (throughput vs juste saturation)

---

## Phase 2: Integration dans le Registry (Priorite P0)

**Fichier**: `rca_pipeline/temporal_patterns.py`

```python
# Ajouter au PATTERN_REGISTRY existant

PATTERN_REGISTRY: Dict[str, TemporalPattern] = {
    # Existants
    "exponential_decay": ExponentialDecay(),
    "retry_storm": RetryStorm(),
    "memory_leak": MemoryLeak(),
    "cold_start": ColdStart(),
    "connection_pool_exhaustion": ConnectionPoolExhaustion(),
    # NOUVEAUX (SOTA)
    "gray_failure": GrayFailure(),
    "metastable_failure": MetastableFailure(),
    "limplock_failure": LimplockFailure(),
}

# Metadata pour ponderation realiste
PATTERN_WEIGHTS = {
    "gray_failure": 0.39,              # 39% des incidents (SOTA)
    "exponential_decay": 0.15,
    "retry_storm": 0.12,
    "memory_leak": 0.10,
    "metastable_failure": 0.10,
    "connection_pool_exhaustion": 0.08,
    "cold_start": 0.04,
    "limplock_failure": 0.02,
}
```

---

## Phase 3: Mise a Jour du Generateur de Donnees (Priorite P1)

**Fichier**: `rca_pipeline/simulated_data_scalable.py`

### 3.1 Distribution Realiste des Patterns

```python
def generate_scalable_dataset_v3(
    # ... params existants ...
    use_realistic_pattern_weights: bool = True,  # NOUVEAU
    gray_failure_snr_reduction: float = 0.5,     # NOUVEAU: SNR plus bas pour gray
):
    from temporal_patterns import PATTERN_REGISTRY, PATTERN_NAMES, PATTERN_WEIGHTS

    # ...

    if is_anomaly:
        if use_realistic_pattern_weights:
            # Ponderer par frequence reelle (39% gray failures!)
            pattern_probs = [PATTERN_WEIGHTS.get(p, 0.1) for p in PATTERN_NAMES]
            pattern_probs = np.array(pattern_probs) / sum(pattern_probs)
            pattern_type = rng.choice(PATTERN_NAMES, p=pattern_probs)
        else:
            pattern_type = rng.choice(PATTERN_NAMES)

        # Ajuster SNR pour gray failures (plus difficiles a detecter)
        if pattern_type == "gray_failure":
            signal_strength *= gray_failure_snr_reduction
```

### 3.2 Metadata Enrichie

```python
data.pattern_difficulty = {
    "gray_failure": "hard",
    "metastable_failure": "medium",
    "limplock_failure": "medium",
}.get(pattern_type, "normal")

data.sli_impact = compute_sli_impact(
    pattern_type=pattern_type,
    signal_strength=signal_strength,
    cascade_depth=cascade_depth,
    is_gray=pattern_type == "gray_failure",
)
```

---

## Phase 4: Scoring de Severite Base SLI (Priorite P1)

**Fichier**: `rca_pipeline/simulated_data_scalable.py`

```python
def compute_sli_impact(
    pattern_type: str,
    signal_strength: float,
    cascade_depth: int,
    is_gray: bool,
    N: int = 100,
) -> Dict[str, float]:
    """
    Calcule l'impact SLI selon les categories SOTA:
    - availability: % de temps ou le service est up
    - latency: p99 latency deviation
    - error_rate: % de requetes en erreur
    - throughput: % de capacite nominale
    """
    base_impact = min(signal_strength / 5.0, 1.0)
    depth_factor = min(cascade_depth / 5.0, 1.0)

    if pattern_type == "gray_failure":
        # Gray: Faible impact apparent mais reel
        return {
            "availability": 0.95 + 0.05 * (1 - base_impact * 0.5),  # ~95-100%
            "latency": base_impact * 0.6,      # Latency degradee mais pas catastrophique
            "error_rate": base_impact * 0.15,  # Peu d'erreurs visibles
            "throughput": 1.0 - base_impact * 0.3,  # Throughput reduit subtilement
            "detection_difficulty": "hard",
        }
    elif pattern_type == "metastable_failure":
        # Metastable: Impact prolonge
        return {
            "availability": 0.85 + 0.15 * (1 - base_impact),
            "latency": base_impact * 0.8,
            "error_rate": base_impact * 0.4,
            "throughput": 1.0 - base_impact * 0.5,
            "detection_difficulty": "medium",
        }
    elif pattern_type == "limplock_failure":
        # Limplock: Throughput effondre
        return {
            "availability": 0.99,  # Techniquement up
            "latency": base_impact,  # Latency max
            "error_rate": base_impact * 0.3,
            "throughput": 0.1,  # Pres de zero!
            "detection_difficulty": "medium",
        }
    else:
        # Patterns existants
        return {
            "availability": max(0.5, 1.0 - base_impact * 0.5 - depth_factor * 0.2),
            "latency": base_impact * (1 + depth_factor * 0.5),
            "error_rate": base_impact * 0.5,
            "throughput": max(0.2, 1.0 - base_impact * 0.4),
            "detection_difficulty": "normal",
        }
```

---

## Phase 5: Tests de Validation (Priorite P0)

**Fichier**: `rca_pipeline/test_causality.py` (nouveau)

```python
import pytest
import torch
import numpy as np
from temporal_patterns import (
    PATTERN_REGISTRY, GrayFailure, MetastableFailure, LimplockFailure
)


class TestGrayFailure:
    def test_low_amplitude(self):
        """Gray failures should have low signal amplitude."""
        pattern = GrayFailure()
        rng = np.random.RandomState(42)
        signal = pattern.generate(12, 5, [0, 1, 2], rng)

        assert signal.max() < 2.5, "Gray failure amplitude should be low"
        assert signal.mean() < 0.5, "Gray failure mean should be low"

    def test_intermittence(self):
        """Gray failures should be intermittent, not continuous."""
        pattern = GrayFailure()
        rng = np.random.RandomState(42)
        signal = pattern.generate(12, 5, [0, 1, 2], rng)

        active_ratio = (signal.abs() > 0.1).float().mean()
        assert active_ratio < 0.5, "Gray failure should be intermittent"


class TestMetastableFailure:
    def test_plateau_persistence(self):
        """Metastable should show sustained plateau after trigger."""
        pattern = MetastableFailure()
        rng = np.random.RandomState(42)
        signal = pattern.generate(12, 5, [0, 1, 2], rng)

        # Check plateau (low variance in second half)
        second_half = signal[6:, [0, 1, 2]]
        plateau_variance = second_half.var()
        assert plateau_variance < 1.0, "Should have stable plateau"

    def test_trigger_brief(self):
        """Trigger should be brief (< 25% of window)."""
        pattern = MetastableFailure()
        rng = np.random.RandomState(42)

        # Multiple trials
        for _ in range(10):
            signal = pattern.generate(12, 5, [0, 1, 2], rng)
            max_t = signal.max(dim=1).values.argmax()
            assert max_t < 3, "Trigger should be in first 25% of window"


class TestLimplockFailure:
    def test_saturation_reached(self):
        """Resources should reach saturation."""
        pattern = LimplockFailure()
        rng = np.random.RandomState(42)
        signal = pattern.generate(12, 5, [0, 1, 2, 3], rng)

        # Last timesteps should be at saturation
        last_values = signal[-3:, [0, 1]].mean()
        assert last_values > 2.0, "Should reach saturation"


class TestPatternDistinguishability:
    def test_patterns_visually_distinct(self):
        """Each pattern should produce distinct temporal signatures."""
        rng = np.random.RandomState(42)
        signals = {}

        for name, pattern in PATTERN_REGISTRY.items():
            signals[name] = pattern.generate(12, 5, [0, 1, 2], rng)

        # Pairwise correlation should be moderate (not identical)
        for n1, s1 in signals.items():
            for n2, s2 in signals.items():
                if n1 != n2:
                    corr = torch.corrcoef(
                        torch.stack([s1.flatten(), s2.flatten()])
                    )[0, 1]
                    assert corr < 0.8, f"{n1} and {n2} should be distinct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Phase 6: Mise a Jour des Metriques (Priorite P1)

**Fichier**: `rca_pipeline/evaluate.py`

```python
def compute_breakdown(self) -> Dict[str, Dict[str, float]]:
    """Compute metrics breakdown by pattern type and recurrence."""

    # Breakdown par difficulte (NOUVEAU)
    by_difficulty = {
        "hard": {"mrr": [], "top_1": []},   # gray_failure
        "medium": {"mrr": [], "top_1": []}, # metastable, limplock
        "normal": {"mrr": [], "top_1": []}, # autres
    }

    for entry in self.entries:
        difficulty = entry.metadata.get("pattern_difficulty", "normal")
        mrr = self._compute_mrr_single(entry.logits, entry.target)
        top1 = (entry.logits.argmax() == entry.target).float().item()

        by_difficulty[difficulty]["mrr"].append(mrr)
        by_difficulty[difficulty]["top_1"].append(top1)

    # Aggreger
    result = {}
    for diff, metrics in by_difficulty.items():
        if metrics["mrr"]:
            result[f"{diff}_mrr"] = np.mean(metrics["mrr"])
            result[f"{diff}_top1"] = np.mean(metrics["top_1"])

    return result
```

---

## Phase 7: Roadmap Multi-Modal (Priorite P2 - Future)

### 7.1 Support Logs (Future)

```python
class LogFeatureExtractor:
    """Extract features from log streams for multi-modal RCA."""

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 32):
        self.tokenizer = ...  # Log-specific tokenizer
        self.encoder = nn.TransformerEncoder(...)

    def extract(self, log_lines: List[str]) -> torch.Tensor:
        # Tokenize -> Embed -> Temporal pooling
        pass
```

### 7.2 Support Traces (Future)

```python
class TraceFeatureExtractor:
    """Extract features from distributed traces."""

    def extract(self, trace: Dict) -> torch.Tensor:
        # Span duration anomalies
        # Service dependency patterns
        # Error propagation paths
        pass
```

---

## Calendrier d'Execution

| Phase | Tache | Priorite | Duree | Dependances |
|-------|-------|----------|-------|-------------|
| 1.1 | GrayFailure pattern | P0 | 1 jour | - |
| 1.2 | MetastableFailure pattern | P0 | 1 jour | - |
| 1.3 | LimplockFailure pattern | P0 | 1 jour | - |
| 2 | Integration Registry | P0 | 0.5 jour | 1.* |
| 3 | Dataset generator v3 | P1 | 1 jour | 2 |
| 4 | SLI scoring | P1 | 1 jour | 3 |
| 5 | Tests validation | P0 | 1 jour | 1.*, 2 |
| 6 | Metriques breakdown | P1 | 0.5 jour | 5 |
| 7 | Multi-modal (roadmap) | P2 | 5+ jours | 6 |

**Total Phase P0-P1**: ~6 jours

---

## Criteres de Succes

| Metrique | Actuel | Cible |
|----------|--------|-------|
| Patterns temporels | 5 | 8 |
| Gray failure support | Non | Oui (39%) |
| MRR global | ~0.75 | >0.75 |
| MRR gray failures | N/A | >0.55 |
| MRR patterns faciles | N/A | >0.85 |
| Top-1 global | ~65% | >65% |
| Conformite SOTA | 7/10 | 9/10 |

---

## Commandes de Validation

```bash
# 1. Generer dataset V3 avec nouveaux patterns
python rca_pipeline/simulated_data_scalable.py \
    --num-nodes 100 --num-samples 5000 --v2 \
    --use-realistic-pattern-weights

# 2. Executer tests
pytest rca_pipeline/test_causality.py -v

# 3. Entrainer et evaluer
python rca_pipeline/train_sta.py --dataset scalable_dataset_N100_v2.pt
python rca_pipeline/evaluate.py --model checkpoints/sta.pt --breakdown

# 4. Verifier breakdown par difficulte
# Attendu: hard_mrr < normal_mrr < easy_mrr
```
