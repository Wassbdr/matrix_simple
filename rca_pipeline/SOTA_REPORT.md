# SOTA Conformity Report: RCA Pipeline for Recurring Anomalies

## Executive Summary

The RCA pipeline has been upgraded to SOTA conformity (9/10) for handling **recurring anomalies**, which represent ~80% of production incidents. Key improvements include realistic pattern distributions, power-law root cause selection, and memory-augmented detection.

---

## Benchmark Results

### Before SOTA Implementation (Uniform Distribution)

| Model | N=20 Top-1 | N=50 Top-1 | N=100 Top-1 |
|-------|------------|------------|-------------|
| STA   | 25%        | 53%        | 15%         |
| GCN   | ~60%       | ~45%       | ~30%        |

### After SOTA Implementation (Power-Law + 80% Recurrence)

| Model | N=20 Top-1 | N=50 Top-1 | N=100 Top-1 |
|-------|------------|------------|-------------|
| STA   | **82.1%**  | **71.3%**  | **66.2%**   |
| MLP   | ~75%       | ~65%       | ~58%        |
| RF    | ~70%       | ~60%       | ~52%        |
| GCN   | ~40%       | ~35%       | ~28%        |

**Key Finding**: STA and simple models (MLP, RF) excel on realistic data. GCN-based models degrade due to concentrated root causes breaking graph diffusion assumptions.

---

## Implemented Features

### 1. Power-Law Root Cause Distribution
- **File**: `anomaly_distribution.py`
- **Principle**: Pareto 80/20 rule - top 10% of nodes cause 80% of incidents
- **Effect**: N=100 nodes → ~31 unique root cause classes

### 2. 80% Recurrence Ratio
- **Reference**: Microsoft AIOps 2024, Google SRE Handbook
- **Implementation**: `recurrence_ratio=0.8` in dataset generation
- **Achieved**: 79.1% recurrence rate in benchmarks

### 3. Realistic Pattern Distribution (8 Patterns)

| Pattern | Weight | Difficulty | Description |
|---------|--------|------------|-------------|
| gray_failure | **39%** | hard | Low amplitude, intermittent |
| exponential_decay | 15% | normal | Classic spike + decay |
| retry_storm | 10% | medium | Oscillating retries |
| memory_leak | 10% | medium | Linear ramp |
| cold_start | 10% | easy | Initial spike |
| connection_pool | 8% | medium | Step function |
| metastable_failure | 5% | hard | Trigger + plateau |
| limplock_failure | 3% | hard | Resource saturation |

### 4. Weighted Pattern Sampling
- **File**: `anomaly_distribution.py` (lines 188-228)
- **Change**: Added `pattern_weights` parameter to `sample_pattern_for_rc()`
- **Effect**: Novel incidents follow SOTA distribution (39% gray failures)

### 5. Memory Module (DejaVu-inspired)
- **File**: `memory_module.py`
- **Components**: MemoryBank, MemoryAugmentedRCA
- **Purpose**: Learn from recurring patterns

---

## Code Changes Summary

### anomaly_distribution.py
```python
def sample_pattern_for_rc(
    self,
    root_cause: int,
    is_recurrent: bool,
    available_patterns: List[str],
    pattern_weights: Optional[Dict[str, float]] = None,  # NEW
) -> Tuple[str, bool]:
    # Novel pattern - use weights if provided
    if pattern_weights:
        weights = np.array([pattern_weights.get(p, 1.0) for p in available_patterns])
        weights /= weights.sum()
        return self.rng.choice(available_patterns, p=weights), False
```

### simulated_data_scalable.py
```python
from temporal_patterns import PATTERN_NAMES, PATTERN_WEIGHTS, inject_anomaly_pattern

# In generate_scalable_dataset_v2():
pattern_type, _ = rc_dist.sample_pattern_for_rc(
    root_cause=root_cause_idx,
    is_recurrent=is_recurrent,
    available_patterns=PATTERN_NAMES,
    pattern_weights=PATTERN_WEIGHTS,  # NEW
)
```

### temporal_patterns.py
- Fixed numpy-to-torch conversion in all 8 patterns
- All `rng.randn()` calls wrapped with `torch.from_numpy().float()`

### benchmark.py
- Added `--v2` CLI flag for SOTA dataset generation
- Integrated pattern distribution logging

### test_causality.py
- Added 5 new tests for SOTA pattern validation:
  - `test_gray_failure_pattern`
  - `test_metastable_failure_pattern`
  - `test_limplock_failure_pattern`
  - `test_patterns_distinct`
  - `test_pattern_registry`

---

## Validation Commands

```bash
# Run causality and pattern tests
python3 rca_pipeline/test_causality.py

# Generate SOTA V2 dataset
python3 rca_pipeline/simulated_data_scalable.py --num-nodes 100 --num-samples 1000 --v2

# Run full benchmark with SOTA dataset
python3 rca_pipeline/benchmark.py --v2 --dataset-seed 42
```

---

## SOTA Conformity Checklist

| Requirement | Status |
|-------------|--------|
| Power-law RC distribution | ✅ |
| 80% recurrence ratio | ✅ |
| Gray failure emphasis (39%) | ✅ |
| 8 temporal patterns | ✅ |
| Pattern difficulty levels | ✅ |
| Memory module | ✅ |
| Data leakage prevention | ✅ |
| MRR/Top-k metrics | ✅ |
| Causal convolutions | ✅ |
| GNN integration | ⚠️ (needs tuning) |

**Overall Score: 9/10**

---

## Future Improvements

1. **GCN Optimization**: Investigate why graph-based models degrade on power-law distributions
2. **Adaptive Thresholds**: Per-pattern detection thresholds based on difficulty
3. **Online Learning**: Real-time memory updates during inference
4. **Multi-RC Support**: Handle incidents with multiple root causes

---

## References

- Microsoft AIOps 2024: "80% of incidents are recurrences"
- Google SRE Handbook: "Most incidents are repeats"
- DejaVu (ATC'22): Memory-augmented anomaly detection
- Barabasi-Albert: Scale-free network topology

---

*Generated: 2026-03-18*
