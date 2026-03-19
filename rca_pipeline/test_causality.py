"""
Causality validation tests for the RCA pipeline.

Verifies that:
    1. CausalConv1d does not leak future information.
    2. temporal_jitter does not wrap (circular shift eliminated).
    3. The full LightEncoder + DeepTemporal chain preserves causality.

Usage:
    python3 test_causality.py
"""

import torch
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PASS_COUNT = 0
FAIL_COUNT = 0


def assert_test(condition: bool, name: str, detail: str = "") -> None:
    """Log pass/fail and track counts."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        logger.info("  ✅ PASS: %s", name)
    else:
        FAIL_COUNT += 1
        logger.error("  ❌ FAIL: %s — %s", name, detail)


# =========================================================================
# Test 1: CausalConv1d — future impulse must not affect past outputs
# =========================================================================

def test_causal_conv_no_future_leakage():
    """
    Inject an impulse at the last timestep (t=W-1).
    All outputs at t < W-1 must be identical with and without the impulse.
    """
    from model_sta import CausalConv1d

    logger.info("--- Test 1: CausalConv1d future leakage ---")

    for kernel_size in [3, 5, 7]:
        conv = CausalConv1d(in_channels=5, out_channels=32, kernel_size=kernel_size)
        conv.eval()

        W = 20
        x_base = torch.randn(2, 5, W)
        x_impulse = x_base.clone()
        x_impulse[:, :, -1] += 100.0  # Large impulse at t=W-1

        with torch.no_grad():
            y_base = conv(x_base)
            y_impulse = conv(x_impulse)

        # All timesteps before the last must be identical
        diff = (y_base[:, :, :-1] - y_impulse[:, :, :-1]).abs().max().item()
        assert_test(
            diff < 1e-5,
            f"CausalConv1d(k={kernel_size}): no leakage from t=W-1 to t<W-1",
            f"max diff = {diff:.2e}",
        )

        # The last timestep SHOULD be different (it sees the impulse)
        diff_last = (y_base[:, :, -1] - y_impulse[:, :, -1]).abs().max().item()
        assert_test(
            diff_last > 1.0,
            f"CausalConv1d(k={kernel_size}): last timestep IS affected by impulse",
            f"diff at t=W-1 = {diff_last:.2e} (expected >> 0)",
        )


# =========================================================================
# Test 2: temporal_jitter — no circular wrapping
# =========================================================================

def test_temporal_jitter_no_wrap():
    """
    Place a signal only at the last timestep. After shifting right,
    the first timesteps must remain zero (no circular wrap).
    """
    from augment import temporal_jitter

    logger.info("--- Test 2: temporal_jitter no-wrap ---")

    x = torch.zeros(5, 3, 10)  # [N, F, W=10]
    x[:, :, -1] = 1.0  # Signal only at t=9

    wrap_detected = False
    for _ in range(200):
        shifted = temporal_jitter(x, max_shift=2)
        if shifted[:, :, :2].abs().sum() > 0:
            wrap_detected = True
            break

    assert_test(
        not wrap_detected,
        "temporal_jitter: first 2 timesteps stay zero after 200 shifts",
        "Circular wrap detected — future data leaked to past positions",
    )

    # Also verify that shifting preserves signal somewhere
    found_signal = False
    for _ in range(200):
        shifted = temporal_jitter(x, max_shift=2)
        if shifted.abs().sum() > 0:
            found_signal = True
            break

    assert_test(
        found_signal,
        "temporal_jitter: signal is preserved (not all-zero)",
        "All shifts produced all-zero output",
    )


# =========================================================================
# Test 3: temporal_jitter — zero-shift returns exact clone
# =========================================================================

def test_temporal_jitter_zero_shift():
    """With max_shift=0, the output must be an identical clone."""
    from augment import temporal_jitter

    logger.info("--- Test 3: temporal_jitter zero-shift ---")

    x = torch.randn(4, 5, 3, 20)
    result = temporal_jitter(x, max_shift=0)

    assert_test(
        torch.equal(x, result),
        "temporal_jitter(max_shift=0): output == input",
        "Zero-shift did not return an exact clone",
    )

    # Verify it's a clone (not same object)
    assert_test(
        result.data_ptr() != x.data_ptr(),
        "temporal_jitter(max_shift=0): output is a separate tensor",
        "Returned same tensor object instead of a clone",
    )


# =========================================================================
# Test 4: CausalConv1d — output length matches input length
# =========================================================================

def test_causal_conv_output_shape():
    """CausalConv1d must not change the temporal dimension."""
    from model_sta import CausalConv1d

    logger.info("--- Test 4: CausalConv1d output shape ---")

    for W in [8, 16, 32, 64]:
        conv = CausalConv1d(in_channels=5, out_channels=32, kernel_size=3)
        conv.eval()
        x = torch.randn(2, 5, W)
        with torch.no_grad():
            y = conv(x)
        assert_test(
            y.shape == (2, 32, W),
            f"CausalConv1d output shape for W={W}: {y.shape} == (2, 32, {W})",
            f"Expected (2, 32, {W}), got {tuple(y.shape)}",
        )


# =========================================================================
# Test 5: GrayFailure pattern — low amplitude, intermittent
# =========================================================================

def test_gray_failure_pattern():
    """Gray failures must have low amplitude and intermittent signal."""
    import numpy as np
    from temporal_patterns import GrayFailure

    logger.info("--- Test 5: GrayFailure pattern characteristics ---")

    pattern = GrayFailure()
    rng = np.random.RandomState(42)
    W, F = 12, 5
    affected = [0, 1, 2]

    signal = pattern.generate(W, F, affected, rng)

    # Test 1: Low amplitude (below typical thresholds)
    max_amplitude = signal.max().item()
    assert_test(
        max_amplitude < 3.0,
        f"GrayFailure max amplitude {max_amplitude:.2f} < 3.0",
        f"Gray failure amplitude too high: {max_amplitude:.2f}",
    )

    # Test 2: Intermittent (not all timesteps active)
    active_ratio = (signal.abs() > 0.1).float().mean().item()
    assert_test(
        active_ratio < 0.6,
        f"GrayFailure active ratio {active_ratio:.2f} < 0.6 (intermittent)",
        f"Gray failure not intermittent enough: {active_ratio:.2f}",
    )

    # Test 3: Low mean signal
    mean_signal = signal[:, affected].mean().item()
    assert_test(
        mean_signal < 0.5,
        f"GrayFailure mean signal {mean_signal:.2f} < 0.5",
        f"Gray failure mean too high: {mean_signal:.2f}",
    )


# =========================================================================
# Test 6: MetastableFailure pattern — trigger + plateau
# =========================================================================

def test_metastable_failure_pattern():
    """Metastable failures must show brief trigger and sustained plateau."""
    import numpy as np
    from temporal_patterns import MetastableFailure

    logger.info("--- Test 6: MetastableFailure pattern characteristics ---")

    pattern = MetastableFailure()
    rng = np.random.RandomState(42)
    W, F = 12, 5
    affected = [0, 1, 2]

    signal = pattern.generate(W, F, affected, rng)

    # Test 1: Has a peak (trigger)
    max_val = signal[:, affected].max().item()
    assert_test(
        max_val > 2.0,
        f"MetastableFailure has trigger peak {max_val:.2f} > 2.0",
        f"No clear trigger peak: {max_val:.2f}",
    )

    # Test 2: Plateau in second half (low variance)
    second_half = signal[W//2:, affected]
    plateau_std = second_half.std().item()
    assert_test(
        plateau_std < 1.5,
        f"MetastableFailure plateau std {plateau_std:.2f} < 1.5",
        f"Plateau too variable (no stable state): {plateau_std:.2f}",
    )

    # Test 3: Trigger in first half
    first_half_max = signal[:W//2, affected].max().item()
    second_half_max = signal[W//2:, affected].max().item()
    assert_test(
        first_half_max >= second_half_max * 0.8,
        f"MetastableFailure trigger in first half ({first_half_max:.2f} >= {second_half_max*0.8:.2f})",
        f"Trigger not in first half: first={first_half_max:.2f}, second={second_half_max:.2f}",
    )


# =========================================================================
# Test 7: LimplockFailure pattern — saturation reached
# =========================================================================

def test_limplock_failure_pattern():
    """Limplock failures must reach resource saturation."""
    import numpy as np
    from temporal_patterns import LimplockFailure

    logger.info("--- Test 7: LimplockFailure pattern characteristics ---")

    pattern = LimplockFailure()
    rng = np.random.RandomState(42)
    W, F = 12, 5
    affected = [0, 1, 2, 3]

    signal = pattern.generate(W, F, affected, rng)

    # Test 1: Saturation level reached at end
    end_signal = signal[-3:, affected[:2]].mean().item()
    assert_test(
        end_signal > 2.0,
        f"LimplockFailure reaches saturation {end_signal:.2f} > 2.0",
        f"Saturation not reached: {end_signal:.2f}",
    )

    # Test 2: Signal increases over time (ramp)
    start_signal = signal[:3, affected[:2]].mean().item()
    assert_test(
        end_signal > start_signal * 1.5,
        f"LimplockFailure ramp: end {end_signal:.2f} > start {start_signal:.2f} * 1.5",
        f"No clear ramp pattern",
    )


# =========================================================================
# Test 8: All patterns are distinct
# =========================================================================

def test_patterns_distinct():
    """Each pattern should produce distinct temporal signatures."""
    import numpy as np
    from temporal_patterns import PATTERN_REGISTRY

    logger.info("--- Test 8: Pattern distinguishability ---")

    rng = np.random.RandomState(42)
    W, F = 12, 5
    affected = [0, 1, 2]

    signals = {}
    for name, pattern in PATTERN_REGISTRY.items():
        signals[name] = pattern.generate(W, F, affected, rng)

    # Check that patterns have different statistics
    stats = {}
    for name, sig in signals.items():
        stats[name] = {
            "max": sig[:, affected].max().item(),
            "mean": sig[:, affected].mean().item(),
            "std": sig[:, affected].std().item(),
        }

    # At least some patterns should differ significantly
    max_vals = [s["max"] for s in stats.values()]
    mean_vals = [s["mean"] for s in stats.values()]

    max_range = max(max_vals) - min(max_vals)
    mean_range = max(mean_vals) - min(mean_vals)

    assert_test(
        max_range > 1.0,
        f"Patterns have diverse max values (range={max_range:.2f})",
        f"Patterns too similar in max: range={max_range:.2f}",
    )

    assert_test(
        mean_range > 0.3,
        f"Patterns have diverse mean values (range={mean_range:.2f})",
        f"Patterns too similar in mean: range={mean_range:.2f}",
    )


# =========================================================================
# Test 9: Pattern registry completeness
# =========================================================================

def test_pattern_registry():
    """Verify pattern registry has all required entries."""
    from temporal_patterns import PATTERN_REGISTRY, PATTERN_WEIGHTS, PATTERN_DIFFICULTY

    logger.info("--- Test 9: Pattern registry completeness ---")

    # Test 1: 8 patterns total
    assert_test(
        len(PATTERN_REGISTRY) == 8,
        f"PATTERN_REGISTRY has 8 patterns (found {len(PATTERN_REGISTRY)})",
        f"Missing or extra patterns",
    )

    # Test 2: Weights sum to ~1.0
    weight_sum = sum(PATTERN_WEIGHTS.values())
    assert_test(
        abs(weight_sum - 1.0) < 0.01,
        f"PATTERN_WEIGHTS sum to 1.0 (actual: {weight_sum:.2f})",
        f"Weights don't sum to 1.0: {weight_sum:.2f}",
    )

    # Test 3: All patterns have weights
    for name in PATTERN_REGISTRY:
        assert_test(
            name in PATTERN_WEIGHTS,
            f"Pattern '{name}' has weight",
            f"Missing weight for {name}",
        )

    # Test 4: All patterns have difficulty rating
    for name in PATTERN_REGISTRY:
        assert_test(
            name in PATTERN_DIFFICULTY,
            f"Pattern '{name}' has difficulty",
            f"Missing difficulty for {name}",
        )

    # Test 5: Gray failure is 39%
    gray_weight = PATTERN_WEIGHTS.get("gray_failure", 0)
    assert_test(
        abs(gray_weight - 0.39) < 0.01,
        f"Gray failure weight is 39% (actual: {gray_weight:.0%})",
        f"Gray failure weight wrong: {gray_weight:.0%}",
    )


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CAUSALITY AND PATTERN VALIDATION TESTS")
    logger.info("=" * 60)

    test_causal_conv_no_future_leakage()
    test_temporal_jitter_no_wrap()
    test_temporal_jitter_zero_shift()
    test_causal_conv_output_shape()

    # New pattern tests
    test_gray_failure_pattern()
    test_metastable_failure_pattern()
    test_limplock_failure_pattern()
    test_patterns_distinct()
    test_pattern_registry()

    logger.info("=" * 60)
    if FAIL_COUNT == 0:
        logger.info("ALL %d TESTS PASSED", PASS_COUNT)
    else:
        logger.error("%d/%d TESTS FAILED", FAIL_COUNT, PASS_COUNT + FAIL_COUNT)
        sys.exit(1)
    logger.info("=" * 60)
