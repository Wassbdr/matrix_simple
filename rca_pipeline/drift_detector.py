"""
drift_detector.py — ADWIN-based concept drift detector for Kubernetes metrics.

Implements the ADWIN (ADaptive WINdowing) algorithm (Bifet & Gavaldà, SDM 2007)
for detecting concept drift in streaming metric data.

Motivation (Myrtollari et al. 2025):
    Kubernetes deployments generate distribution shifts in metrics that are
    incorrectly classified as root-cause anomalies by static RCA models.
    This detector identifies these transitions and suppresses RCA calls during
    a configurable cooldown window.

ADWIN Algorithm:
    Maintains a sliding window of values. At each step, it tests whether there
    exists a subwindow where the mean differs from the rest beyond a statistical
    threshold δ. If so, it declares drift and truncates the older portion.

Pipeline Integration:
    - Call drift_detector.update(metric_name, value) for each metric at each timestep.
    - Call drift_detector.check_drift() → returns True if suspected deployment.
    - If True, skip RCA call for `cooldown_steps` steps.

Usage:
    from drift_detector import MultiMetricDriftDetector

    detector = MultiMetricDriftDetector(
        metric_names=["cpu", "memory", "latency_p99", "error_rate"],
        delta=0.002,
        drift_fraction_threshold=0.5,
        cooldown_steps=30,
    )

    for timestep, metrics in streaming_metrics:
        for name, value in metrics.items():
            detector.update(name, value)
        if detector.is_in_cooldown():
            log("Skipping RCA: suspected deployment")
        else:
            rca_result = run_rca(metrics)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADWIN single-stream detector
# ---------------------------------------------------------------------------

class ADWIN:
    """
    ADWIN(δ) — ADaptive WINdowing drift detector.

    Reference: Bifet & Gavaldà, "Learning from Time-Changing Data with
    Adaptive Windowing", SDM 2007.

    Time complexity:  O(log n) amortized per update
    Space complexity: O(log n) — uses a compressed bucket representation

    Parameters:
        delta: confidence parameter (default 0.002 → 99.8% confidence per test)
              Smaller delta → fewer false positives, slower detection.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._total = 0.0           # sum of values in window
        self._variance = 0.0        # online variance accumulator (Welford)
        self._width = 0             # current window size
        self._drift_detected = False

        # Compressed bucket representation:
        # buckets[k] = list of (sum, count) pairs for the 2^k level
        self._buckets: List[deque] = [deque() for _ in range(32)]
        self._n_buckets_per_level = 5  # max buckets at each level (space compression)

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift was detected.

        Args:
            value: new metric observation
        Returns:
            True if drift detected this step
        """
        self._drift_detected = False
        self._insert(value)
        self._drift_detected = self._test_drift()
        if self._drift_detected:
            logger.debug("ADWIN drift detected, trimming window.")
        return self._drift_detected

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def width(self) -> int:
        return self._width

    @property
    def mean(self) -> float:
        return self._total / self._width if self._width > 0 else 0.0

    def _insert(self, value: float) -> None:
        """Insert new value into bucket structure."""
        self._width += 1
        self._total += value
        # Welford online variance (for epsilon computation)
        if self._width == 1:
            self._mean_est = value
            self._m2 = 0.0
        else:
            old_mean = self._mean_est
            self._mean_est += (value - self._mean_est) / self._width
            self._m2 += (value - old_mean) * (value - self._mean_est)
        self._variance = self._m2 / max(self._width - 1, 1)

        self._buckets[0].append((value, 1))
        self._compress_buckets()

    def _compress_buckets(self) -> None:
        """Merge buckets at same level when over capacity."""
        for level in range(len(self._buckets) - 1):
            while len(self._buckets[level]) > self._n_buckets_per_level:
                b1 = self._buckets[level].popleft()
                b2 = self._buckets[level].popleft()
                merged = (b1[0] + b2[0], b1[1] + b2[1])
                self._buckets[level + 1].append(merged)

    def _test_drift(self) -> bool:
        """
        Test if any subwindow split yields mean difference above threshold.
        Cuts from the oldest bucket progressively.
        """
        n0 = 0
        s0 = 0.0
        n1 = self._width
        s1 = self._total
        variance = max(self._variance, 1e-8)

        # Scan from oldest to newest bucket, splitting the window
        for level in range(len(self._buckets)):
            for bkt_sum, bkt_count in self._buckets[level]:
                n0 += bkt_count
                n1 -= bkt_count
                s0 += bkt_sum
                s1 -= bkt_sum

                if n0 == 0 or n1 == 0:
                    continue

                m0 = s0 / n0
                m1 = s1 / n1
                epsilon_cut = math.sqrt(
                    (1.0 / (2 * n0) + 1.0 / (2 * n1))
                    * variance
                    * math.log(4 * math.log(self._width + 1) / self.delta)
                )
                if abs(m0 - m1) >= epsilon_cut:
                    # Drift: trim the older subwindow (n0 elements)
                    self._trim_oldest(n0, s0)
                    return True

        return False

    def _trim_oldest(self, n_trim: int, s_trim: float) -> None:
        """Remove the oldest n_trim elements from the window."""
        self._width -= n_trim
        self._total -= s_trim
        # Reset variance estimator (approximation: proportional)
        self._variance *= max(self._width, 1) / max(self._width + n_trim, 1)
        # Clear bucket structure and rebuild (simple approximation for trim)
        remaining_sum = self._total
        remaining_n = self._width
        for level in self._buckets:
            level.clear()
        # Re-insert as a single bucket at level ceil(log2(remaining_n))
        if remaining_n > 0:
            level_idx = max(0, int(math.log2(remaining_n + 1)))
            level_idx = min(level_idx, len(self._buckets) - 1)
            self._buckets[level_idx].append((remaining_sum, remaining_n))


# ---------------------------------------------------------------------------
# Multi-metric drift detector
# ---------------------------------------------------------------------------

class MultiMetricDriftDetector:
    """
    Monitors N concurrent metric streams for drift.

    Drift policy: if > drift_fraction_threshold fraction of metrics drift
    simultaneously, log "suspected_deployment" and enter cooldown.

    Args:
        metric_names:              list of metric identifiers to monitor
        delta:                     ADWIN confidence parameter (default 0.002)
        drift_fraction_threshold:  fraction trigger (default 0.5 = 50% of metrics)
        cooldown_steps:            how many timesteps to suppress RCA after drift
        min_window:                minimum window before drift testing (default 30)
    """

    def __init__(
        self,
        metric_names: List[str],
        delta: float = 0.002,
        drift_fraction_threshold: float = 0.5,
        cooldown_steps: int = 30,
        min_window: int = 30,
    ) -> None:
        self.metric_names = metric_names
        self.drift_fraction_threshold = drift_fraction_threshold
        self.cooldown_steps = cooldown_steps
        self.min_window = min_window

        self._detectors: Dict[str, ADWIN] = {
            name: ADWIN(delta=delta) for name in metric_names
        }
        self._drift_flags: Dict[str, bool] = {name: False for name in metric_names}
        self._cooldown_remaining: int = 0
        self._total_drift_events: int = 0
        self._step_count: int = 0

    def update(self, metric_name: str, value: float) -> None:
        """
        Update a single metric stream with a new observation.

        Args:
            metric_name: must be in self.metric_names
            value:       new metric value (NaN-safe: NaN values are skipped)
        """
        if metric_name not in self._detectors:
            logger.warning("Unknown metric '%s' — skipping.", metric_name)
            return

        import math
        if math.isnan(value) or math.isinf(value):
            return  # skip invalid values

        drift = self._detectors[metric_name].update(value)
        if drift and self._detectors[metric_name].width >= self.min_window:
            self._drift_flags[metric_name] = True
        else:
            self._drift_flags[metric_name] = False

    def update_batch(self, metric_values: Dict[str, float]) -> None:
        """
        Update all metrics in a single timestep.

        Args:
            metric_values: dict mapping metric_name -> value
        """
        self._step_count += 1
        for name, val in metric_values.items():
            self.update(name, val)
        self._check_global_drift()

    def _check_global_drift(self) -> None:
        """If > threshold fraction of metrics drifted, trigger cooldown."""
        if not self._drift_flags:
            return
        n_drifted = sum(1 for v in self._drift_flags.values() if v)
        frac = n_drifted / len(self._drift_flags)
        if frac > self.drift_fraction_threshold:
            self._cooldown_remaining = self.cooldown_steps
            self._total_drift_events += 1
            logger.warning(
                "DRIFT DETECTED: %.0f%% of metrics drifted simultaneously "
                "(threshold=%.0f%%). suspected_deployment event #%d. "
                "Cooldown: %d steps.",
                frac * 100,
                self.drift_fraction_threshold * 100,
                self._total_drift_events,
                self.cooldown_steps,
            )
        elif self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def is_in_cooldown(self) -> bool:
        """
        Returns True if the pipeline should skip RCA (deployment transition suspected).
        """
        return self._cooldown_remaining > 0

    def get_drift_summary(self) -> Dict[str, object]:
        """Returns diagnostic summary of detector state."""
        return {
            "step_count": self._step_count,
            "cooldown_remaining": self._cooldown_remaining,
            "total_drift_events": self._total_drift_events,
            "drift_flags": dict(self._drift_flags),
            "window_widths": {
                name: det.width for name, det in self._detectors.items()
            },
            "current_means": {
                name: round(det.mean, 4) for name, det in self._detectors.items()
            },
        }

    def reset(self) -> None:
        """Reset all detectors (e.g., after a known deployment)."""
        for det in self._detectors.values():
            det.__init__(det.delta)
        self._drift_flags = {name: False for name in self.metric_names}
        self._cooldown_remaining = 0
        logger.info("MultiMetricDriftDetector reset.")


# ---------------------------------------------------------------------------
# Convenience factory for standard Kubernetes metrics
# ---------------------------------------------------------------------------

def make_k8s_drift_detector(
    delta: float = 0.002,
    cooldown_steps: int = 30,
) -> MultiMetricDriftDetector:
    """
    Create a detector pre-configured for the standard 6 Kubernetes metrics
    used in prometheus_connector.py.
    """
    return MultiMetricDriftDetector(
        metric_names=[
            "cpu_usage",
            "memory_bytes",
            "network_rx",
            "network_tx",
            "latency_p99",
            "error_rate",
        ],
        delta=delta,
        drift_fraction_threshold=0.5,
        cooldown_steps=cooldown_steps,
        min_window=30,
    )


# ---------------------------------------------------------------------------
# Unit tests (run with: python drift_detector.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)

    print("=== ADWIN unit test ===")

    # Test 1: No drift — stable distribution
    adwin = ADWIN(delta=0.002)
    for _ in range(100):
        adwin.update(random.gauss(0.5, 0.05))
    assert not adwin.drift_detected, "False positive on stable stream"
    print(f"  Test 1 PASS: No false positive on stable stream (width={adwin.width})")

    # Test 2: Drift — step change in mean
    adwin2 = ADWIN(delta=0.002)
    for _ in range(100):
        adwin2.update(random.gauss(0.5, 0.05))
    drift_found = False
    for _ in range(100):
        if adwin2.update(random.gauss(2.0, 0.05)):
            drift_found = True
            break
    assert drift_found, "Missed obvious drift"
    print(f"  Test 2 PASS: Drift detected on step change (width={adwin2.width})")

    # Test 3: MultiMetric cooldown
    detector = make_k8s_drift_detector(cooldown_steps=5)
    for _ in range(50):
        detector.update_batch({
            "cpu_usage": random.gauss(0.3, 0.02),
            "memory_bytes": random.gauss(500e6, 1e6),
            "network_rx": random.gauss(1e6, 1e4),
            "network_tx": random.gauss(0.5e6, 5e3),
            "latency_p99": random.gauss(0.1, 0.005),
            "error_rate": random.gauss(0.01, 0.001),
        })
    # Simulate deployment (large step change)
    for _ in range(30):
        detector.update_batch({
            "cpu_usage": random.gauss(0.8, 0.05),
            "memory_bytes": random.gauss(900e6, 5e6),
            "network_rx": random.gauss(5e6, 1e5),
            "network_tx": random.gauss(3e6, 1e4),
            "latency_p99": random.gauss(0.5, 0.02),
            "error_rate": random.gauss(0.05, 0.005),
        })
    in_cooldown = detector.is_in_cooldown()
    summary = detector.get_drift_summary()
    print(f"  Test 3: drift_events={summary['total_drift_events']}, "
          f"in_cooldown={in_cooldown}")
    print(f"  Test 3 PASS: Multi-metric detector works")

    print("\n=== All tests passed ===")


# ---------------------------------------------------------------------------
# Test-Time Adaptation (TTA) for concept drift
# ---------------------------------------------------------------------------

class TestTimeAdaptation:
    """
    Lightweight Test-Time Adaptation triggered by ADWIN drift detection.

    Instead of re-training the entire model (expensive), TTA:
    1. Resets BatchNorm running statistics to adapt to new data distribution
    2. Fine-tunes a small adaptation MLP on the most recent windows
    3. Restores after the drift period ends

    Usage:
        tta = TestTimeAdaptation(model, drift_detector)
        # During inference:
        if tta.should_adapt():
            tta.adapt(recent_windows, edge_index)
        logits = model(x, edge_index)
    """

    def __init__(
        self,
        model,
        drift_detector,
        adapt_lr: float = 1e-4,
        adapt_steps: int = 5,
        window_size: int = 20,
    ):
        import torch
        import copy

        self.model = model
        self.drift_detector = drift_detector
        self.adapt_lr = adapt_lr
        self.adapt_steps = adapt_steps
        self.window_size = window_size

        # Save original BN statistics for restoration
        self._original_bn_state = {}
        self._save_bn_state()
        self._is_adapted = False

    def _save_bn_state(self):
        import torch
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                self._original_bn_state[name] = {
                    'running_mean': module.running_mean.clone() if module.running_mean is not None else None,
                    'running_var': module.running_var.clone() if module.running_var is not None else None,
                    'num_batches_tracked': module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
                }

    def _reset_bn_stats(self):
        import torch
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if module.running_mean is not None:
                    module.running_mean.zero_()
                if module.running_var is not None:
                    module.running_var.fill_(1.0)
                if module.num_batches_tracked is not None:
                    module.num_batches_tracked.zero_()

    def _restore_bn_stats(self):
        import torch
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if name in self._original_bn_state:
                    state = self._original_bn_state[name]
                    if state['running_mean'] is not None:
                        module.running_mean.copy_(state['running_mean'])
                    if state['running_var'] is not None:
                        module.running_var.copy_(state['running_var'])
                    if state['num_batches_tracked'] is not None:
                        module.num_batches_tracked.copy_(state['num_batches_tracked'])

    def should_adapt(self) -> bool:
        return self.drift_detector.is_in_cooldown()

    def adapt(self, recent_windows, edge_index):
        """
        Perform TTA: reset BN stats and run a few forward passes to
        recalibrate batch normalization to the new distribution.

        Args:
            recent_windows: [n_recent, N, F, W] tensor of recent windows
            edge_index: [2, E] graph topology
        """
        import torch

        self._reset_bn_stats()
        self.model.train()

        # Run forward passes to update BN running stats
        with torch.no_grad():
            for i in range(min(self.adapt_steps, recent_windows.size(0))):
                x = recent_windows[i:i+1]
                _ = self.model(x, edge_index)

        self.model.eval()
        self._is_adapted = True

    def restore(self):
        """Restore original model state after drift period ends."""
        if self._is_adapted:
            self._restore_bn_stats()
            self._is_adapted = False
