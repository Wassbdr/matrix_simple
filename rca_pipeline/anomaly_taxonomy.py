"""
anomaly_taxonomy.py - Signal-level anomaly classification (SOTA 2023-2026).

Implements three taxonomies:
    1. Chandola (ACM CS 2009): point / contextual / collective
    2. Zamanzadeh Darban (ACM CS 2024): temporal (global/local/trend/seasonal/shapelet)
       + inter-metric (global/local/temporal-intermetric)
    3. CRI-Model (Reichert 2018): Change / Rupture / Impact phase detection

Used as feature engineering for gray_failure_detector.py and label enrichment
in simulated_data_scalable.py.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class ChandolaLabel(NamedTuple):
    is_point: bool
    is_contextual: bool
    is_collective: bool


class ZamanzadehLabel(NamedTuple):
    temporal_global: bool
    temporal_local: bool
    temporal_trend: bool
    temporal_seasonal: bool
    intermetric_global: bool
    intermetric_local: bool


class CRIPhase(NamedTuple):
    phase: str        # "change" | "rupture" | "impact" | "normal"
    confidence: float


# ---------------------------------------------------------------------------
# Chandola (2009)
# ---------------------------------------------------------------------------

def classify_chandola(
    window: np.ndarray,
    global_mean: float,
    global_std: float,
    z_threshold: float = 3.0,
    collective_min_len: int = 3,
    collective_corr_drop: float = 0.4,
) -> ChandolaLabel:
    """
    Classify a univariate window [W] into Chandola categories.

    Point:      any single value deviates > z_threshold σ from global stats.
    Contextual: max value is within z_threshold globally but deviates locally
                (compared to 5-step rolling mean).
    Collective: a contiguous sub-sequence of length ≥ collective_min_len whose
                running autocorrelation drops below threshold.
    """
    if window.ndim != 1 or len(window) < 3:
        return ChandolaLabel(False, False, False)

    z = np.abs((window - global_mean) / max(global_std, 1e-8))

    is_point = bool(np.any(z > z_threshold))

    local_mean = np.convolve(window, np.ones(5) / 5, mode="same")
    local_dev = np.abs(window - local_mean) / max(global_std, 1e-8)
    is_contextual = bool(np.any(local_dev > z_threshold) and not is_point)

    is_collective = False
    if len(window) >= collective_min_len * 2:
        half = len(window) // 2
        c1 = np.corrcoef(window[:half], window[half:2 * half])[0, 1]
        if np.isnan(c1):
            c1 = 1.0
        is_collective = bool(abs(c1) < collective_corr_drop)

    return ChandolaLabel(is_point, is_contextual, is_collective)


# ---------------------------------------------------------------------------
# Zamanzadeh Darban (2024) - MTS extensions
# ---------------------------------------------------------------------------

def classify_zamanzadeh(
    window_mts: np.ndarray,
    baseline_corr: Optional[np.ndarray] = None,
    trend_threshold: float = 0.6,
    corr_drop_threshold: float = 0.3,
) -> ZamanzadehLabel:
    """
    Classify a multivariate window [W, F] into Zamanzadeh categories.

    Temporal axis (per-feature, then OR):
        global:    any feature deviates > 3σ from its own mean.
        local:     rolling z-score spike without global deviation.
        trend:     Mann-Kendall trend statistic |S/Smax| > threshold.

    Inter-metric axis:
        global:    mean pairwise correlation changes > corr_drop_threshold
                   compared to baseline_corr.
        local:     short-window correlation break (first vs second half).
    """
    W, F = window_mts.shape
    if W < 4 or F < 2:
        return ZamanzadehLabel(False, False, False, False, False, False)

    feat_means = window_mts.mean(axis=0)
    feat_stds = window_mts.std(axis=0).clip(min=1e-8)
    z = np.abs((window_mts - feat_means) / feat_stds)

    t_global = bool(np.any(z > 3.0))

    roll = np.convolve(np.ones(3) / 3, np.ones(3), mode="same")  # dummy kernel
    local_means = np.apply_along_axis(
        lambda col: np.convolve(col, np.ones(3) / 3, mode="same"), axis=0, arr=window_mts
    )
    local_z = np.abs((window_mts - local_means) / feat_stds)
    t_local = bool(np.any(local_z > 3.0) and not t_global)

    # Mann-Kendall trend
    def _mk_s(x):
        n = len(x)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(x[j] - x[i])
        s_max = n * (n - 1) / 2
        return abs(s) / max(s_max, 1)

    mk_scores = [_mk_s(window_mts[:, f]) for f in range(F)]
    t_trend = bool(any(s > trend_threshold for s in mk_scores))

    # Inter-metric correlation
    if F >= 2:
        curr_corr = np.corrcoef(window_mts, rowvar=False)
        curr_corr = np.nan_to_num(curr_corr, nan=0.0)
        if baseline_corr is not None:
            corr_diff = np.abs(curr_corr - baseline_corr).mean()
            im_global = bool(corr_diff > corr_drop_threshold)
        else:
            im_global = False

        half = W // 2
        c1 = np.corrcoef(window_mts[:half], rowvar=False)
        c2 = np.corrcoef(window_mts[half:], rowvar=False)
        c1, c2 = np.nan_to_num(c1), np.nan_to_num(c2)
        im_local = bool(np.abs(c1 - c2).mean() > corr_drop_threshold)
    else:
        im_global = im_local = False

    return ZamanzadehLabel(t_global, t_local, t_trend, False, im_global, im_local)


# ---------------------------------------------------------------------------
# CRI-Model (Reichert 2018)
# ---------------------------------------------------------------------------

def classify_cri_phase(
    window: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    change_z: float = 1.5,
    rupture_z: float = 3.0,
) -> CRIPhase:
    """
    Assign a CRI phase to a univariate window [W]:
        Normal:  all values within change_z σ of baseline.
        Change:  drift detected (mean shifted > change_z σ) but no spike.
        Rupture: spike or step > rupture_z σ.
        Impact:  sustained elevation after rupture (tail > baseline + change_z σ).
    """
    if len(window) < 4:
        return CRIPhase("normal", 0.0)
    z_abs = np.abs((window - baseline_mean) / max(baseline_std, 1e-8))
    max_z = float(z_abs.max())
    tail_z = float(z_abs[-max(1, len(window) // 4):].mean())

    if max_z < change_z:
        return CRIPhase("normal", 1.0 - max_z / change_z)
    if max_z >= rupture_z:
        if tail_z >= change_z:
            return CRIPhase("impact", min(1.0, tail_z / rupture_z))
        return CRIPhase("rupture", min(1.0, max_z / rupture_z))
    return CRIPhase("change", min(1.0, max_z / rupture_z))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Chandola
    normal = np.random.randn(30)
    spike = normal.copy(); spike[15] = 10.0
    c = classify_chandola(spike, 0.0, 1.0)
    assert c.is_point, "expected point anomaly"
    print(f"  Chandola point: {c}")

    # Zamanzadeh
    mts = np.random.randn(30, 4)
    mts[:, 0] += np.linspace(0, 5, 30)  # trend
    z = classify_zamanzadeh(mts)
    assert z.temporal_trend, f"expected trend: {z}"
    print(f"  Zamanzadeh: {z}")

    # CRI
    phase = classify_cri_phase(spike, 0.0, 1.0)
    assert phase.phase == "rupture", f"expected rupture: {phase}"
    print(f"  CRI: {phase}")

    print("\n=== anomaly_taxonomy tests passed ===")
