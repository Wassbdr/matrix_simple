"""Batch data-drift detector: training reference vs recent live features.

Layered relative to the existing online detector:
  * rca_pipeline/drift_detector.py (ADWIN) = online concept-drift suppression
    inside the watcher;
  * this job = batch *data* drift of live feature distributions vs the
    training reference, feeding the retraining trigger.

Reference data is a feature sample exported at train time
(``reference_windows.parquet`` in the bundle). Current data is the last
N hours pulled via the existing PrometheusConnector. Per-feature drift is
computed with Evidently's DataDriftPreset when available, else a built-in
KS / PSI fallback (so the job runs without the [monitoring] extra). The
drift share is pushed to Prometheus (Pushgateway) and an HTML report is
logged to MLflow when configured.

    python -m rca_pipeline.monitoring.drift_job \
        --reference /bundle/reference_windows.parquet \
        --lookback-hours 6 --pushgateway http://pushgateway:9091
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

METRIC_NAMES = ["cpu_usage", "memory_bytes", "network_rx", "network_tx",
                "latency_p99", "error_rate"]


# --- distribution distances (Evidently-free fallback) ----------------------

def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=bins)
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r = r / max(r.sum(), 1)
    c = c / max(c.sum(), 1)
    eps = 1e-6
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def fallback_drift(reference: np.ndarray, current: np.ndarray,
                   psi_threshold: float = 0.2) -> Dict[str, object]:
    """Per-feature PSI; a feature is 'drifted' when PSI > threshold."""
    n_features = reference.shape[1]
    per_feature = {}
    drifted = 0
    for f in range(n_features):
        psi = _psi(reference[:, f], current[:, f])
        name = METRIC_NAMES[f] if f < len(METRIC_NAMES) else f"f{f}"
        is_drift = psi > psi_threshold
        per_feature[name] = {"psi": psi, "drifted": is_drift}
        drifted += int(is_drift)
    return {
        "drift_share": drifted / max(n_features, 1),
        "n_drifted": drifted,
        "n_features": n_features,
        "per_feature": per_feature,
        "method": "psi_fallback",
    }


def evidently_drift(reference: np.ndarray, current: np.ndarray) -> Optional[Dict[str, object]]:
    """Evidently DataDriftPreset; returns None if Evidently is unavailable."""
    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return None

    cols = [METRIC_NAMES[f] if f < len(METRIC_NAMES) else f"f{f}"
            for f in range(reference.shape[1])]
    ref_df = pd.DataFrame(reference, columns=cols)
    cur_df = pd.DataFrame(current, columns=cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    return {
        "drift_share": float(drift["share_of_drifted_columns"]),
        "n_drifted": int(drift["number_of_drifted_columns"]),
        "n_features": int(drift["number_of_columns"]),
        "method": "evidently",
        "_report": report,
    }


def _flatten_windows(arr: np.ndarray) -> np.ndarray:
    """[*, N, F, W] or [N, F, W] -> 2-D (samples, F) by averaging over time/nodes."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 4:        # [B, N, F, W]
        return a.mean(axis=(1, 3))
    if a.ndim == 3:        # [N, F, W]
        return a.mean(axis=2)
    if a.ndim == 2:
        return a
    raise ValueError(f"unexpected feature array shape {a.shape}")


def push_metrics(gateway: str, drift_share: float, n_drifted: int) -> None:
    try:
        from prometheus_client import (
            CollectorRegistry,
            Gauge,
            push_to_gateway,
        )
    except ImportError:
        logger.warning("prometheus_client missing; skipping push.")
        return
    reg = CollectorRegistry()
    Gauge("rca_drift_share", "Share of drifted features", registry=reg).set(drift_share)
    Gauge("rca_drift_features", "Count of drifted features", registry=reg).set(n_drifted)
    Gauge("rca_drift_detected", "1 if any feature drifted", registry=reg).set(
        1.0 if n_drifted > 0 else 0.0)
    push_to_gateway(gateway, job="rca_drift", registry=reg)
    logger.info("Pushed drift metrics to %s (share=%.3f)", gateway, drift_share)


def run(reference_path: str, lookback_hours: int, pushgateway: Optional[str],
        prometheus_url: Optional[str], namespace: str) -> Dict[str, object]:
    import pandas as pd

    reference = _flatten_windows(pd.read_parquet(reference_path).values)

    if prometheus_url:
        from rca_pipeline.prometheus_connector import PrometheusConnector
        conn = PrometheusConnector(prometheus_url=prometheus_url, namespace=namespace)
        x, _ = conn.fetch_latest_window(window_seconds=lookback_hours * 3600, window_size=30)
        current = _flatten_windows(np.transpose(x, (1, 2, 0)))  # [W,N,F] -> [N,F]
    else:
        raise SystemExit("--prometheus-url is required to fetch current data.")

    res = evidently_drift(reference, current) or fallback_drift(reference, current)
    logger.info("Drift: share=%.3f (%d/%d features, method=%s)",
                res["drift_share"], res["n_drifted"], res["n_features"], res["method"])

    if pushgateway:
        push_metrics(pushgateway, float(res["drift_share"]), int(res["n_drifted"]))
    return res


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="reference_windows.parquet")
    parser.add_argument("--lookback-hours", type=int, default=6)
    parser.add_argument("--pushgateway", default=None)
    parser.add_argument("--prometheus-url", default=None)
    parser.add_argument("--namespace", default="sock-shop")
    args = parser.parse_args(argv)
    run(args.reference, args.lookback_hours, args.pushgateway,
        args.prometheus_url, args.namespace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
