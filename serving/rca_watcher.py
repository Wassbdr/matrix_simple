"""
rca-watcher: polling loop that feeds live Prometheus metrics to the deployed
STA inference service and fires alerts when the model identifies a likely
root cause.

Architecture:
    every POLL_INTERVAL_S seconds:
        1. query Prometheus for the 6 sock-shop metrics per service over the
           last WINDOW_SECONDS seconds
        2. build an [N, F, W] window aligned with the sock_shop topology
        3. POST the window to <INFERENCE_URL>/predict
        4. read top-1 score; if it exceeds CONFIDENCE_THRESHOLD AND the
           system shows a degraded state (overall error rate above baseline),
           emit an alert:
             - log to stdout (always)
             - POST to <ALERTMANAGER_URL>/api/v2/alerts (if reachable)
             - bump prometheus counters exposed at /metrics

The watcher itself runs as a tiny FastAPI app so it also exposes /metrics
and /health.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven, sane defaults for in-cluster deploy)
# ---------------------------------------------------------------------------

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://prometheus-server.monitoring-metrics.svc.cluster.local",
)
INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://rca-sta-serve.rca-sandbox.svc.cluster.local:8080",
)
ALERTMANAGER_URL = os.environ.get(
    "ALERTMANAGER_URL",
    "http://prometheus-alertmanager.monitoring-metrics.svc.cluster.local:9093",
)
TARGET_NAMESPACE = os.environ.get("TARGET_NAMESPACE", "sock-shop")
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "60"))
WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "360"))  # 6 min
STEP_SECONDS = int(os.environ.get("STEP_SECONDS", "30"))       # 30s -> 12 samples
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "5.0"))
ERROR_RATE_TRIGGER = float(os.environ.get("ERROR_RATE_TRIGGER", "0.01"))  # 1% req errors
TOP_K = int(os.environ.get("TOP_K", "3"))

# Aligned with sock_shop topology in rca_pipeline/topologies.py
NODE_NAMES: List[str] = [
    "front-end", "catalogue", "catalogue-db", "carts", "carts-db",
    "orders", "orders-db", "user", "user-db", "session-db",
    "payment", "shipping", "rabbitmq", "queue-master",
]
N = len(NODE_NAMES)
F = 6   # cpu_usage, ram_usage, network_tx, network_rx, error_rate, latency_p99
W = WINDOW_SECONDS // STEP_SECONDS

# ---------------------------------------------------------------------------
# Prometheus client (in this watcher itself)
# ---------------------------------------------------------------------------

WATCHER_POLLS = Counter("rca_watcher_polls_total", "Polls executed", ["status"])
WATCHER_ALERTS = Counter("rca_watcher_alerts_total", "Alerts fired",
                          ["root_cause", "severity"])
WATCHER_LAST_TOP_SCORE = Gauge("rca_watcher_last_top_score",
                                "Score of the top-1 candidate at the last successful prediction")
WATCHER_LAST_ERROR_RATE = Gauge("rca_watcher_last_observed_error_rate",
                                 "Cluster-wide error rate observed at the last poll")

# ---------------------------------------------------------------------------
# Prometheus query helpers
# ---------------------------------------------------------------------------

def _promql_query_range(query: str, start: float, end: float, step: int) -> dict:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=8,
    )
    r.raise_for_status()
    return r.json()


def _promql_query_instant(query: str) -> dict:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


# Map a PromQL output (list of series keyed by pod) into an [N, W] matrix.
def _series_by_node(result: list, w: int) -> np.ndarray:
    arr = np.zeros((N, w), dtype=np.float32)
    for series in result:
        pod = series.get("metric", {}).get("pod", "")
        # Strip the replica-set + pod-template-hash suffix to get the service name
        # e.g. "carts-f4f6c98b9-zpdfn" -> "carts"
        parts = pod.split("-")
        # Greedy match against NODE_NAMES (longest prefix that is in our set)
        node = None
        for n_try in range(len(parts), 0, -1):
            cand = "-".join(parts[:n_try])
            if cand in NODE_NAMES:
                node = cand
                break
        if node is None:
            continue
        idx = NODE_NAMES.index(node)
        values = series.get("values", [])
        for j, (_ts, v) in enumerate(values[-w:]):
            try:
                arr[idx, j] = float(v)
            except (TypeError, ValueError):
                pass
    return arr


def collect_window() -> np.ndarray:
    """Build the [N, F, W] tensor expected by /predict."""
    now = time.time()
    start = now - WINDOW_SECONDS
    # Use sum-by-pod aggregations consistent with prometheus_connector.py
    ns = TARGET_NAMESPACE
    queries = {
        "cpu_usage": (
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}}[2m])) by (pod)'
        ),
        "ram_usage": (
            f'sum(container_memory_working_set_bytes{{namespace="{ns}",container!=""}}) by (pod)'
        ),
        "network_tx": (
            f'sum(rate(container_network_transmit_bytes_total{{namespace="{ns}"}}[2m])) by (pod)'
        ),
        "network_rx": (
            f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}"}}[2m])) by (pod)'
        ),
        # Application metrics may not exist for every service; default to 0.
        "error_rate": (
            f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}",code=~"5.."}}[2m])) by (pod)'
        ),
        "latency_p99": (
            f'histogram_quantile(0.99,'
            f' sum(rate(http_request_duration_seconds_bucket{{namespace="{ns}"}}[2m])) by (pod, le))'
        ),
    }
    feature_order = ["cpu_usage", "ram_usage", "network_tx", "network_rx",
                     "error_rate", "latency_p99"]
    tensor = np.zeros((N, F, W), dtype=np.float32)
    for f_idx, feat in enumerate(feature_order):
        try:
            resp = _promql_query_range(queries[feat], start, now, STEP_SECONDS)
        except requests.RequestException as exc:
            logger.warning("Prometheus query failed for %s: %s", feat, exc)
            continue
        result = resp.get("data", {}).get("result", [])
        m = _series_by_node(result, W)
        tensor[:, f_idx, :] = m
    return tensor


def observed_error_rate() -> float:
    """Cluster-wide 5xx error rate over the last 5 min, in [0, 1]."""
    ns = TARGET_NAMESPACE
    q = (
        f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}",code=~"5.."}}[5m]))'
        f' / clamp_min(sum(rate(http_request_duration_seconds_count{{namespace="{ns}"}}[5m])), 1e-9)'
    )
    try:
        resp = _promql_query_instant(q)
        result = resp.get("data", {}).get("result", [])
        if not result:
            return 0.0
        return float(result[0]["value"][1])
    except (requests.RequestException, ValueError, KeyError):
        return 0.0


# ---------------------------------------------------------------------------
# Inference call
# ---------------------------------------------------------------------------

def call_predict(window: np.ndarray) -> Optional[List[Tuple[str, float, int]]]:
    payload = {"window": window.tolist(), "top_k": TOP_K}
    try:
        r = requests.post(f"{INFERENCE_URL}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        logger.warning("predict request failed: %s", exc)
        return None
    if r.status_code != 200:
        logger.warning("predict returned %d: %s", r.status_code, r.text[:200])
        return None
    body = r.json()
    cands = body.get("candidates", [])
    return [(c["service"], float(c["score"]), int(c["rank"])) for c in cands]


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def post_alertmanager(root_cause: str, score: float, runners_up: List[Tuple[str, float]],
                       err_rate: float, severity: str = "warning") -> bool:
    """Send a single alert to Alertmanager. Return True on success."""
    payload = [
        {
            "labels": {
                "alertname": "RCAStaServeRootCauseDetected",
                "severity": severity,
                "component": "rca-watcher",
                "namespace": TARGET_NAMESPACE,
                "root_cause": root_cause,
                "model": "sta_sockshop_v1",
            },
            "annotations": {
                "summary": f"STA identifies root cause: {root_cause}",
                "description": (
                    f"Top-1 candidate = {root_cause} (score {score:.3f}). "
                    f"Cluster error rate = {err_rate:.1%}. "
                    f"Runners-up: {', '.join(f'{s}({sc:.2f})' for s, sc in runners_up)}."
                ),
            },
        }
    ]
    try:
        r = requests.post(f"{ALERTMANAGER_URL}/api/v2/alerts", json=payload, timeout=5)
        return r.status_code < 400
    except requests.RequestException as exc:
        logger.warning("Alertmanager POST failed: %s", exc)
        return False


def maybe_fire_alert(ranked: List[Tuple[str, float, int]], err_rate: float) -> None:
    if not ranked:
        return
    top_svc, top_score, _ = ranked[0]
    WATCHER_LAST_TOP_SCORE.set(top_score)
    if top_score < CONFIDENCE_THRESHOLD:
        logger.info("top score %.3f below threshold %.3f, no alert", top_score, CONFIDENCE_THRESHOLD)
        return
    if err_rate < ERROR_RATE_TRIGGER:
        logger.info("error rate %.4f below trigger %.4f, no alert (model spike alone)",
                    err_rate, ERROR_RATE_TRIGGER)
        return
    severity = "critical" if err_rate > 0.05 else "warning"
    runners = [(s, sc) for s, sc, _ in ranked[1:]]
    logger.warning("ALERT: root_cause=%s score=%.3f err_rate=%.1f%% severity=%s",
                   top_svc, top_score, err_rate * 100.0, severity)
    sent = post_alertmanager(top_svc, top_score, runners, err_rate, severity)
    WATCHER_ALERTS.labels(root_cause=top_svc, severity=severity).inc()
    logger.info("alertmanager delivery: %s", "ok" if sent else "failed")


# ---------------------------------------------------------------------------
# FastAPI surface (health + metrics)
# ---------------------------------------------------------------------------

app = FastAPI(title="RCA watcher", version="1.0.0")
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    return {
        "prometheus_url": PROMETHEUS_URL,
        "inference_url": INFERENCE_URL,
        "alertmanager_url": ALERTMANAGER_URL,
        "target_namespace": TARGET_NAMESPACE,
        "poll_interval_s": POLL_INTERVAL_S,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "n_samples_per_window": W,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "error_rate_trigger": ERROR_RATE_TRIGGER,
        "top_k": TOP_K,
        "nodes": NODE_NAMES,
    }


@app.on_event("startup")
def _start_loop():
    import threading
    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()


def watch_loop():
    """Background loop, started at FastAPI startup."""
    logger.info(
        "Watcher up: poll=%ds window=%ds step=%ds N=%d F=%d W=%d threshold=%.2f",
        POLL_INTERVAL_S, WINDOW_SECONDS, STEP_SECONDS, N, F, W, CONFIDENCE_THRESHOLD,
    )
    while True:
        try:
            window = collect_window()
            err = observed_error_rate()
            WATCHER_LAST_ERROR_RATE.set(err)
            ranked = call_predict(window)
            if ranked is None:
                WATCHER_POLLS.labels(status="predict_failed").inc()
            else:
                top = ranked[0]
                logger.info("poll ok: top=%s score=%.3f err_rate=%.1f%%",
                            top[0], top[1], err * 100.0)
                WATCHER_POLLS.labels(status="ok").inc()
                maybe_fire_alert(ranked, err)
        except Exception:
            logger.exception("poll failed")
            WATCHER_POLLS.labels(status="error").inc()
        time.sleep(POLL_INTERVAL_S)
