"""
prometheus_connector.py — Live Prometheus metric connector.

Queries Prometheus /api/v1/query_range for 6 standard Kubernetes pod metrics
and returns a numpy array compatible with graph_builder.py.

Output format: [window_size, n_services, n_metrics]
    - window_size: W time steps in the queried range
    - n_services: N pods/services (derived from pod discovery or explicit list)
    - n_metrics: 6 (cpu_usage, memory_bytes, network_rx, network_tx, latency_p99, error_rate)
      Note: observ-it CSV builder currently exports a 5-metric subset (without latency_p99).
      graph_builder.py now accepts both schemas and uses latency_p99 when present.

NaN handling:
    - Missing series (pod absent during window): filled with 0.0
    - Sparse series (shorter than W): forward-filled then zero-padded
    - Inf/NaN signal values: replaced with 0.0

Usage:
    from prometheus_connector import PrometheusConnector

    connector = PrometheusConnector(
        url="http://prometheus.monitoring.svc:9090",
        namespace="production",
        step="2s",
    )

    # Fetch 60s of data ending now
    import time
    end = time.time()
    start = end - 60
    X, pod_names = connector.fetch_window(start, end, window_size=30)
    # X shape: [30, n_services, 6]
"""

import time
import math
import logging
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard metric names (also the dimension order in output array)
METRIC_NAMES = ["cpu_usage", "memory_bytes", "network_rx", "network_tx", "latency_p99", "error_rate"]
N_METRICS = len(METRIC_NAMES)


# ---------------------------------------------------------------------------
# PromQL query templates
# ---------------------------------------------------------------------------

def _build_queries(namespace: str) -> Dict[str, str]:
    """Build the 6 PromQL queries for standard Kubernetes pod metrics."""
    ns = namespace
    return {
        "cpu_usage": (
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{ns}",container!="",container!="POD"}}[1m])) by (pod)'
        ),
        "memory_bytes": (
            f'sum(container_memory_working_set_bytes{{'
            f'namespace="{ns}",container!="",container!="POD"}}) by (pod)'
        ),
        "network_rx": (
            f'sum(rate(container_network_receive_bytes_total{{'
            f'namespace="{ns}"}}[1m])) by (pod)'
        ),
        "network_tx": (
            f'sum(rate(container_network_transmit_bytes_total{{'
            f'namespace="{ns}"}}[1m])) by (pod)'
        ),
        "latency_p99": (
            # Falls back to 0 if Istio/Linkerd not deployed
            f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{'
            f'namespace="{ns}",reporter="destination"}}[1m])) by (destination_workload, le)) or '
            f'vector(0)'
        ),
        "error_rate": (
            f'sum(rate(istio_requests_total{{'
            f'namespace="{ns}",reporter="destination",response_code!~"2.."}}[1m])) by (destination_workload)'
            f' / sum(rate(istio_requests_total{{namespace="{ns}",reporter="destination"}}[1m])) by (destination_workload)'
            f' or vector(0)'
        ),
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PrometheusConnector:
    """
    Fetches multi-metric time-series from Prometheus and returns a numpy
    array compatible with graph_builder.py / model_sta.py.

    Args:
        url:           Prometheus base URL (e.g., "http://localhost:9090")
        namespace:     Kubernetes namespace to query
        step:          Prometheus query step (e.g., "2s", "5s")
        timeout:       HTTP timeout in seconds
        pod_allowlist: Optional list of pod prefixes to include.
                       If None, all pods in the namespace are included.
    """

    def __init__(
        self,
        url: str = "http://localhost:9090",
        namespace: str = "default",
        step: str = "2s",
        timeout: int = 30,
        pod_allowlist: Optional[List[str]] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.namespace = namespace
        self.step = step
        self.timeout = timeout
        self.pod_allowlist = pod_allowlist
        self._queries = _build_queries(namespace)

        logger.info(
            "PrometheusConnector: url=%s, namespace=%s, step=%s",
            self.url, self.namespace, self.step,
        )

    def _query_range(
        self, query: str, start: float, end: float
    ) -> List[Dict[str, Any]]:
        """Execute a PromQL query_range request."""
        endpoint = f"{self.url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": self.step,
        }
        try:
            resp = requests.get(endpoint, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                logger.error("Prometheus query failed: %s", data.get("error", "unknown"))
                return []
            return data["data"]["result"]
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Prometheus at %s", self.url)
            return []
        except Exception as e:
            logger.error("Prometheus query error: %s", e)
            return []

    def _discover_pods(self, start: float, end: float) -> List[str]:
        """Discover all pods present in the namespace during the time range."""
        result = self._query_range(
            f'group(container_cpu_usage_seconds_total{{namespace="{self.namespace}",'
            f'container!="",container!="POD"}}) by (pod)',
            start, end,
        )
        pods = sorted({r["metric"].get("pod", "unknown") for r in result if "pod" in r["metric"]})
        if self.pod_allowlist:
            pods = [p for p in pods if any(p.startswith(prefix) for prefix in self.pod_allowlist)]
        return pods

    def _extract_label(self, result_item: Dict[str, Any]) -> str:
        """Extract a consistent pod label from a Prometheus result entry."""
        metric = result_item.get("metric", {})
        # Try pod first, then destination_workload (for Istio metrics)
        label = metric.get("pod") or metric.get("destination_workload") or "unknown"
        return label

    def _series_to_array(
        self, values: List, window_size: int, expected_timestamps: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Convert a Prometheus values list [(timestamp, value), ...] to a
        fixed-length numpy array of shape [window_size].

        Handles:
            - Series shorter than window_size (zero-padded at front)
            - NaN / Inf values (replaced with 0.0)
            - Series longer than window_size (truncated to last window_size)
        """
        arr = np.array([
            float(v) if not (math.isnan(float(v)) or math.isinf(float(v))) else 0.0
            for _, v in values
        ], dtype=np.float32)

        if len(arr) >= window_size:
            return arr[-window_size:]
        else:
            # Zero-pad at the front
            padded = np.zeros(window_size, dtype=np.float32)
            padded[-len(arr):] = arr
            return padded

    def fetch_window(
        self,
        start: float,
        end: float,
        window_size: int = 30,
        pod_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fetch metrics for all pods in the time range and return a numpy array.

        Args:
            start:        UNIX timestamp (seconds) of window start
            end:          UNIX timestamp (seconds) of window end
            window_size:  W — number of time steps expected
            pod_names:    Optional fixed pod list (skip discovery).
                          If None, pods are auto-discovered.
        Returns:
            X:          numpy array [window_size, n_services, N_METRICS] float32
            pod_names:  List[str] of pod names (length = n_services)
        """
        # 1. Discover pods if not provided
        if pod_names is None:
            pod_names = self._discover_pods(start, end)
            if not pod_names:
                logger.warning("No pods discovered in namespace '%s'.", self.namespace)
                return np.zeros((window_size, 0, N_METRICS), dtype=np.float32), []

        n_services = len(pod_names)
        pod_index = {p: i for i, p in enumerate(pod_names)}

        # 2. Initialize output array
        X = np.zeros((window_size, n_services, N_METRICS), dtype=np.float32)

        # 3. Query each metric
        for feat_idx, (metric_name, query) in enumerate(self._queries.items()):
            results = self._query_range(query, start, end)
            if not results:
                logger.warning("Empty result for metric '%s'.", metric_name)
                continue

            # Map each result series to the pod index
            for item in results:
                label = self._extract_label(item)
                values = item.get("values", [])

                # Try exact match first
                pod_idx = pod_index.get(label)
                if pod_idx is None:
                    # Fallback: longest-prefix match (same logic as graph_builder.py)
                    best_len = 0
                    best_idx = None
                    for pod_name, idx in pod_index.items():
                        if label.startswith(pod_name) or pod_name.startswith(label):
                            match_len = min(len(label), len(pod_name))
                            if match_len > best_len:
                                best_len = match_len
                                best_idx = idx
                    pod_idx = best_idx

                if pod_idx is None:
                    logger.debug("Pod label '%s' not in pod_names, skipping.", label)
                    continue

                if values:
                    series = self._series_to_array(values, window_size)
                    X[:, pod_idx, feat_idx] = series

        logger.info(
            "Fetched window: shape=%s, pods=%d, range=[%.0f, %.0f]",
            X.shape, n_services, start, end,
        )
        return X, pod_names

    def fetch_latest_window(
        self,
        window_seconds: int = 60,
        window_size: int = 30,
        pod_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convenience method: fetch the most recent window_seconds of data ending now.

        Returns:
            X:          [window_size, n_services, N_METRICS]
            pod_names:  List[str]
        """
        end = time.time()
        start = end - window_seconds
        return self.fetch_window(start, end, window_size=window_size, pod_names=pod_names)

    def to_graph_builder_format(
        self, X: np.ndarray
    ) -> "torch.Tensor":
        """
        Convert the numpy array to the PyTorch format expected by SelectThenAnalyze:
        [N, F, W] (nodes × features × time).

        Requires torch to be installed.

        Args:
            X: [W, N, F] numpy array from fetch_window()
        Returns:
            torch.Tensor of shape [N, F, W]
        """
        import torch
        # X is [W, N, F] -> permute to [N, F, W]
        return torch.from_numpy(X).permute(1, 2, 0)  # [N, F, W]

    # ------------------------------------------------------------------
    # Multimodal: Loki (logs) and Jaeger (traces)
    # ------------------------------------------------------------------

    def fetch_logs(
        self,
        start: float,
        end: float,
        pod_names: Optional[List[str]] = None,
        loki_url: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Fetch log lines per pod via Loki HTTP API.

        Args:
            start, end:  UNIX timestamps.
            pod_names:   pod list (uses all discovered if None).
            loki_url:    Loki base URL (default: same host as Prometheus, port 3100).
        Returns:
            Dict mapping pod_name → list of log line strings.
        """
        if loki_url is None:
            loki_url = self.url.rsplit(":", 1)[0] + ":3100"
        loki_url = loki_url.rstrip("/")

        if pod_names is None:
            pod_names = self._discover_pods(start, end)

        logs: Dict[str, List[str]] = {p: [] for p in pod_names}

        for pod in pod_names:
            query = f'{{namespace="{self.namespace}", pod="{pod}"}}'
            endpoint = f"{loki_url}/loki/api/v1/query_range"
            params = {
                "query": query,
                "start": int(start * 1e9),
                "end": int(end * 1e9),
                "limit": 5000,
            }
            try:
                resp = requests.get(endpoint, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                for stream in data.get("data", {}).get("result", []):
                    for ts, line in stream.get("values", []):
                        logs[pod].append(line)
            except requests.exceptions.ConnectionError:
                logger.debug("Cannot connect to Loki at %s for pod %s", loki_url, pod)
            except Exception as e:
                logger.debug("Loki query error for pod %s: %s", pod, e)

        return logs

    def fetch_traces(
        self,
        start: float,
        end: float,
        pod_names: Optional[List[str]] = None,
        jaeger_url: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, List[Dict]]:
        """
        Fetch trace spans per service via Jaeger API.

        Args:
            start, end:   UNIX timestamps.
            pod_names:    service list.
            jaeger_url:   Jaeger base URL (default: same host, port 16686).
            limit:        max traces per service.
        Returns:
            Dict mapping service_name → list of span dicts.
        """
        if jaeger_url is None:
            jaeger_url = self.url.rsplit(":", 1)[0] + ":16686"
        jaeger_url = jaeger_url.rstrip("/")

        if pod_names is None:
            pod_names = self._discover_pods(start, end)

        traces: Dict[str, List[Dict]] = {p: [] for p in pod_names}

        for svc in pod_names:
            endpoint = f"{jaeger_url}/api/traces"
            params = {
                "service": svc,
                "start": int(start * 1e6),
                "end": int(end * 1e6),
                "limit": limit,
            }
            try:
                resp = requests.get(endpoint, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                for trace in data.get("data", []):
                    for span in trace.get("spans", []):
                        traces[svc].append(span)
            except requests.exceptions.ConnectionError:
                logger.debug("Cannot connect to Jaeger at %s for %s", jaeger_url, svc)
            except Exception as e:
                logger.debug("Jaeger query error for %s: %s", svc, e)

        return traces

    def fetch_multimodal(
        self,
        start: float,
        end: float,
        window_size: int = 30,
        pod_names: Optional[List[str]] = None,
        loki_url: Optional[str] = None,
        jaeger_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch all three data sources in one call.

        Returns:
            {
                "metrics": (X [W,N,F], pod_names),
                "logs": {pod: [lines]},
                "traces": {pod: [spans]},
            }
        """
        X, pods = self.fetch_window(start, end, window_size, pod_names)
        logs = self.fetch_logs(start, end, pods, loki_url)
        traces = self.fetch_traces(start, end, pods, jaeger_url)

        return {
            "metrics": (X, pods),
            "logs": logs,
            "traces": traces,
        }


# ---------------------------------------------------------------------------
# Offline test / dry-run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus Connector — dry run")
    parser.add_argument("--url", type=str, default="http://localhost:9090")
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument("--step", type=str, default="2s")
    parser.add_argument("--window", type=int, default=60, help="Window duration in seconds")
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    connector = PrometheusConnector(
        url=args.url,
        namespace=args.namespace,
        step=args.step,
    )

    print(f"Fetching last {args.window}s from {args.url} (namespace={args.namespace})...")
    X, pods = connector.fetch_latest_window(
        window_seconds=args.window,
        window_size=args.window_size,
    )
    print(f"Output shape: {X.shape}  ({len(pods)} pods, {N_METRICS} metrics)")
    print(f"Pods: {pods}")
    if X.size > 0:
        print(f"Value range: [{X.min():.4f}, {X.max():.4f}]")
        nan_count = np.isnan(X).sum()
        print(f"NaN count: {nan_count}")
    else:
        print("No data returned (check connectivity and namespace).")
