#!/usr/bin/env python3
"""
Collect real metrics from a Kubernetes cluster for RCA training.

This script:
1. Connects to Prometheus/VictoriaMetrics
2. Collects metrics for specified services
3. Detects anomalies (optional: uses your trained model)
4. Generates PyG Data format for training

Usage:
    # Collect last 24h of data
    python collect_cluster_data.py --prometheus-url http://prometheus:9090 --duration 24h

    # Collect with anomaly labels from alerts
    python collect_cluster_data.py --prometheus-url http://prometheus:9090 \
        --alertmanager-url http://alertmanager:9093 --duration 7d

    # Collect specific services
    python collect_cluster_data.py --services frontend,backend,database --duration 1d
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import argparse
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Default metrics to collect (maps to 5 features)
DEFAULT_METRICS = {
    "cpu": 'rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[1m])',
    "memory": 'container_memory_usage_bytes{{pod=~"{service}.*"}} / container_spec_memory_limit_bytes{{pod=~"{service}.*"}}',
    "latency_p99": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{service="{service}"}}[1m]))',
    "error_rate": 'rate(http_requests_total{{service="{service}", status=~"5.."}}[1m]) / rate(http_requests_total{{service="{service}"}}[1m])',
    "request_rate": 'rate(http_requests_total{{service="{service}"}}[1m])',
}


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '1h', '24h', '7d'."""
    unit = duration_str[-1]
    value = int(duration_str[:-1])
    if unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit}")


def query_prometheus(
    url: str,
    query: str,
    start: datetime,
    end: datetime,
    step: str = "1m",
) -> List[Tuple[float, float]]:
    """Query Prometheus range API."""
    import requests

    response = requests.get(
        f"{url}/api/v1/query_range",
        params={
            "query": query,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "step": step,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if data["status"] != "success":
        logger.warning("Query failed: %s", query)
        return []

    if not data["data"]["result"]:
        return []

    # Return timestamp, value pairs
    values = data["data"]["result"][0]["values"]
    return [(float(v[0]), float(v[1])) for v in values]


def get_alerts_in_range(
    alertmanager_url: str,
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    """Get fired alerts from Alertmanager for labeling."""
    import requests

    try:
        response = requests.get(
            f"{alertmanager_url}/api/v2/alerts",
            params={
                "active": "true",
                "silenced": "false",
                "inhibited": "false",
            },
            timeout=30,
        )
        response.raise_for_status()
        alerts = response.json()

        # Filter by time range
        relevant = []
        for alert in alerts:
            starts_at = datetime.fromisoformat(alert["startsAt"].replace("Z", ""))
            if start <= starts_at <= end:
                relevant.append({
                    "service": alert["labels"].get("service", alert["labels"].get("pod", "unknown")),
                    "alertname": alert["labels"].get("alertname", "unknown"),
                    "severity": alert["labels"].get("severity", "warning"),
                    "starts_at": starts_at,
                    "ends_at": datetime.fromisoformat(alert["endsAt"].replace("Z", "")) if alert.get("endsAt") else None,
                })
        return relevant
    except Exception as e:
        logger.warning("Failed to get alerts: %s", e)
        return []


def discover_services(prometheus_url: str) -> List[str]:
    """Auto-discover services from Prometheus labels."""
    import requests

    try:
        # Query for unique service labels
        response = requests.get(
            f"{prometheus_url}/api/v1/label/service/values",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        services = data["data"]
        logger.info("Discovered %d services: %s", len(services), services[:10])
        return services
    except Exception as e:
        logger.warning("Service discovery failed: %s", e)
        return []


def build_topology_from_traces(
    jaeger_url: str,
    services: List[str],
    duration: timedelta,
) -> Dict[str, List[str]]:
    """Build service topology from Jaeger traces."""
    import requests

    topology = defaultdict(set)
    end = datetime.utcnow()
    start = end - duration

    for service in services:
        try:
            response = requests.get(
                f"{jaeger_url}/api/traces",
                params={
                    "service": service,
                    "start": int(start.timestamp() * 1e6),
                    "end": int(end.timestamp() * 1e6),
                    "limit": 100,
                },
                timeout=30,
            )
            response.raise_for_status()
            traces = response.json()["data"]

            for trace in traces:
                processes = {p: info["serviceName"] for p, info in trace["processes"].items()}
                for span in trace["spans"]:
                    parent_refs = span.get("references", [])
                    for ref in parent_refs:
                        if ref["refType"] == "CHILD_OF":
                            parent_span_id = ref["spanID"]
                            # Find parent service
                            for other_span in trace["spans"]:
                                if other_span["spanID"] == parent_span_id:
                                    parent_svc = processes.get(other_span["processID"])
                                    child_svc = processes.get(span["processID"])
                                    if parent_svc and child_svc and parent_svc != child_svc:
                                        topology[parent_svc].add(child_svc)

        except Exception as e:
            logger.warning("Failed to get traces for %s: %s", service, e)

    return {k: list(v) for k, v in topology.items()}


def collect_trace_features(
    jaeger_url: str,
    services: List[str],
    start: datetime,
    end: datetime,
    step: str = "1m",
) -> Tuple[torch.Tensor, List[float]]:
    """
    Extract trace-based features from Jaeger for each service.

    Features extracted (4 per service):
    - span_count: Number of spans per time bucket (throughput indicator)
    - avg_span_duration_ms: Average span duration in milliseconds
    - error_span_ratio: Ratio of error spans to total spans
    - critical_path_ratio: Ratio of time spent on critical path

    Args:
        jaeger_url: Jaeger query API URL (e.g., http://jaeger:16686)
        services: List of service names
        start: Start datetime
        end: End datetime
        step: Time bucket size (e.g., "1m", "5m")

    Returns:
        X_trace: Tensor [T, N, 4] of trace features
        timestamps: List of bucket timestamps
    """
    import requests
    from math import ceil

    # Parse step to seconds
    step_seconds = 60  # default 1m
    if step.endswith('m'):
        step_seconds = int(step[:-1]) * 60
    elif step.endswith('s'):
        step_seconds = int(step[:-1])
    elif step.endswith('h'):
        step_seconds = int(step[:-1]) * 3600

    # Create time buckets
    total_seconds = (end - start).total_seconds()
    num_buckets = ceil(total_seconds / step_seconds)
    timestamps = [start.timestamp() + i * step_seconds for i in range(num_buckets)]

    num_services = len(services)
    num_features = 4  # span_count, avg_duration, error_ratio, critical_path_ratio

    # Initialize output tensor
    X_trace = torch.zeros(num_buckets, num_services, num_features)

    # Service name to index mapping
    svc_to_idx = {s: i for i, s in enumerate(services)}

    for svc_idx, service in enumerate(services):
        try:
            # Query Jaeger for traces
            response = requests.get(
                f"{jaeger_url}/api/traces",
                params={
                    "service": service,
                    "start": int(start.timestamp() * 1e6),
                    "end": int(end.timestamp() * 1e6),
                    "limit": 5000,  # Get more traces for better stats
                },
                timeout=60,
            )
            response.raise_for_status()
            traces = response.json().get("data", [])

            # Aggregate spans by time bucket
            bucket_spans: Dict[int, List[Dict]] = defaultdict(list)

            for trace in traces:
                processes = {p: info["serviceName"] for p, info in trace.get("processes", {}).items()}
                trace_duration = 0
                critical_path_time = 0

                for span in trace.get("spans", []):
                    span_svc = processes.get(span.get("processID", ""))
                    if span_svc != service:
                        continue

                    span_start_us = span.get("startTime", 0)
                    span_duration_us = span.get("duration", 0)
                    span_start_sec = span_start_us / 1e6

                    # Determine time bucket
                    bucket_idx = int((span_start_sec - start.timestamp()) / step_seconds)
                    if 0 <= bucket_idx < num_buckets:
                        has_error = any(
                            tag.get("key") == "error" and tag.get("value") == True
                            for tag in span.get("tags", [])
                        )
                        # Check for HTTP status errors
                        for tag in span.get("tags", []):
                            if tag.get("key") == "http.status_code":
                                status = int(tag.get("value", 200))
                                if status >= 400:
                                    has_error = True

                        bucket_spans[bucket_idx].append({
                            "duration_us": span_duration_us,
                            "has_error": has_error,
                            "is_root": len(span.get("references", [])) == 0,
                        })

                        trace_duration = max(trace_duration, span_duration_us)
                        # Simple critical path heuristic: root spans contribute to critical path
                        if len(span.get("references", [])) == 0:
                            critical_path_time += span_duration_us

            # Compute features per bucket
            for bucket_idx, spans in bucket_spans.items():
                if not spans:
                    continue

                span_count = len(spans)
                total_duration = sum(s["duration_us"] for s in spans)
                error_count = sum(1 for s in spans if s["has_error"])
                root_duration = sum(s["duration_us"] for s in spans if s["is_root"])

                X_trace[bucket_idx, svc_idx, 0] = span_count  # span_count
                X_trace[bucket_idx, svc_idx, 1] = total_duration / span_count / 1000  # avg_duration_ms
                X_trace[bucket_idx, svc_idx, 2] = error_count / span_count if span_count > 0 else 0  # error_ratio
                X_trace[bucket_idx, svc_idx, 3] = root_duration / total_duration if total_duration > 0 else 0  # critical_path

        except Exception as e:
            logger.warning("Failed to collect trace features for %s: %s", service, e)
            # Leave zeros for this service

    # Forward fill zeros (similar to metrics)
    for n in range(num_services):
        for f in range(num_features):
            for t in range(1, num_buckets):
                if X_trace[t, n, f] == 0 and X_trace[t-1, n, f] != 0:
                    X_trace[t, n, f] = X_trace[t-1, n, f]

    logger.info("Collected trace features: shape=%s", X_trace.shape)
    return X_trace, timestamps


def collect_service_metrics(
    prometheus_url: str,
    services: List[str],
    start: datetime,
    end: datetime,
    metrics: Dict[str, str] = None,
    step: str = "1m",
) -> Tuple[torch.Tensor, List[str], List[float]]:
    """
    Collect metrics for services from Prometheus.

    Returns:
        X: Tensor [T, N, F]
        timestamps: List of timestamps
        service_names: List of service names
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    num_services = len(services)
    num_features = len(metrics)

    all_timestamps = set()
    data_by_service: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(dict)

    for service in services:
        for feat_name, query_template in metrics.items():
            query = query_template.format(service=service)
            values = query_prometheus(prometheus_url, query, start, end, step)

            if values:
                data_by_service[service][feat_name] = values
                all_timestamps.update(v[0] for v in values)

    if not all_timestamps:
        raise ValueError("No data collected from Prometheus")

    timestamps = sorted(all_timestamps)
    T = len(timestamps)
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    logger.info("Collected %d timestamps for %d services", T, num_services)

    # Build tensor
    X = torch.zeros(T, num_services, num_features)
    feat_names = list(metrics.keys())

    for svc_idx, service in enumerate(services):
        for feat_idx, feat_name in enumerate(feat_names):
            if feat_name in data_by_service[service]:
                for ts, val in data_by_service[service][feat_name]:
                    if ts in ts_to_idx:
                        t_idx = ts_to_idx[ts]
                        X[t_idx, svc_idx, feat_idx] = val if not np.isnan(val) else 0.0

    # Fill NaN with forward fill then zero
    for n in range(num_services):
        for f in range(num_features):
            for t in range(1, T):
                if X[t, n, f] == 0 and X[t-1, n, f] != 0:
                    X[t, n, f] = X[t-1, n, f]

    return X, timestamps, services


def create_windows(
    X: torch.Tensor,
    timestamps: List[float],
    window_size: int = 12,
    stride: int = 6,
    alerts: Optional[List[Dict]] = None,
    services: Optional[List[str]] = None,
) -> List[Data]:
    """
    Create sliding windows from collected metrics.

    Args:
        X: Tensor [T, N, F]
        timestamps: List of timestamps
        window_size: Window size in timesteps
        stride: Stride between windows
        alerts: Optional list of alerts for labeling
        services: Service names for topology

    Returns:
        List of PyG Data objects
    """
    T, N, F = X.shape
    dataset = []

    # Build edge index (fully connected for now)
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Service name to index
    name_to_idx = {s: i for i, s in enumerate(services)} if services else {}

    for t_start in range(0, T - window_size + 1, stride):
        t_end = t_start + window_size
        window_ts = timestamps[t_start:t_end]

        # Extract window [W, N, F] -> [N, F, W]
        x = X[t_start:t_end].permute(1, 2, 0)

        data = Data(x=x, edge_index=edge_index.clone())

        # Determine if this window contains an anomaly
        is_anomaly = False
        root_cause = None

        if alerts:
            window_start = datetime.fromtimestamp(window_ts[0])
            window_end = datetime.fromtimestamp(window_ts[-1])

            for alert in alerts:
                alert_start = alert["starts_at"]
                alert_end = alert.get("ends_at") or datetime.utcnow()

                # Check overlap
                if alert_start <= window_end and alert_end >= window_start:
                    is_anomaly = True
                    svc = alert["service"]
                    if svc in name_to_idx:
                        root_cause = svc
                        break

        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [root_cause] if root_cause else ["unknown"]
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]

        # Metadata
        data.timestamp_start = window_ts[0]
        data.timestamp_end = window_ts[-1]
        data.is_recurrent = False  # Can be labeled later
        data.pattern_type = "real"

        dataset.append(data)

    logger.info("Created %d windows (%d anomalous)",
                len(dataset), sum(1 for d in dataset if d.y.item() == 1))

    return dataset


def normalize_dataset(dataset: List[Data]) -> List[Data]:
    """Z-normalize the dataset."""
    # Compute global stats
    all_x = torch.cat([d.x for d in dataset], dim=-1)  # [N, F, T_total]
    mean = all_x.mean(dim=-1, keepdim=True)
    std = all_x.std(dim=-1, keepdim=True) + 1e-8

    for d in dataset:
        d.x = (d.x - mean) / std

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Collect cluster data for RCA")
    parser.add_argument("--prometheus-url", required=True, help="Prometheus/VM URL")
    parser.add_argument("--alertmanager-url", help="Alertmanager URL for labels")
    parser.add_argument("--jaeger-url", help="Jaeger URL for topology")
    parser.add_argument("--services", help="Comma-separated service names (or 'auto')")
    parser.add_argument("--duration", default="24h", help="Duration to collect (e.g., 1h, 24h, 7d)")
    parser.add_argument("--window-size", type=int, default=12, help="Window size")
    parser.add_argument("--stride", type=int, default=6, help="Window stride")
    parser.add_argument("--step", default="1m", help="Prometheus step")
    parser.add_argument("--output", default="cluster_dataset.pt", help="Output path")
    parser.add_argument("--normalize", action="store_true", help="Z-normalize features")
    args = parser.parse_args()

    # Time range
    end = datetime.utcnow()
    duration = parse_duration(args.duration)
    start = end - duration

    logger.info("Collecting data from %s to %s", start, end)

    # Discover services
    if args.services == "auto" or args.services is None:
        services = discover_services(args.prometheus_url)
        if not services:
            logger.error("No services discovered. Specify --services manually.")
            return
        services = services[:50]  # Limit for first run
    else:
        services = args.services.split(",")

    logger.info("Services: %s", services)

    # Collect metrics
    X, timestamps, services = collect_service_metrics(
        args.prometheus_url,
        services,
        start,
        end,
        step=args.step,
    )

    # Get alerts for labeling
    alerts = []
    if args.alertmanager_url:
        alerts = get_alerts_in_range(args.alertmanager_url, start, end)
        logger.info("Found %d alerts in range", len(alerts))

    # Build topology from traces
    topology = {}
    if args.jaeger_url:
        topology = build_topology_from_traces(args.jaeger_url, services, duration)
        logger.info("Built topology: %d edges", sum(len(v) for v in topology.values()))

    # Create windows
    dataset = create_windows(
        X=X,
        timestamps=timestamps,
        window_size=args.window_size,
        stride=args.stride,
        alerts=alerts,
        services=services,
    )

    if args.normalize:
        dataset = normalize_dataset(dataset)

    # Build edge index from topology
    if topology:
        name_to_idx = {s: i for i, s in enumerate(services)}
        edges = []
        for src, dsts in topology.items():
            if src in name_to_idx:
                for dst in dsts:
                    if dst in name_to_idx:
                        edges.append([name_to_idx[src], name_to_idx[dst]])
                        edges.append([name_to_idx[dst], name_to_idx[src]])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            for d in dataset:
                d.edge_index = edge_index.clone()

    # Save
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    torch.save({
        "train": train_data,
        "test": test_data,
        "node_names": services,
        "topology": topology,
        "metadata": {
            "duration": args.duration,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "window_size": args.window_size,
            "num_alerts": len(alerts),
        },
        "version": "cluster_v1",
    }, args.output)

    logger.info("Saved to %s", args.output)
    logger.info("Train: %d, Test: %d", len(train_data), len(test_data))

    anomalies = [d for d in dataset if d.y.item() == 1]
    logger.info("Anomaly windows: %d (%.1f%%)", len(anomalies), 100 * len(anomalies) / len(dataset))


if __name__ == "__main__":
    main()
