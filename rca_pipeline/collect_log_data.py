#!/usr/bin/env python3
"""
Collect log features from Loki/Elasticsearch for RCA training.

This module extracts numerical features from logs that can be combined
with metrics and traces for multi-modal root cause analysis.

Features extracted (4 per service):
- log_volume: Total log lines per time bucket
- error_log_ratio: Ratio of ERROR/WARN logs to total
- unique_error_count: Number of unique error signatures
- log_entropy: Shannon entropy of log templates (abnormal patterns)

Usage:
    from collect_log_data import collect_log_features

    X_logs, timestamps = collect_log_features(
        loki_url="http://loki:3100",
        services=["frontend", "backend", "database"],
        start=datetime.utcnow() - timedelta(hours=24),
        end=datetime.utcnow(),
    )
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Common error patterns for detection
ERROR_PATTERNS = [
    r"error",
    r"exception",
    r"failed",
    r"timeout",
    r"connection refused",
    r"out of memory|OOM",
    r"stack ?trace",
    r"panic",
    r"fatal",
    r"crash",
    r"killed",
    r"segfault|segmentation fault",
]
ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# Log level patterns
LOG_LEVEL_PATTERNS = {
    "error": re.compile(r"\b(ERROR|FATAL|CRITICAL|PANIC)\b", re.IGNORECASE),
    "warn": re.compile(r"\b(WARN|WARNING)\b", re.IGNORECASE),
    "info": re.compile(r"\b(INFO)\b", re.IGNORECASE),
    "debug": re.compile(r"\b(DEBUG|TRACE)\b", re.IGNORECASE),
}


def extract_log_template(log_line: str) -> str:
    """
    Extract a template from a log line by replacing variable parts.

    This is a simplified version of Drain3 log parsing.
    Variables replaced:
    - Numbers -> <NUM>
    - UUIDs -> <UUID>
    - IPs -> <IP>
    - Hex strings -> <HEX>
    - Timestamps -> <TS>
    """
    # Remove timestamp at start (common log formats)
    line = re.sub(r"^\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?\s*", "", log_line)

    # Replace UUIDs
    line = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", line, flags=re.I)

    # Replace IP addresses
    line = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", line)

    # Replace hex strings (>4 chars)
    line = re.sub(r"\b0x[0-9a-f]{4,}\b", "<HEX>", line, flags=re.I)
    line = re.sub(r"\b[0-9a-f]{8,}\b", "<HEX>", line, flags=re.I)

    # Replace numbers
    line = re.sub(r"\b\d+\.?\d*\b", "<NUM>", line)

    # Replace paths
    line = re.sub(r"(/[a-zA-Z0-9_.-]+)+", "<PATH>", line)

    return line.strip()


def compute_entropy(counts: Counter) -> float:
    """Compute Shannon entropy from a Counter object."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def get_error_signature(log_line: str) -> Optional[str]:
    """
    Extract a unique error signature from a log line.

    Returns None if no error detected.
    """
    if not ERROR_REGEX.search(log_line):
        return None

    # Extract template and hash it
    template = extract_log_template(log_line)
    # Take first 100 chars to create signature
    signature = hashlib.md5(template[:100].encode()).hexdigest()[:8]
    return signature


def query_loki(
    loki_url: str,
    query: str,
    start: datetime,
    end: datetime,
    limit: int = 5000,
) -> List[Dict]:
    """
    Query Loki for logs.

    Args:
        loki_url: Loki URL (e.g., http://loki:3100)
        query: LogQL query
        start: Start datetime
        end: End datetime
        limit: Maximum number of log entries

    Returns:
        List of log entries with timestamps and lines
    """
    import requests

    try:
        response = requests.get(
            f"{loki_url}/loki/api/v1/query_range",
            params={
                "query": query,
                "start": int(start.timestamp() * 1e9),  # Loki uses nanoseconds
                "end": int(end.timestamp() * 1e9),
                "limit": limit,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            logger.warning("Loki query failed: %s", query)
            return []

        logs = []
        for stream in data.get("data", {}).get("result", []):
            labels = stream.get("stream", {})
            for value in stream.get("values", []):
                timestamp_ns, line = value
                logs.append({
                    "timestamp": int(timestamp_ns) / 1e9,  # Convert to seconds
                    "line": line,
                    "labels": labels,
                })

        return logs

    except Exception as e:
        logger.warning("Loki query error: %s", e)
        return []


def query_elasticsearch(
    es_url: str,
    index: str,
    service: str,
    start: datetime,
    end: datetime,
    limit: int = 5000,
) -> List[Dict]:
    """
    Query Elasticsearch for logs (alternative to Loki).

    Args:
        es_url: Elasticsearch URL
        index: Index pattern (e.g., "logs-*")
        service: Service name to filter
        start: Start datetime
        end: End datetime
        limit: Maximum number of log entries

    Returns:
        List of log entries with timestamps and lines
    """
    import requests

    try:
        query = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"service": service}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start.isoformat() + "Z",
                                    "lte": end.isoformat() + "Z",
                                }
                            }
                        },
                    ]
                }
            },
            "sort": [{"@timestamp": "asc"}],
        }

        response = requests.post(
            f"{es_url}/{index}/_search",
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        logs = []
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            timestamp_str = source.get("@timestamp", "")
            message = source.get("message", source.get("log", ""))

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "")).timestamp()
            except Exception:
                continue

            logs.append({
                "timestamp": timestamp,
                "line": message,
                "labels": source,
            })

        return logs

    except Exception as e:
        logger.warning("Elasticsearch query error: %s", e)
        return []


def collect_log_features(
    loki_url: Optional[str],
    services: List[str],
    start: datetime,
    end: datetime,
    step: str = "1m",
    elasticsearch_url: Optional[str] = None,
    elasticsearch_index: str = "logs-*",
) -> Tuple[torch.Tensor, List[float]]:
    """
    Extract log-based features for each service.

    Features extracted (4 per service):
    - log_volume: Number of log lines per time bucket
    - error_log_ratio: Ratio of ERROR/WARN logs to total
    - unique_error_count: Number of unique error signatures
    - log_entropy: Shannon entropy of log templates

    Args:
        loki_url: Loki URL (preferred, use None if using Elasticsearch)
        services: List of service names
        start: Start datetime
        end: End datetime
        step: Time bucket size (e.g., "1m", "5m")
        elasticsearch_url: Elasticsearch URL (alternative to Loki)
        elasticsearch_index: Elasticsearch index pattern

    Returns:
        X_logs: Tensor [T, N, 4] of log features
        timestamps: List of bucket timestamps
    """
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
    num_features = 4  # log_volume, error_ratio, unique_errors, entropy

    # Initialize output tensor
    X_logs = torch.zeros(num_buckets, num_services, num_features)

    for svc_idx, service in enumerate(services):
        try:
            # Query logs
            if loki_url:
                # Loki query - filter by service label
                query = f'{{service="{service}"}} |= ""'
                logs = query_loki(loki_url, query, start, end)
            elif elasticsearch_url:
                logs = query_elasticsearch(
                    elasticsearch_url, elasticsearch_index, service, start, end
                )
            else:
                logger.warning("No log source configured")
                continue

            if not logs:
                logger.debug("No logs found for service %s", service)
                continue

            # Aggregate by time bucket
            bucket_logs: Dict[int, List[str]] = defaultdict(list)

            for log in logs:
                ts = log["timestamp"]
                bucket_idx = int((ts - start.timestamp()) / step_seconds)
                if 0 <= bucket_idx < num_buckets:
                    bucket_logs[bucket_idx].append(log["line"])

            # Compute features per bucket
            for bucket_idx, lines in bucket_logs.items():
                if not lines:
                    continue

                log_volume = len(lines)
                error_count = 0
                unique_errors: Set[str] = set()
                template_counts: Counter = Counter()

                for line in lines:
                    # Check for error level
                    if LOG_LEVEL_PATTERNS["error"].search(line) or LOG_LEVEL_PATTERNS["warn"].search(line):
                        error_count += 1

                    # Extract error signature
                    sig = get_error_signature(line)
                    if sig:
                        unique_errors.add(sig)

                    # Extract template for entropy
                    template = extract_log_template(line)
                    template_counts[template] += 1

                # Compute entropy
                entropy = compute_entropy(template_counts)

                # Store features
                X_logs[bucket_idx, svc_idx, 0] = log_volume
                X_logs[bucket_idx, svc_idx, 1] = error_count / log_volume if log_volume > 0 else 0
                X_logs[bucket_idx, svc_idx, 2] = len(unique_errors)
                X_logs[bucket_idx, svc_idx, 3] = entropy

        except Exception as e:
            logger.warning("Failed to collect log features for %s: %s", service, e)
            # Leave zeros for this service

    # Forward fill zeros (similar to metrics)
    for n in range(num_services):
        for f in range(num_features):
            for t in range(1, num_buckets):
                if X_logs[t, n, f] == 0 and X_logs[t-1, n, f] != 0:
                    X_logs[t, n, f] = X_logs[t-1, n, f]

    logger.info("Collected log features: shape=%s", X_logs.shape)
    return X_logs, timestamps


def align_timestamps(
    timestamps_list: List[List[float]],
    tensors: List[torch.Tensor],
    target_timestamps: List[float],
) -> List[torch.Tensor]:
    """
    Align multiple tensors to a common timestamp grid using linear interpolation.

    Args:
        timestamps_list: List of timestamp lists (one per tensor)
        tensors: List of tensors [T_i, N, F_i]
        target_timestamps: Target timestamp grid

    Returns:
        List of aligned tensors [T_target, N, F_i]
    """
    aligned = []

    for ts, tensor in zip(timestamps_list, tensors):
        T_src, N, F = tensor.shape
        T_tgt = len(target_timestamps)

        if T_src == T_tgt and ts == target_timestamps:
            aligned.append(tensor)
            continue

        # Create aligned tensor
        aligned_tensor = torch.zeros(T_tgt, N, F)

        # Convert to numpy for interpolation
        ts_arr = np.array(ts)
        target_arr = np.array(target_timestamps)

        for n in range(N):
            for f in range(F):
                values = tensor[:, n, f].numpy()
                # Linear interpolation
                aligned_values = np.interp(target_arr, ts_arr, values)
                aligned_tensor[:, n, f] = torch.tensor(aligned_values)

        aligned.append(aligned_tensor)

    return aligned


if __name__ == "__main__":
    # Test with mock data
    import argparse

    parser = argparse.ArgumentParser(description="Collect log features")
    parser.add_argument("--loki-url", default="http://localhost:3100", help="Loki URL")
    parser.add_argument("--services", default="frontend,backend", help="Comma-separated services")
    parser.add_argument("--duration", default="1h", help="Duration to collect")
    args = parser.parse_args()

    from collect_cluster_data import parse_duration

    end = datetime.utcnow()
    duration = parse_duration(args.duration)
    start = end - duration

    services = args.services.split(",")

    X_logs, timestamps = collect_log_features(
        loki_url=args.loki_url,
        services=services,
        start=start,
        end=end,
    )

    print(f"Log features shape: {X_logs.shape}")
    print(f"Timestamps: {len(timestamps)}")
    print(f"Non-zero entries: {(X_logs != 0).sum().item()}")
