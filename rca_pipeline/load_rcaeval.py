"""
RCAEval Dataset Loader for benchmarking against SOTA.

RCAEval is a standard benchmark for Root Cause Analysis containing
real-world incidents from multiple sources (Alibaba, Azure, etc.).

Installation:
    pip install rcaeval

Usage:
    from load_rcaeval import load_rcaeval_as_pyg
    train_data, test_data, node_names = load_rcaeval_as_pyg(dataset_name="alibaba")

References:
    - RCAEval: https://github.com/NetManAIOps/RCAEval
    - Paper: "RCAEval: A Benchmark for Root Cause Analysis" (ICSE 2024)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_rcaeval_as_pyg(
    dataset_name: str = "alibaba",
    window_size: int = 12,
    num_features: int = 5,
    train_ratio: float = 0.8,
    normalize: bool = True,
) -> Tuple[List[Data], List[Data], List[str]]:
    """
    Load RCAEval dataset and convert to PyG Data format.

    Args:
        dataset_name: One of "alibaba", "azure", "gaia", "aiops2020"
        window_size: Temporal window size (will truncate/pad)
        num_features: Number of features to use (will select top-k)
        train_ratio: Train/test split ratio
        normalize: Whether to z-normalize features

    Returns:
        train_data: List[Data] for training
        test_data: List[Data] for testing
        node_names: List[str] of node/service names
    """
    try:
        from rcaeval.data import load_dataset
        from rcaeval.utils import get_node_names
    except ImportError:
        logger.error("RCAEval not installed. Run: pip install rcaeval")
        raise ImportError("Please install rcaeval: pip install rcaeval")

    logger.info("Loading RCAEval dataset: %s", dataset_name)

    # Load raw data from RCAEval
    raw_data = load_dataset(dataset_name)
    node_names = get_node_names(raw_data)
    num_nodes = len(node_names)

    logger.info("Dataset: %s, Nodes: %d, Incidents: %d",
                dataset_name, num_nodes, len(raw_data['incidents']))

    # Build adjacency from RCAEval topology
    if 'topology' in raw_data:
        edge_index = _build_edge_index(raw_data['topology'], node_names)
    else:
        # Fallback: fully connected
        edge_index = _build_fully_connected(num_nodes)

    # Convert incidents to PyG Data
    dataset = []
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    for incident in raw_data['incidents']:
        # Extract time series: [T, N, F]
        ts = incident['time_series']
        root_cause = incident['root_cause']

        # Skip if root cause not in topology
        if root_cause not in name_to_idx:
            continue

        # Truncate/pad to window_size
        T = ts.shape[0]
        if T > window_size:
            ts = ts[-window_size:]  # Take last window_size
        elif T < window_size:
            pad = np.zeros((window_size - T, ts.shape[1], ts.shape[2]))
            ts = np.concatenate([pad, ts], axis=0)

        # Select features (top-k by variance)
        if ts.shape[2] > num_features:
            feat_var = ts.var(axis=(0, 1))
            top_feats = np.argsort(feat_var)[-num_features:]
            ts = ts[:, :, top_feats]
        elif ts.shape[2] < num_features:
            pad = np.zeros((ts.shape[0], ts.shape[1], num_features - ts.shape[2]))
            ts = np.concatenate([ts, pad], axis=2)

        # Normalize
        if normalize:
            mean = ts.mean(axis=0, keepdims=True)
            std = ts.std(axis=0, keepdims=True) + 1e-8
            ts = (ts - mean) / std

        # Convert to tensor [N, F, W]
        x = torch.tensor(ts, dtype=torch.float32).permute(1, 2, 0)

        # Create Data object
        data = Data(x=x, edge_index=edge_index.clone())
        data.y = torch.tensor([1], dtype=torch.long)  # All incidents are anomalies
        data.root_cause = [root_cause]
        data.is_recurrent = incident.get('is_recurrent', False)
        data.pattern_type = incident.get('pattern_type', 'unknown')
        data.severity = incident.get('severity', 1.0)

        # Store original metadata
        data.incident_id = incident.get('id', 'unknown')
        data.timestamp = incident.get('timestamp', None)

        dataset.append(data)

    # Train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split_idx = int(len(dataset) * train_ratio)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    logger.info("Loaded %d incidents: %d train, %d test",
                len(dataset), len(train_data), len(test_data))

    return train_data, test_data, node_names


def _build_edge_index(topology: Dict, node_names: List[str]) -> torch.Tensor:
    """Build edge_index from topology dict."""
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    edges = []

    for src, dsts in topology.items():
        if src not in name_to_idx:
            continue
        src_idx = name_to_idx[src]
        for dst in dsts:
            if dst in name_to_idx:
                dst_idx = name_to_idx[dst]
                edges.append([src_idx, dst_idx])
                edges.append([dst_idx, src_idx])  # Bidirectional

    if not edges:
        return _build_fully_connected(len(node_names))

    return torch.tensor(edges, dtype=torch.long).T


def _build_fully_connected(num_nodes: int) -> torch.Tensor:
    """Build fully connected edge_index."""
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).T


def load_cluster_metrics(
    prometheus_url: str,
    service_names: List[str],
    start_time: str,
    end_time: str,
    step: str = "1m",
    metrics: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load metrics from a Prometheus server (real cluster data).

    Args:
        prometheus_url: Prometheus server URL (e.g., "http://localhost:9090")
        service_names: List of service/pod names to query
        start_time: Start time (ISO format or relative like "-1h")
        end_time: End time (ISO format or "now")
        step: Query step (e.g., "1m", "5m")
        metrics: List of metric names (default: cpu, memory, latency, error_rate, request_rate)

    Returns:
        X: Tensor [T, N, F] of metrics
        timestamps: List of timestamp strings
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed: pip install requests")

    if metrics is None:
        metrics = [
            "container_cpu_usage_seconds_total",
            "container_memory_usage_bytes",
            "http_request_duration_seconds",
            "http_requests_total",
            "http_request_errors_total",
        ]

    logger.info("Querying Prometheus: %s", prometheus_url)
    logger.info("Services: %s", service_names)
    logger.info("Time range: %s to %s", start_time, end_time)

    num_nodes = len(service_names)
    num_features = len(metrics)

    all_data = []
    timestamps = None

    for metric_idx, metric_name in enumerate(metrics):
        for node_idx, service in enumerate(service_names):
            query = f'{metric_name}{{service="{service}"}}'

            try:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query_range",
                    params={
                        "query": query,
                        "start": start_time,
                        "end": end_time,
                        "step": step,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if data["status"] == "success" and data["data"]["result"]:
                    values = data["data"]["result"][0]["values"]
                    if timestamps is None:
                        timestamps = [v[0] for v in values]
                    metric_values = [float(v[1]) for v in values]
                    all_data.append((node_idx, metric_idx, metric_values))

            except Exception as e:
                logger.warning("Failed to query %s for %s: %s", metric_name, service, e)

    if not all_data or timestamps is None:
        raise ValueError("No data retrieved from Prometheus")

    # Build tensor
    T = len(timestamps)
    X = torch.zeros(T, num_nodes, num_features)

    for node_idx, metric_idx, values in all_data:
        if len(values) == T:
            X[:, node_idx, metric_idx] = torch.tensor(values)

    logger.info("Loaded metrics: shape=%s", X.shape)

    return X, timestamps


# ============================================================================
# Synthetic-to-Real Domain Adaptation
# ============================================================================

def add_real_noise_profile(
    synthetic_data: List[Data],
    real_reference: torch.Tensor,
    noise_ratio: float = 0.3,
) -> List[Data]:
    """
    Add noise profile from real data to synthetic dataset for domain adaptation.

    This helps bridge the gap between synthetic training data and real inference.

    Args:
        synthetic_data: Synthetic dataset (from simulated_data_scalable)
        real_reference: Real data tensor [T, N, F] for noise extraction
        noise_ratio: How much real noise to add (0-1)

    Returns:
        Modified synthetic dataset with real noise characteristics
    """
    # Compute real data statistics
    real_mean = real_reference.mean(dim=0)
    real_std = real_reference.std(dim=0)

    # Compute autocorrelation structure (temporal patterns)
    T = real_reference.shape[0]
    autocorr = torch.zeros_like(real_reference[0])
    for t in range(1, min(T, 10)):
        autocorr += (real_reference[:-t] * real_reference[t:]).mean(dim=0) / t

    logger.info("Applying real noise profile (ratio=%.2f)", noise_ratio)

    modified_data = []
    for d in synthetic_data:
        x = d.x.clone()  # [N, F, W]
        W = x.shape[2]

        # Add real noise characteristics
        for t in range(W):
            noise = torch.randn_like(x[:, :, t]) * real_std * noise_ratio
            # Add temporal correlation
            if t > 0:
                noise += autocorr * x[:, :, t-1] * 0.1 * noise_ratio
            x[:, :, t] += noise

        new_d = Data(x=x, edge_index=d.edge_index.clone())
        # Copy all attributes
        for key in d.keys:
            if key not in ['x', 'edge_index']:
                setattr(new_d, key, getattr(d, key))

        modified_data.append(new_d)

    return modified_data


if __name__ == "__main__":
    # Test RCAEval loading (if installed)
    try:
        train, test, names = load_rcaeval_as_pyg("alibaba")
        print(f"Loaded: {len(train)} train, {len(test)} test, {len(names)} nodes")
        print(f"Sample shape: {train[0].x.shape}")
    except ImportError as e:
        print(f"RCAEval not installed: {e}")
        print("To install: pip install rcaeval")
