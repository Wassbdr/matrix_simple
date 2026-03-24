"""
Unified Data Pipeline for RCA Training.

Handles multiple data sources with consistent format:
- Synthetic V2 (simulated_data_scalable)
- Real cluster metrics (Prometheus/VictoriaMetrics)
- Benchmark datasets (RCAEval)

All sources output the same format:
    Data(x=[N, F, W], edge_index=[2, E], y=[1], root_cause=[str], ...)

Usage:
    from unified_data import UnifiedDataLoader

    loader = UnifiedDataLoader()
    loader.add_synthetic(num_nodes=100, num_samples=8000)
    loader.add_cluster("http://prometheus:9090", duration="7d")
    loader.add_rcaeval("alibaba")

    train, test = loader.get_splits()
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    source_type: str  # "synthetic", "cluster", "rcaeval", "file"
    weight: float = 1.0  # Sampling weight for mixing
    params: Dict = field(default_factory=dict)


class UnifiedDataLoader:
    """
    Unified loader that combines multiple data sources.

    Features:
    - Consistent output format across all sources
    - Automatic normalization (per-source or global)
    - Weighted mixing for domain adaptation
    - Train/test splitting with stratification
    """

    def __init__(
        self,
        window_size: int = 12,
        num_features: int = 5,
        normalize: str = "global",  # "global", "per_source", "none"
        seed: int = 42,
    ):
        self.window_size = window_size
        self.num_features = num_features
        self.normalize = normalize
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.sources: List[DataSourceConfig] = []
        self.datasets: Dict[str, List[Data]] = {}
        self.node_names: Optional[List[str]] = None
        self.num_nodes: Optional[int] = None
        self.scaler_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _feature_presence_from_dim(feature_dim: int) -> torch.Tensor:
        """Build feature-level presence mask for canonical 13D layout."""
        if feature_dim >= 13:
            return torch.ones(13, dtype=torch.float32)
        mask = torch.zeros(13, dtype=torch.float32)
        if feature_dim >= 5:
            mask[:5] = 1.0
        if feature_dim >= 9:
            mask[5:9] = 1.0
        if feature_dim >= 13:
            mask[9:13] = 1.0
        return mask

    def _ensure_feature_layout(
        self,
        dataset: List[Data],
        modality_mask: torch.Tensor,
    ) -> List[Data]:
        """Ensure each sample follows configured feature width with explicit modality masks."""
        if not dataset:
            return dataset

        target_f = self.num_features
        for d in dataset:
            x = d.x  # [N, F, W]
            _, feat_dim, _ = x.shape

            if feat_dim < target_f:
                pad = torch.zeros(x.size(0), target_f - feat_dim, x.size(2), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            elif feat_dim > target_f:
                x = x[:, :target_f, :]

            d.x = x

            # Keep masks explicit so "missing modality" is not confused with true zero values.
            d.modality_mask = modality_mask.clone()
            if target_f == 13:
                d.feature_presence_mask = self._feature_presence_from_dim(feat_dim)
            else:
                d.feature_presence_mask = torch.ones(target_f, dtype=torch.float32)

        return dataset

    def add_synthetic(
        self,
        num_nodes: int = 100,
        num_samples: int = 8000,
        weight: float = 1.0,
        **kwargs,
    ) -> "UnifiedDataLoader":
        """Add synthetic V2 dataset."""
        config = DataSourceConfig(
            name="synthetic_v2",
            source_type="synthetic",
            weight=weight,
            params={
                "num_nodes": num_nodes,
                "num_samples": num_samples,
                "window_size": self.window_size,
                "num_features": self.num_features,
                **kwargs,
            },
        )
        self.sources.append(config)
        self._set_topology(num_nodes)
        return self

    def add_cluster(
        self,
        prometheus_url: str,
        duration: str = "24h",
        services: Optional[List[str]] = None,
        alertmanager_url: Optional[str] = None,
        jaeger_url: Optional[str] = None,
        loki_url: Optional[str] = None,
        elasticsearch_url: Optional[str] = None,
        weight: float = 1.0,
        **kwargs,
    ) -> "UnifiedDataLoader":
        """
        Add real cluster observability data (metrics + traces + logs).

        Args:
            prometheus_url: Prometheus/VictoriaMetrics URL for metrics
            duration: Duration to collect (e.g., "24h", "7d")
            services: List of service names (or None to auto-discover)
            alertmanager_url: Alertmanager URL for anomaly labels
            jaeger_url: Jaeger URL for trace features (optional)
            loki_url: Loki URL for log features (optional)
            elasticsearch_url: Elasticsearch URL for logs (alternative to Loki)
            weight: Sampling weight for mixing

        Feature dimensions:
            - Metrics only: F=5
            - Metrics + Traces: F=9
            - Metrics + Logs: F=9
            - Metrics + Traces + Logs: F=13
        """
        config = DataSourceConfig(
            name="cluster",
            source_type="cluster",
            weight=weight,
            params={
                "prometheus_url": prometheus_url,
                "duration": duration,
                "services": services,
                "alertmanager_url": alertmanager_url,
                "jaeger_url": jaeger_url,
                "loki_url": loki_url,
                "elasticsearch_url": elasticsearch_url,
                "window_size": self.window_size,
                **kwargs,
            },
        )
        self.sources.append(config)
        return self

    def add_rcaeval(
        self,
        dataset_name: str = "alibaba",
        weight: float = 1.0,
    ) -> "UnifiedDataLoader":
        """Add RCAEval benchmark dataset."""
        config = DataSourceConfig(
            name=f"rcaeval_{dataset_name}",
            source_type="rcaeval",
            weight=weight,
            params={
                "dataset_name": dataset_name,
                "window_size": self.window_size,
                "num_features": self.num_features,
            },
        )
        self.sources.append(config)
        return self

    def add_file(
        self,
        filepath: str,
        weight: float = 1.0,
    ) -> "UnifiedDataLoader":
        """Add pre-saved dataset file."""
        config = DataSourceConfig(
            name=f"file_{os.path.basename(filepath)}",
            source_type="file",
            weight=weight,
            params={"filepath": filepath},
        )
        self.sources.append(config)
        return self

    def _set_topology(self, num_nodes: int) -> None:
        """Set expected topology size."""
        if self.num_nodes is None:
            self.num_nodes = num_nodes
        elif self.num_nodes != num_nodes:
            logger.warning(
                "Topology mismatch: expected %d nodes, got %d. Will adapt.",
                self.num_nodes, num_nodes,
            )

    def _load_synthetic(self, config: DataSourceConfig) -> List[Data]:
        """Load synthetic V2 dataset."""
        try:
            from simulated_data_scalable import generate_scalable_dataset_v2
        except ImportError:
            from rca_pipeline.simulated_data_scalable import generate_scalable_dataset_v2

        params = config.params.copy()
        # Synthetic generator is metric-centric in this project; traces/logs are represented via explicit zero-padding.
        params["num_features"] = min(params.get("num_features", self.num_features), 5)
        params.setdefault("seed", self.seed)
        params.setdefault("seasonal_noise", True)
        params.setdefault("correlated_features", True)

        dataset, node_names, edge_index, metadata = generate_scalable_dataset_v2(**params)

        if self.node_names is None:
            self.node_names = node_names
            self.num_nodes = len(node_names)

        dataset = self._ensure_feature_layout(
            dataset,
            modality_mask=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        )

        logger.info("Loaded synthetic: %d samples", len(dataset))
        return dataset

    def _load_cluster(self, config: DataSourceConfig) -> List[Data]:
        """
        Load cluster observability data with multi-modal fusion.

        Combines:
        - Metrics from Prometheus (5 features)
        - Traces from Jaeger (4 features, optional)
        - Logs from Loki/Elasticsearch (4 features, optional)

        Returns Data objects with x shape [N, F, W] where F varies:
        - Metrics only: F=5
        - Metrics + Traces: F=9
        - Metrics + Logs: F=9
        - Full multi-modal: F=13
        """
        try:
            from collect_cluster_data import (
                collect_service_metrics,
                collect_trace_features,
                create_windows,
                discover_services,
                get_alerts_in_range,
                parse_duration,
            )
        except ImportError:
            try:
                from rca_pipeline.collect_cluster_data import (
                    collect_service_metrics,
                    collect_trace_features,
                    create_windows,
                    discover_services,
                    get_alerts_in_range,
                    parse_duration,
                )
            except ImportError:
                logger.error("collect_cluster_data.py not found")
                return []

        from datetime import datetime

        params = config.params
        prometheus_url = params["prometheus_url"]
        duration = parse_duration(params["duration"])

        end = datetime.utcnow()
        start = end - duration

        # Discover or use provided services
        services = params.get("services")
        if services is None:
            services = discover_services(prometheus_url)
            if not services:
                logger.error("No services discovered from Prometheus")
                return []
            services = services[:self.num_nodes] if self.num_nodes else services[:50]

        step = params.get("step", "1m")

        # ========================================
        # Collect Metrics (always required)
        # ========================================
        try:
            X_metrics, timestamps, services = collect_service_metrics(
                prometheus_url, services, start, end, step=step
            )
            logger.info("Collected metrics: shape=%s", X_metrics.shape)
        except Exception as e:
            logger.error("Failed to collect cluster metrics: %s", e)
            return []

        # ========================================
        # Collect Trace Features (optional)
        # ========================================
        X_traces = None
        jaeger_url = params.get("jaeger_url")
        if jaeger_url:
            try:
                X_traces, trace_ts = collect_trace_features(
                    jaeger_url, services, start, end, step=step
                )
                # Align timestamps
                X_traces = self._align_to_timestamps(X_traces, trace_ts, timestamps)
                logger.info("Collected trace features: shape=%s", X_traces.shape)
            except Exception as e:
                logger.warning("Failed to collect trace features: %s", e)
                X_traces = None

        # ========================================
        # Collect Log Features (optional)
        # ========================================
        X_logs = None
        loki_url = params.get("loki_url")
        elasticsearch_url = params.get("elasticsearch_url")
        if loki_url or elasticsearch_url:
            try:
                try:
                    from collect_log_data import collect_log_features
                except ImportError:
                    from rca_pipeline.collect_log_data import collect_log_features

                X_logs, log_ts = collect_log_features(
                    loki_url=loki_url,
                    services=services,
                    start=start,
                    end=end,
                    step=step,
                    elasticsearch_url=elasticsearch_url,
                )
                # Align timestamps
                X_logs = self._align_to_timestamps(X_logs, log_ts, timestamps)
                logger.info("Collected log features: shape=%s", X_logs.shape)
            except Exception as e:
                logger.warning("Failed to collect log features: %s", e)
                X_logs = None

        # ========================================
        # Fuse Features
        # ========================================
        has_traces = X_traces is not None
        has_logs = X_logs is not None

        # Canonical explicit fusion layout: metrics(5) + traces(4) + logs(4) = 13
        if not has_traces:
            X_traces = torch.zeros(X_metrics.size(0), X_metrics.size(1), 4, dtype=X_metrics.dtype)
        if not has_logs:
            X_logs = torch.zeros(X_metrics.size(0), X_metrics.size(1), 4, dtype=X_metrics.dtype)

        X = torch.cat([X_metrics, X_traces, X_logs], dim=2)
        total_features = X.shape[2]
        feature_names = ["metrics"]
        if has_traces:
            feature_names.append("traces")
        if has_logs:
            feature_names.append("logs")

        logger.info("Fused features: %s -> total F=%d", feature_names, total_features)

        # ========================================
        # Get Alerts for Labeling
        # ========================================
        alerts = []
        if params.get("alertmanager_url"):
            alerts = get_alerts_in_range(params["alertmanager_url"], start, end)

        # ========================================
        # Create Windows
        # ========================================
        dataset = create_windows(
            X=X,
            timestamps=timestamps,
            window_size=params.get("window_size", self.window_size),
            stride=params.get("stride", 6),
            alerts=alerts,
            services=services,
        )

        modality_mask = torch.tensor([1.0, 1.0 if has_traces else 0.0, 1.0 if has_logs else 0.0], dtype=torch.float32)
        feature_presence_mask = torch.tensor(
            [1.0] * 5 + ([1.0] * 4 if has_traces else [0.0] * 4) + ([1.0] * 4 if has_logs else [0.0] * 4),
            dtype=torch.float32,
        )

        # Tag with modality info
        for d in dataset:
            d.modalities = feature_names
            d.num_features = total_features
            d.modality_mask = modality_mask.clone()
            d.feature_presence_mask = feature_presence_mask.clone()

        # Adapt topology if needed
        dataset = self._adapt_topology(dataset, services)

        dataset = self._ensure_feature_layout(dataset, modality_mask=modality_mask)

        logger.info("Loaded cluster: %d samples (modalities: %s)", len(dataset), feature_names)
        return dataset

    def _align_to_timestamps(
        self,
        tensor: torch.Tensor,
        source_ts: List[float],
        target_ts: List[float],
    ) -> torch.Tensor:
        """
        Align tensor to target timestamps using linear interpolation.

        Args:
            tensor: Source tensor [T_src, N, F]
            source_ts: Source timestamps
            target_ts: Target timestamps

        Returns:
            Aligned tensor [T_tgt, N, F]
        """
        T_src, N, F = tensor.shape
        T_tgt = len(target_ts)

        if T_src == T_tgt and len(source_ts) == T_tgt:
            # Check if timestamps are close enough
            if all(abs(s - t) < 60 for s, t in zip(source_ts, target_ts)):
                return tensor

        import numpy as np

        aligned = torch.zeros(T_tgt, N, F)
        source_arr = np.array(source_ts)
        target_arr = np.array(target_ts)

        for n in range(N):
            for f in range(F):
                values = tensor[:, n, f].numpy()
                aligned_values = np.interp(target_arr, source_arr, values)
                aligned[:, n, f] = torch.tensor(aligned_values)

        return aligned

    def _load_rcaeval(self, config: DataSourceConfig) -> List[Data]:
        """Load RCAEval benchmark dataset."""
        try:
            from load_rcaeval import load_rcaeval_as_pyg
        except ImportError:
            try:
                from rca_pipeline.load_rcaeval import load_rcaeval_as_pyg
            except ImportError:
                logger.error("load_rcaeval.py not found or rcaeval not installed")
                return []

        params = config.params
        try:
            train, test, node_names = load_rcaeval_as_pyg(
                dataset_name=params["dataset_name"],
                window_size=params.get("window_size", self.window_size),
                num_features=min(params.get("num_features", self.num_features), 5),
            )
            dataset = train + test

            # Adapt topology
            dataset = self._adapt_topology(dataset, node_names)

            dataset = self._ensure_feature_layout(
                dataset,
                modality_mask=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            )

            logger.info("Loaded RCAEval %s: %d samples", params["dataset_name"], len(dataset))
            return dataset
        except Exception as e:
            logger.error("Failed to load RCAEval: %s", e)
            return []

    def _load_file(self, config: DataSourceConfig) -> List[Data]:
        """Load dataset from file."""
        filepath = config.params["filepath"]

        if not os.path.exists(filepath):
            logger.error("File not found: %s", filepath)
            return []

        payload = torch.load(filepath, weights_only=False)

        if isinstance(payload, dict):
            if "dataset" in payload:
                dataset = payload["dataset"]
            elif "train" in payload and "test" in payload:
                dataset = payload["train"] + payload["test"]
            else:
                logger.error("Unknown file format: %s", filepath)
                return []

            if "node_names" in payload and self.node_names is None:
                self.node_names = payload["node_names"]
                self.num_nodes = len(self.node_names)
        elif isinstance(payload, list):
            dataset = payload
        else:
            logger.error("Unknown file format: %s", filepath)
            return []

        dataset = self._ensure_feature_layout(
            dataset,
            modality_mask=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        )

        logger.info("Loaded file %s: %d samples", filepath, len(dataset))
        return dataset

    def _adapt_topology(
        self,
        dataset: List[Data],
        source_node_names: List[str],
    ) -> List[Data]:
        """
        Adapt dataset to target topology size.

        If source has fewer nodes: pad with zeros
        If source has more nodes: truncate or project
        """
        if self.num_nodes is None or len(source_node_names) == self.num_nodes:
            return dataset

        target_N = self.num_nodes
        source_N = len(source_node_names)

        adapted = []
        for d in dataset:
            x = d.x  # [N, F, W]
            N, F, W = x.shape

            if N < target_N:
                # Pad with zeros
                pad = torch.zeros(target_N - N, F, W)
                x_new = torch.cat([x, pad], dim=0)
            elif N > target_N:
                # Truncate (keep first target_N nodes)
                x_new = x[:target_N]
            else:
                x_new = x

            # Create new edge index for target topology
            # Simple fully connected for now (can be improved)
            edges = []
            for i in range(target_N):
                for j in range(target_N):
                    if i != j:
                        edges.append([i, j])
            edge_index = torch.tensor(edges, dtype=torch.long).T

            new_d = Data(x=x_new, edge_index=edge_index)

            # Copy attributes
            for key in d.keys():
                if key not in ['x', 'edge_index', 'root_cause']:
                    setattr(new_d, key, getattr(d, key))

            # Adapt root_cause if present
            if hasattr(d, 'root_cause') and d.root_cause and self.node_names is not None:
                new_rc = []
                for rc in getattr(d, 'root_cause'):
                    if rc in source_node_names:
                        idx = source_node_names.index(rc)
                        if idx < target_N:
                            new_rc.append(self.node_names[idx])
                new_d.root_cause = new_rc
            elif hasattr(d, 'root_cause'):
                new_d.root_cause = d.root_cause

            # Mark as adapted
            new_d.source_topology_size = N
            new_d.adapted = True

            adapted.append(new_d)

        logger.info("Adapted %d samples from %d to %d nodes", len(adapted), source_N, target_N)
        return adapted

    def _normalize_dataset(self, dataset: List[Data]) -> List[Data]:
        """DEPRECATED: normalizes train+test together, causing data leakage.

        DO NOT USE. Use fit_scaler(train) + apply_scaler(data, stats) instead.
        This method is kept only to avoid ImportError in legacy scripts that
        may reference it, but it will raise immediately if called.
        """
        raise RuntimeError(
            "_normalize_dataset() applies normalization on the ENTIRE dataset "
            "(train + test together), causing data leakage that inflates scores. "
            "Use fit_scaler(train_data) + apply_scaler(data, stats) instead. "
            "See benchmark.py for the correct pattern."
        )

    def load_all(self) -> "UnifiedDataLoader":
        """Load all configured data sources."""
        for config in self.sources:
            if config.source_type == "synthetic":
                data = self._load_synthetic(config)
            elif config.source_type == "cluster":
                data = self._load_cluster(config)
            elif config.source_type == "rcaeval":
                data = self._load_rcaeval(config)
            elif config.source_type == "file":
                data = self._load_file(config)
            else:
                logger.warning("Unknown source type: %s", config.source_type)
                continue

            if data:
                # Tag source
                for d in data:
                    d.source = config.name
                    d.source_weight = config.weight

                self.datasets[config.name] = data

        # Important: we intentionally do NOT normalize in load_all.
        # Normalization must be fit on train only and then applied to both train/test.

        return self

    def fit_scaler(self, train_data: List[Data]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Fit per-source or global scaler stats on train split only."""
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if not train_data:
            self.scaler_stats = stats
            return stats

        if self.normalize == "none":
            self.scaler_stats = stats
            return stats

        if self.normalize == "global":
            xs = torch.cat([d.x for d in train_data], dim=0)  # [sumN, F, W]
            mean = xs.mean(dim=(0, 2), keepdim=True)
            std = xs.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            stats["__global__"] = (mean, std)
            self.scaler_stats = stats
            return stats

        # per_source
        by_source: Dict[str, List[Data]] = defaultdict(list)
        for d in train_data:
            by_source[getattr(d, "source", "unknown")].append(d)

        for source, ds in by_source.items():
            xs = torch.cat([d.x for d in ds], dim=0)
            mean = xs.mean(dim=(0, 2), keepdim=True)
            std = xs.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            stats[source] = (mean, std)

        self.scaler_stats = stats
        return stats

    def apply_scaler(
        self,
        data: List[Data],
        stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> List[Data]:
        """Apply pre-fitted scaler stats to a split."""
        if self.normalize == "none":
            return data

        stats = stats if stats is not None else self.scaler_stats
        if not stats:
            raise ValueError("Scaler stats are empty. Call fit_scaler(train_data) first.")

        out: List[Data] = []
        for d in data:
            nd = d.clone()
            source = getattr(nd, "source", "unknown")
            if self.normalize == "global":
                mean, std = stats["__global__"]
            else:
                if source not in stats:
                    # Fallback to global-like aggregate if a source exists only in test split.
                    all_means = torch.stack([m for m, _ in stats.values()], dim=0).mean(dim=0)
                    all_stds = torch.stack([s for _, s in stats.values()], dim=0).mean(dim=0).clamp(min=1e-8)
                    mean, std = all_means, all_stds
                else:
                    mean, std = stats[source]
            nd.x = (nd.x - mean) / std
            out.append(nd)

        return out

    def get_splits(
        self,
        train_ratio: float = 0.8,
        stratify_by: str = "source",  # "source", "pattern", "both"
        balance_sources: bool = True,
    ) -> Tuple[List[Data], List[Data]]:
        """
        Get train/test splits with optional stratification.

        Args:
            train_ratio: Fraction for training
            stratify_by: Stratification key
            balance_sources: Whether to balance samples across sources by weight

        Returns:
            train_data, test_data
        """
        if not self.datasets:
            self.load_all()

        train_data = []
        test_data = []

        for name, data in self.datasets.items():
            # Shuffle
            indices = self.rng.permutation(len(data))

            # Apply source weight for balancing
            if balance_sources:
                config = next((c for c in self.sources if c.name == name), None)
                weight = config.weight if config else 1.0
                # Subsample if weight < 1
                if weight < 1.0:
                    n_keep = int(len(data) * weight)
                    indices = indices[:n_keep]

            split_idx = int(len(indices) * train_ratio)

            for i in indices[:split_idx]:
                train_data.append(data[i])
            for i in indices[split_idx:]:
                test_data.append(data[i])

        # Final shuffle
        self.rng.shuffle(train_data)
        self.rng.shuffle(test_data)

        logger.info("Split: %d train, %d test", len(train_data), len(test_data))

        # Log source distribution
        train_sources = defaultdict(int)
        test_sources = defaultdict(int)
        for d in train_data:
            train_sources[getattr(d, 'source', 'unknown')] += 1
        for d in test_data:
            test_sources[getattr(d, 'source', 'unknown')] += 1

        logger.info("Train sources: %s", dict(train_sources))
        logger.info("Test sources: %s", dict(test_sources))

        return train_data, test_data

    def save(self, filepath: str) -> None:
        """Save unified dataset to file."""
        train, test = self.get_splits()

        torch.save({
            "train": train,
            "test": test,
            "node_names": self.node_names,
            "sources": [s.name for s in self.sources],
            "version": "unified_v1",
        }, filepath)

        logger.info("Saved unified dataset to %s", filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.datasets:
            self.load_all()

        stats = {
            "num_sources": len(self.datasets),
            "sources": {},
            "total_samples": 0,
            "num_nodes": self.num_nodes,
            "window_size": self.window_size,
            "num_features": self.num_features,
        }

        for name, data in self.datasets.items():
            n_anomalies = sum(1 for d in data if d.y.item() == 1)
            n_healthy = len(data) - n_anomalies

            patterns = defaultdict(int)
            for d in data:
                pt = getattr(d, 'pattern_type', 'unknown')
                patterns[pt] += 1

            stats["sources"][name] = {
                "total": len(data),
                "anomalies": n_anomalies,
                "healthy": n_healthy,
                "patterns": dict(patterns),
            }
            stats["total_samples"] += len(data)

        return stats


def create_hybrid_dataset(
    num_nodes: int = 100,
    synthetic_samples: int = 8000,
    prometheus_url: Optional[str] = None,
    jaeger_url: Optional[str] = None,
    loki_url: Optional[str] = None,
    cluster_duration: str = "7d",
    output_path: str = "hybrid_dataset.pt",
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[str]]:
    """
    Convenience function to create a hybrid multi-modal dataset.

    Args:
        num_nodes: Number of nodes in topology
        synthetic_samples: Number of synthetic samples
        prometheus_url: Prometheus URL (None to skip cluster data)
        jaeger_url: Jaeger URL for trace features (optional)
        loki_url: Loki URL for log features (optional)
        cluster_duration: Duration for cluster data collection
        output_path: Where to save the dataset
        seed: Random seed

    Returns:
        train_data, test_data, node_names
    """
    loader = UnifiedDataLoader(seed=seed)

    # Always add synthetic
    loader.add_synthetic(
        num_nodes=num_nodes,
        num_samples=synthetic_samples,
        weight=1.0,
        recurrence_ratio=0.8,
        multi_root_prob=0.15,
    )

    # Add cluster if URL provided
    if prometheus_url:
        loader.add_cluster(
            prometheus_url=prometheus_url,
            jaeger_url=jaeger_url,
            loki_url=loki_url,
            duration=cluster_duration,
            weight=0.5,  # Less weight to avoid overfitting to specific cluster
        )

    # Load and save
    loader.load_all()

    stats = loader.get_statistics()
    logger.info("Dataset statistics: %s", stats)

    train, test = loader.get_splits()

    loader.save(output_path)

    return train, test, loader.node_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create unified multi-modal dataset")
    parser.add_argument("--num-nodes", type=int, default=100)
    parser.add_argument("--synthetic-samples", type=int, default=8000)
    parser.add_argument("--prometheus-url", type=str, default=None)
    parser.add_argument("--jaeger-url", type=str, default=None,
                        help="Jaeger URL for trace features")
    parser.add_argument("--loki-url", type=str, default=None,
                        help="Loki URL for log features")
    parser.add_argument("--cluster-duration", type=str, default="7d")
    parser.add_argument("--output", type=str, default="hybrid_dataset.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, test, names = create_hybrid_dataset(
        num_nodes=args.num_nodes,
        synthetic_samples=args.synthetic_samples,
        prometheus_url=args.prometheus_url,
        jaeger_url=args.jaeger_url,
        loki_url=args.loki_url,
        cluster_duration=args.cluster_duration,
        output_path=args.output,
        seed=args.seed,
    )

    # Print feature info
    if train:
        sample = train[0]
        modalities = getattr(sample, 'modalities', ['metrics'])
        num_features = sample.x.shape[1]
        print(f"\n=== Dataset Ready ===")
        print(f"Train: {len(train)} samples")
        print(f"Test: {len(test)} samples")
        print(f"Nodes: {len(names)}")
        print(f"Features: {num_features} (modalities: {modalities})")
        print(f"Saved to: {args.output}")
