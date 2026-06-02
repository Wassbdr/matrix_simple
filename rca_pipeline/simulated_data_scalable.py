"""
Scalable Simulated RCA Dataset -- Barabasi-Albert microservice topologies.

Generates realistic microservice architectures with N = 50 to 200 nodes
using the Barabasi-Albert preferential attachment model.

Why Barabasi-Albert?
    Real microservice deployments exhibit power-law degree distributions:
    a few critical nodes (databases, message queues, auth services) have
    many connections, while most leaf services have only 2-3 dependencies.
    BA(N, m) naturally produces this pattern.

Topology construction:
    1. nx.barabasi_albert_graph(N, m) produces an undirected power-law graph.
    2. Edges are directed: higher-degree node = infrastructure/dependency.
       If deg(u) > deg(v), the edge becomes v -> u (v depends on u).
    3. A propagation matrix P is built: anomaly at node i propagates to
       all nodes that depend (directly or transitively) on i.

Anomaly dynamics (same formulation as the 5/20-node datasets):
    X(t+1) = f(A, X(t)) + B * U(t)
    - Root cause: transient exponential decay at a single node.
    - Cascade: multi-hop propagation via P with stochastic jitter.
    - Distractors: random spikes with 1-hop propagation.
    - Feature subset: only some features react per anomaly type.

Usage:
    python simulated_data_scalable.py --num-nodes 100 --num-samples 5000
"""

import argparse
import logging
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Log feature indices (from log_feature_extractor.py)
# 0: error_freq, 1: warn_freq, 2: log_rate, 3: error_burst
# 4: unique_errors, 5: oom_flag, 6: restart_flag, 7: timeout_flag
N_LOG_FEATURES = 8

# Trace feature indices (from trace_feature_extractor.py)
# 0: call_depth, 1: error_span_ratio, 2: p99_span_ms
# 3: upstream_errors, 4: downstream_errors, 5: retry_count
N_TRACE_FEATURES = 6


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------

def generate_ba_topology(
    num_nodes: int,
    m: int = 3,
    seed: int = 42,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, nx.DiGraph]:
    """
    Generate a directed Barabasi-Albert microservice topology.

    Args:
        num_nodes: total number of service nodes (50-200)
        m:         number of edges each new node attaches to (2-4 typical)
        seed:      random seed for reproducibility
    Returns:
        node_names:  List[str] of length N
        edge_index:  [2, 2*E] bidirectional (for GCN message passing)
        P:           [N, N] propagation matrix (anomaly flows)
        G_directed:  networkx DiGraph for analysis
    """
    rng = np.random.RandomState(seed)
    N = num_nodes

    # Step 1: undirected BA graph
    G_undirected = nx.barabasi_albert_graph(N, m, seed=seed)

    # Step 2: assign tiers via betweenness centrality
    centrality = nx.betweenness_centrality(G_undirected)
    degree = dict(G_undirected.degree())
    sorted_by_centrality = sorted(range(N), key=lambda n: centrality[n], reverse=True)

    # Tier assignment (cosmetic naming)
    tier = {}
    tier_counters = {"db": 0, "cache": 0, "mq": 0, "svc": 0, "gw": 0, "api": 0}
    for rank, node in enumerate(sorted_by_centrality):
        frac = rank / N
        if frac < 0.10:
            t = rng.choice(["db", "cache", "mq"])
        elif frac < 0.65:
            t = "svc"
        else:
            t = rng.choice(["gw", "api"])
        tier_counters[t] += 1
        tier[node] = t

    node_names = [f"{tier[i]}-{i:03d}" for i in range(N)]

    # Step 3: direct edges -- higher centrality node is the dependency
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(range(N))
    directed_edges = []
    for u, v in G_undirected.edges():
        if centrality[u] >= centrality[v]:
            directed_edges.append((v, u))       # v depends on u
        else:
            directed_edges.append((u, v))       # u depends on v
    G_directed.add_edges_from(directed_edges)

    # Step 4: bidirectional edge_index for GCN (message passing both ways)
    src, dst = [], []
    for s, d in directed_edges:
        src.extend([s, d])
        dst.extend([d, s])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Step 5: propagation matrix P
    #   P[cause, victim] > 0 means anomaly at 'cause' propagates to 'victim'.
    #   Anomaly propagates to direct dependents (reverse of dependency edge).
    P = torch.zeros(N, N)
    for s, d in directed_edges:
        # s depends on d  =>  anomaly at d propagates to s
        P[d, s] = 1.0
    # Weak back-pressure: anomaly at s slightly affects d
    for s, d in directed_edges:
        if P[s, d] == 0:
            P[s, d] = 0.2

    logger.info(
        "Topology: N=%d, E=%d (directed), m=%d, avg_degree=%.1f",
        N, len(directed_edges), m, 2 * len(directed_edges) / N,
    )

    return node_names, edge_index, P, G_directed


# ---------------------------------------------------------------------------
# Multimodal data generation: logs and traces
# ---------------------------------------------------------------------------

def generate_log_features(
    rng: np.random.RandomState,
    N: int,
    is_anomaly: bool,
    root_cause_idx: int,
    failure_type: str,
    cascade_nodes: List[int],
    signal_strength: float,
) -> np.ndarray:
    """
    Generate synthetic log features [N, 8] correlated with anomaly state.

    Log features (from log_feature_extractor.py):
        0: error_freq      - errors / total logs
        1: warn_freq       - warnings / total logs
        2: log_rate        - total logs / window duration
        3: error_burst     - max consecutive errors
        4: unique_errors   - unique error message count (normalized)
        5: oom_flag        - 1.0 if OOM detected
        6: restart_flag    - 1.0 if restart/CrashLoopBackOff
        7: timeout_flag    - 1.0 if timeout detected
    """
    logs = np.zeros((N, N_LOG_FEATURES), dtype=np.float32)

    # Baseline: low noise for all nodes
    logs[:, 0] = rng.uniform(0.0, 0.02, N)  # error_freq
    logs[:, 1] = rng.uniform(0.0, 0.05, N)  # warn_freq
    logs[:, 2] = rng.uniform(0.5, 2.0, N)   # log_rate
    logs[:, 3] = rng.uniform(0.0, 0.05, N)  # error_burst
    logs[:, 4] = rng.uniform(0.0, 0.1, N)   # unique_errors

    if not is_anomaly:
        return logs

    # Root cause: elevated error indicators
    rc = root_cause_idx
    intensity = min(1.0, signal_strength / 5.0)

    logs[rc, 0] = 0.3 + rng.uniform(0.0, 0.4) * intensity   # error_freq
    logs[rc, 1] = 0.2 + rng.uniform(0.0, 0.3) * intensity   # warn_freq
    logs[rc, 2] = 3.0 + rng.uniform(0.0, 5.0) * intensity   # log_rate spike
    logs[rc, 3] = 0.3 + rng.uniform(0.0, 0.5) * intensity   # error_burst
    logs[rc, 4] = 0.4 + rng.uniform(0.0, 0.4) * intensity   # unique_errors

    # Failure-specific flags
    if failure_type == "gray":
        # Gray failures: subtle, no hard errors but warnings and timeouts
        logs[rc, 0] = 0.05 + rng.uniform(0.0, 0.1)  # low error_freq
        logs[rc, 1] = 0.4 + rng.uniform(0.0, 0.3)   # high warn_freq
        logs[rc, 7] = rng.choice([0.0, 1.0], p=[0.4, 0.6])  # timeout_flag
    elif failure_type == "hard":
        # Hard failures: OOM/restart signatures are dominant and mostly local.
        logs[rc, 5] = rng.choice([0.0, 1.0], p=[0.3, 0.7])  # oom_flag
        logs[rc, 6] = rng.choice([0.0, 1.0], p=[0.2, 0.8])  # restart_flag
        logs[rc, 7] = rng.choice([0.0, 1.0], p=[0.3, 0.7])  # timeout_flag
    else:
        # Cascade failures: noisy errors/warnings propagate with lower hard-crash indicators.
        logs[rc, 5] = rng.choice([0.0, 1.0], p=[0.85, 0.15])  # oom_flag
        logs[rc, 6] = rng.choice([0.0, 1.0], p=[0.8, 0.2])    # restart_flag
        logs[rc, 7] = rng.choice([0.0, 1.0], p=[0.5, 0.5])    # timeout_flag

    # Cascade nodes: elevated warnings/errors but less than root cause
    for node in cascade_nodes:
        if node == rc:
            continue
        cascade_intensity = intensity * 0.5
        logs[node, 0] = min(1.0, logs[node, 0] + 0.1 + rng.uniform(0.0, 0.2) * cascade_intensity)
        logs[node, 1] = min(1.0, logs[node, 1] + 0.15 + rng.uniform(0.0, 0.2) * cascade_intensity)
        logs[node, 2] = logs[node, 2] + rng.uniform(0.5, 2.0) * cascade_intensity
        logs[node, 7] = rng.choice([0.0, 1.0], p=[0.7, 0.3])  # timeout propagation

    return logs


def generate_trace_features(
    rng: np.random.RandomState,
    N: int,
    G: nx.DiGraph,
    is_anomaly: bool,
    root_cause_idx: int,
    failure_type: str,
    cascade_nodes: List[int],
    signal_strength: float,
) -> np.ndarray:
    """
    Generate synthetic trace features [N, 6] correlated with anomaly state.

    Trace features (from trace_feature_extractor.py):
        0: call_depth       - mean call depth
        1: error_span_ratio - fraction of error spans
        2: p99_span_ms      - P99 latency
        3: upstream_errors  - errors from parent services
        4: downstream_errors - errors in child services
        5: retry_count      - estimated retries
    """
    traces = np.zeros((N, N_TRACE_FEATURES), dtype=np.float32)

    # Baseline: normal operation
    for node in range(N):
        depth = len(nx.ancestors(G, node)) if node in G else 0
        traces[node, 0] = depth + rng.uniform(0.0, 1.0)      # call_depth
        traces[node, 1] = rng.uniform(0.0, 0.02)             # error_span_ratio
        traces[node, 2] = 10.0 + rng.uniform(0.0, 30.0)      # p99_span_ms (10-40ms normal)
        traces[node, 3] = rng.uniform(0.0, 0.5)              # upstream_errors
        traces[node, 4] = rng.uniform(0.0, 0.5)              # downstream_errors
        traces[node, 5] = rng.uniform(0.0, 1.0)              # retry_count

    if not is_anomaly:
        return traces

    rc = root_cause_idx
    intensity = min(1.0, signal_strength / 5.0)

    # Root cause: high latency, errors, retries
    traces[rc, 1] = 0.3 + rng.uniform(0.0, 0.5) * intensity   # error_span_ratio
    traces[rc, 2] = 100.0 + rng.uniform(0.0, 500.0) * intensity  # p99_span_ms spike
    traces[rc, 5] = 3.0 + rng.uniform(0.0, 5.0) * intensity   # retry_count

    if failure_type == "gray":
        # Gray failures: latency degradation without errors
        traces[rc, 1] = 0.05 + rng.uniform(0.0, 0.1)  # low error_span_ratio
        traces[rc, 2] = 80.0 + rng.uniform(0.0, 200.0)  # elevated but not extreme p99
    elif failure_type == "hard":
        # Hard failures: abrupt local collapse.
        traces[rc, 1] = 0.5 + rng.uniform(0.0, 0.4)   # high error ratio
        traces[rc, 2] = 200.0 + rng.uniform(0.0, 700.0)  # very high p99
        traces[rc, 5] = 4.0 + rng.uniform(0.0, 6.0)   # high retries

    # Cascade: propagate latency and errors
    for node in cascade_nodes:
        if node == rc:
            continue
        cascade_intensity = intensity * 0.6

        # Upstream/downstream error propagation
        if G.has_edge(node, rc):  # node depends on rc
            traces[node, 3] = 1.0 + rng.uniform(0.0, 3.0) * cascade_intensity  # upstream_errors
        if G.has_edge(rc, node):  # rc depends on node (back-pressure)
            traces[node, 4] = 1.0 + rng.uniform(0.0, 2.0) * cascade_intensity  # downstream_errors

        traces[node, 1] = min(1.0, traces[node, 1] + 0.1 * cascade_intensity)
        traces[node, 2] = traces[node, 2] + rng.uniform(20.0, 100.0) * cascade_intensity
        traces[node, 5] = traces[node, 5] + rng.uniform(0.0, 2.0) * cascade_intensity

    return traces


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_scalable_dataset(
    num_nodes: int = 100,
    ba_m: int = 3,
    num_samples: int = 5000,
    window_size: int = 12,
    num_features: int = 5,
    anomaly_ratio: float = 0.4,
    noise_level: float = 0.1,
    max_cascade_hops: int = 6,
    seed: int = 42,
    gray_failure_ratio: float = 0.15,
    hard_failure_ratio: float = 0.35,
    generate_multimodal: bool = True,
) -> Tuple[List[Data], List[str], torch.Tensor]:
    """
    Generate a scalable simulated RCA dataset on a BA topology.

    Args:
        num_nodes:          N in {50, 100, 150, 200}
        ba_m:               BA attachment parameter (2-4)
        num_samples:        total windows
        window_size:        temporal window W
        num_features:       per-node metric features F
        anomaly_ratio:      fraction of anomalous windows
        noise_level:        background noise sigma
        max_cascade_hops:   max propagation depth
        seed:               reproducibility
        gray_failure_ratio: fraction of anomalies that are gray failures
        hard_failure_ratio: fraction of anomalies that are hard failures
        generate_multimodal: if True, generate logs [N,8] and traces [N,6] per sample

    Returns:
        dataset:     List[Data]  with attributes:
                     .x          [N, F, W] metrics
                     .x_logs     [N, 8] log features (if generate_multimodal)
                     .x_traces   [N, 6] trace features (if generate_multimodal)
                     .edge_index, .y, .root_cause, .failure_type
        node_names:  List[str]
        edge_index:  [2, E]
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if gray_failure_ratio < 0.0 or hard_failure_ratio < 0.0:
        raise ValueError("gray_failure_ratio and hard_failure_ratio must be >= 0")
    if gray_failure_ratio + hard_failure_ratio > 1.0:
        raise ValueError("gray_failure_ratio + hard_failure_ratio must be <= 1")

    node_names, edge_index, P_base, G = generate_ba_topology(num_nodes, ba_m, seed)
    N = num_nodes

    # Adjacency for distractor propagation (symmetric, weighted)
    A_raw = torch.zeros(N, N)
    for s, d in G.edges():
        w = 0.6 + rng.rand() * 0.4
        A_raw[s, d] = w
        A_raw[d, s] = w * 0.3
    for i in range(N):
        A_raw[i, i] = 1.0
    neighbor_mask = A_raw > 0
    neighbor_mask.fill_diagonal_(False)

    dataset: List[Data] = []

    for sample_idx in range(num_samples):
        is_anomaly = rng.rand() < anomaly_ratio

        X = torch.randn(window_size, N, num_features) * noise_level

        # ---- Distractor spikes (reduced strength to improve SNR) ----
        num_distractors = rng.randint(2, max(3, N // 15))
        for _ in range(num_distractors):
            t = rng.randint(0, window_size - 1)
            node = rng.randint(0, N)
            n_feats = rng.randint(1, max(2, num_features))
            feats = sorted(int(f) for f in rng.choice(num_features, size=n_feats, replace=False))
            strength = float(0.5 + rng.rand() * 1.5)
            X[t, node, feats] += strength
            # 1-hop propagation (weak)
            if t + 1 < window_size:
                dst_idx = torch.nonzero(neighbor_mask[node], as_tuple=False).squeeze(-1)
                if dst_idx.numel() > 0:
                    for feat in feats:
                        X[t + 1, dst_idx, feat] += strength * 0.15

        root_cause_idx = -1
        failure_type = "healthy"
        signal_strength = 0.0
        cascade_nodes: List[int] = []

        if is_anomaly:
            root_cause_idx = rng.randint(0, N)
            rv = rng.rand()
            if rv < gray_failure_ratio:
                failure_type = "gray"
            elif rv < gray_failure_ratio + hard_failure_ratio:
                failure_type = "hard"
            else:
                failure_type = "cascade"

            # Feature subset affected
            n_affected = rng.randint(2, max(3, num_features))
            affected_feats = sorted(rng.choice(num_features, size=n_affected, replace=False))

            # Root cause: transient exponential decay
            # Gray failures: 10× slower decay, weaker signal (collective anomaly)
            t_start = rng.randint(0, max(1, window_size - 4))
            if failure_type == "gray":
                signal_strength = 0.5 + rng.rand() * 1.0
                decay = 0.85 + rng.rand() * 0.1
            elif failure_type == "hard":
                signal_strength = 3.5 + rng.rand() * 4.0
                decay = 0.1 + rng.rand() * 0.15
            else:
                signal_strength = 2.5 + rng.rand() * 3.0
                decay = 0.4 + rng.rand() * 0.2

            X[t_start, root_cause_idx, affected_feats] += (
                signal_strength + torch.randn(len(affected_feats)) * 0.3
            )
            for dt in range(1, window_size - t_start):
                remaining = signal_strength * (decay ** dt)
                if remaining < 0.05:
                    break
                X[t_start + dt, root_cause_idx, affected_feats] += (
                    remaining + torch.randn(len(affected_feats)) * 0.1
                )

            cascade_nodes = [root_cause_idx]  # Track affected nodes
            if failure_type == "cascade":
                # Stochastic cascade via propagation matrix
                P_sample = P_base * (0.5 + rng.rand())
                jitter = torch.from_numpy(rng.uniform(0.7, 1.3, size=P_sample.shape)).float()
                P_sample = P_sample * jitter

                active = torch.zeros(N)
                active[root_cause_idx] = signal_strength
                amplification = 1.5 + rng.rand() * 1.0

                for hop in range(min(max_cascade_hops, window_size - t_start - 1)):
                    t = t_start + 1 + hop
                    if t >= window_size:
                        break
                    next_active = torch.matmul(P_sample.T, active) * amplification

                    # Stochastic drop (~20 % of paths fail)
                    drop = torch.from_numpy(rng.binomial(1, 0.8, size=N)).float()
                    next_active = next_active * drop

                    active_idx = torch.nonzero(next_active > 0.1, as_tuple=False).squeeze(-1)
                    if active_idx.numel() > 0:
                        # Track cascade nodes for multimodal generation
                        cascade_nodes.extend(active_idx.tolist() if active_idx.dim() > 0 else [active_idx.item()])

                        base = next_active[active_idx]
                        for feat in affected_feats:
                            X[t, active_idx, feat] += base + torch.randn(active_idx.numel()) * 0.2

                        # Feature bleed
                        other = [f for f in range(num_features) if f not in affected_feats]
                        if other:
                            for feat in other:
                                X[t, active_idx, feat] += base * 0.15

                    active = next_active

                # Deduplicate cascade_nodes
                cascade_nodes = list(set(cascade_nodes))

            # Background correlation (reduced)
            for t in range(t_start, window_size):
                X[t] += torch.randn(N, num_features) * 0.05

        # Shape: [N, F, W]
        window_x = X.permute(1, 2, 0)

        # Gray failures: degrade inter-metric correlation progressively
        if is_anomaly and failure_type == "gray":
            for t in range(t_start, window_size):
                frac = (t - t_start) / max(window_size - t_start, 1)
                noise_scale = 0.1 + frac * 0.4
                window_x[root_cause_idx, :, t] += (
                    torch.randn(num_features) * noise_scale
                )

        data = Data(x=window_x, edge_index=edge_index)
        data.failure_type = failure_type
        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [node_names[root_cause_idx]]
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]

        # Generate multimodal data (logs and traces)
        if generate_multimodal:
            x_logs = generate_log_features(
                rng, N, is_anomaly, root_cause_idx, failure_type,
                cascade_nodes, signal_strength,
            )
            x_traces = generate_trace_features(
                rng, N, G, is_anomaly, root_cause_idx, failure_type,
                cascade_nodes, signal_strength,
            )
            data.x_logs = torch.from_numpy(x_logs)
            data.x_traces = torch.from_numpy(x_traces)

        dataset.append(data)

    # Distribution stats
    anomalies = [d for d in dataset if d.y.item() == 1]
    healthy = [d for d in dataset if d.y.item() == 0]
    from collections import Counter
    rc_dist = Counter(d.root_cause[0] for d in anomalies)
    mm_str = " (multimodal: logs+traces)" if generate_multimodal else ""
    logger.info(
        "Dataset: %d total (%d anomalies, %d healthy), %d root-cause classes%s",
        len(dataset), len(anomalies), len(healthy), len(rc_dist), mm_str,
    )

    return dataset, node_names, edge_index


# ---------------------------------------------------------------------------
# Large-scale generation utilities (Phase 5: N=500, N=1000)
# ---------------------------------------------------------------------------

def get_peak_memory_mb() -> dict:
    """Get current peak CPU and GPU memory usage in MB."""
    import tracemalloc
    stats = {}
    if tracemalloc.is_tracing():
        _, peak = tracemalloc.get_traced_memory()
        stats["cpu_peak_mb"] = peak / 1024 / 1024
    else:
        stats["cpu_peak_mb"] = None
    if torch.cuda.is_available():
        stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        stats["gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
    else:
        stats["gpu_peak_mb"] = None
        stats["gpu_reserved_mb"] = None
    return stats


def generate_large_scale_datasets(
    sizes=(500, 1000),
    windows_per_class: int = 30,
    window_size: int = 12,
    num_features: int = 5,
    anomaly_ratio: float = 0.7,
    ba_m: int = 3,
    seed: int = 42,
    output_dir: str = ".",
) -> dict:
    """
    Generate datasets for N=500, N=1000 with memory profiling (Phase 5).

    Class-balanced: num_samples = N * windows_per_class / anomaly_ratio
    ensures each node appears as RC approximately windows_per_class times.
    """
    import os
    import time
    import tracemalloc

    os.makedirs(output_dir, exist_ok=True)
    results_meta = {}

    for N in sizes:
        n_anomalous = N * windows_per_class
        num_samples = int(n_anomalous / anomaly_ratio)
        logger.info("Generating N=%d dataset: %d total samples", N, num_samples)

        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        dataset, node_names, edge_index = generate_scalable_dataset(
            num_nodes=N, ba_m=ba_m, num_samples=num_samples,
            window_size=window_size, num_features=num_features,
            anomaly_ratio=anomaly_ratio, noise_level=0.1, seed=seed,
        )
        elapsed = time.time() - t0
        memory = get_peak_memory_mb()
        tracemalloc.stop()

        output_path = os.path.join(output_dir, f"scalable_dataset_N{N}.pt")
        torch.save({"dataset": dataset, "node_names": node_names}, output_path)

        from collections import Counter
        anomalies = [d for d in dataset if d.y.item() == 1]
        rc_dist = Counter(d.root_cause[0] for d in anomalies)

        cpu_str = f"{memory['cpu_peak_mb']:.1f} MB" if memory.get("cpu_peak_mb") else "N/A"
        gpu_str = f"{memory['gpu_peak_mb']:.1f} MB" if memory.get("gpu_peak_mb") else "N/A (CPU)"

        print(f"\n{'='*60}")
        print(f"N={N} LARGE-SCALE DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"  Samples:       {len(dataset)}")
        print(f"  Anomalies:     {len(anomalies)}")
        print(f"  RC classes:    {len(rc_dist)}")
        print(f"  Window shape:  {dataset[0].x.shape}")
        print(f"  Generation:    {elapsed:.1f}s")
        print(f"  CPU peak RAM:  {cpu_str}")
        print(f"  GPU peak VRAM: {gpu_str}")
        print(f"  Saved to:      {output_path}")
        print("  Top-5 RC classes:")
        for name, count in rc_dist.most_common(5):
            print(f"    {name}: {count} windows")

        results_meta[N] = {
            "path": output_path,
            "n_samples": len(dataset),
            "n_rc_classes": len(rc_dist),
            "memory": memory,
            "elapsed_s": elapsed,
        }

    return results_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scalable simulated RCA dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="N (any positive int)")
    parser.add_argument("--ba-m", type=int, default=3, help="BA attachment param")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Total windows (ignored in large-scale mode)")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--anomaly-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output .pt path")
    parser.add_argument(
        "--mode", type=str, default="standard", choices=["standard", "large-scale"],
        help="standard: single dataset. large-scale: N=500 and N=1000 with memory profiling.",
    )
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory (large-scale mode)")
    args = parser.parse_args()

    if args.mode == "large-scale":
        print("=== Phase 5: Large-Scale Dataset Generation (N=500, N=1000) ===")
        meta = generate_large_scale_datasets(
            sizes=(500, 1000),
            windows_per_class=30,
            window_size=args.window_size,
            ba_m=args.ba_m,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print("\n=== Memory Summary ===")
        for N, info in meta.items():
            cpu_mb = info["memory"].get("cpu_peak_mb")
            gpu_mb = info["memory"].get("gpu_peak_mb")
            cpu_str = f"{cpu_mb:.0f} MB" if cpu_mb else "N/A"
            gpu_str = f"{gpu_mb:.0f} MB" if gpu_mb else "N/A"
            print(
                f"  N={N:5d}: CPU={cpu_str:>10s}  GPU={gpu_str:>10s}  "
                f"time={info['elapsed_s']:.1f}s"
            )
    else:
        if args.output is None:
            args.output = f"scalable_dataset_N{args.num_nodes}.pt"

        dataset, node_names, edge_index = generate_scalable_dataset(
            num_nodes=args.num_nodes,
            ba_m=args.ba_m,
            num_samples=args.num_samples,
            window_size=args.window_size,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.seed,
        )

        torch.save({"dataset": dataset, "node_names": node_names}, args.output)

        print(f"Saved {len(dataset)} samples to {args.output}")
        print(f"Topology: N={args.num_nodes}, E={edge_index.size(1)//2} directed edges")
        print(f"Window shape: {dataset[0].x.shape}")

        from collections import Counter
        anomalies = [d for d in dataset if d.y.item() == 1]
        rc_dist = Counter(d.root_cause[0] for d in anomalies)
        print(f"Root cause classes: {len(rc_dist)}")
        top5 = rc_dist.most_common(5)
        for name, count in top5:
            print(f"  {name}: {count}")

