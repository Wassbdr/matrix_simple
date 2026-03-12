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

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
) -> Tuple[List[Data], List[str], torch.Tensor]:
    """
    Generate a scalable simulated RCA dataset on a BA topology.

    Args:
        num_nodes:        N in {50, 100, 150, 200}
        ba_m:             BA attachment parameter (2-4)
        num_samples:      total windows
        window_size:      temporal window W
        num_features:     per-node features F
        anomaly_ratio:    fraction of anomalous windows
        noise_level:      background noise sigma
        max_cascade_hops: max propagation depth
        seed:             reproducibility

    Returns:
        dataset:     List[Data]  (each has .x [N,F,W], .edge_index, .y, .root_cause)
        node_names:  List[str]
        edge_index:  [2, E]
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

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
            feats = sorted(rng.choice(num_features, size=n_feats, replace=False))
            strength = 0.5 + rng.rand() * 1.5
            X[t, node, feats] += strength
            # 1-hop propagation (weak)
            for dst in range(N):
                if A_raw[node, dst].item() > 0 and dst != node and t + 1 < window_size:
                    X[t + 1, dst, feats] += strength * 0.15

        root_cause_idx = -1

        if is_anomaly:
            root_cause_idx = rng.randint(0, N)

            # Feature subset affected
            n_affected = rng.randint(2, max(3, num_features))
            affected_feats = sorted(rng.choice(num_features, size=n_affected, replace=False))

            # Root cause: transient exponential decay (boosted for clear SNR)
            t_start = rng.randint(0, max(1, window_size - 4))
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

                for nid in range(N):
                    if next_active[nid] > 0.1:
                        X[t, nid, affected_feats] += (
                            next_active[nid] + torch.randn(len(affected_feats)) * 0.2
                        )
                        # Feature bleed
                        other = [f for f in range(num_features) if f not in affected_feats]
                        if other:
                            X[t, nid, other] += next_active[nid] * 0.15

                active = next_active

            # Background correlation (reduced)
            for t in range(t_start, window_size):
                X[t] += torch.randn(N, num_features) * 0.05

        # Shape: [N, F, W]
        window_x = X.permute(1, 2, 0)

        data = Data(x=window_x, edge_index=edge_index)
        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [node_names[root_cause_idx]]
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]

        dataset.append(data)

    # Distribution stats
    anomalies = [d for d in dataset if d.y.item() == 1]
    healthy = [d for d in dataset if d.y.item() == 0]
    from collections import Counter
    rc_dist = Counter(d.root_cause[0] for d in anomalies)
    logger.info(
        "Dataset: %d total (%d anomalies, %d healthy), %d root-cause classes",
        len(dataset), len(anomalies), len(healthy), len(rc_dist),
    )

    return dataset, node_names, edge_index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scalable simulated RCA dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="N (50-200)")
    parser.add_argument("--ba-m", type=int, default=3, help="BA attachment param")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--anomaly-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output .pt path")
    args = parser.parse_args()

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
