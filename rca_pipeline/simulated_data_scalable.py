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

Anomaly dynamics (SOTA version with recurrence):
    V2 adds:
    - Power-law root cause distribution (top 10% = 80% of incidents)
    - 80% recurrence ratio (patterns repeat)
    - 5 varied temporal patterns (not just exponential decay)
    - Metadata per sample: is_recurrent, pattern_type

Usage:
    python simulated_data_scalable.py --num-nodes 100 --num-samples 5000
    python simulated_data_scalable.py --num-nodes 100 --num-samples 5000 --v2  # SOTA version
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_severity(signal_strength: float, cascade_hops: int,
                     n_affected: int, N: int) -> float:
    """Score continu [0, 1] reflétant l'impact de l'anomalie.
    
    Combine 3 facteurs normalisés :
    - impact_ratio : fraction des nœuds affectés par la cascade
    - depth_factor : profondeur de propagation normalisée
    - strength_factor : amplitude du signal initial
    """
    impact_ratio = min(n_affected / max(N, 1), 1.0)
    depth_factor = min(cascade_hops / 5.0, 1.0)
    strength_factor = min(signal_strength / 5.0, 1.0)
    return 0.4 * impact_ratio + 0.3 * depth_factor + 0.3 * strength_factor


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
    noise_level: float = 0.3,
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
    neighbor_mask = A_raw > 0
    neighbor_mask.fill_diagonal_(False)

    dataset: List[Data] = []

    for sample_idx in range(num_samples):
        is_anomaly = rng.rand() < anomaly_ratio

        X = torch.randn(window_size, N, num_features) * noise_level

        # ---- Correlated Distractor spikes (increased strength & count to reduce SNR) ----
        num_distractors = rng.randint(max(3, N // 10), max(5, N // 5))
        for _ in range(num_distractors):
            t = rng.randint(0, window_size - 1)
            node = rng.randint(0, N)
            n_feats = rng.randint(1, max(2, num_features))
            feats = sorted(rng.choice(num_features, size=n_feats, replace=False))
            strength = 1.0 + rng.rand() * 2.0
            X[t, node, feats] += strength
            # 1-hop propagation (stronger correlation)
            if t + 1 < window_size:
                dst_mask = neighbor_mask[node]
                dst_idx = torch.nonzero(dst_mask, as_tuple=False).squeeze(-1)
                if dst_idx.numel() > 0:
                    feat_idx = torch.tensor(feats, dtype=torch.long)
                    X[t + 1][dst_idx.unsqueeze(1), feat_idx.unsqueeze(0)] += strength * 0.3

        root_cause_idx = -1

        if is_anomaly:
            root_cause_idx = rng.randint(0, N)

            # Feature subset affected
            n_affected = rng.randint(2, max(3, num_features))
            affected_feats = sorted(rng.choice(num_features, size=n_affected, replace=False))

            # Root cause: transient exponential decay (reduced for harder SNR)
            t_start = rng.randint(0, max(1, window_size - 4))
            signal_strength = 1.5 + rng.rand() * 2.0
            decay = 0.3 + rng.rand() * 0.3

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

                active_idx = torch.nonzero(next_active > 0.1, as_tuple=False).squeeze(-1)
                if active_idx.numel() > 0:
                    base = next_active[active_idx]
                    for feat in affected_feats:
                        X[t, active_idx, feat] += base + torch.randn(active_idx.numel()) * 0.2

                    # Feature bleed
                    other = [f for f in range(num_features) if f not in affected_feats]
                    if other:
                        for feat in other:
                            X[t, active_idx, feat] += base * 0.15

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
            # --- Severity metadata ---
            n_cascaded = int((active > 0.1).sum().item())
            cascade_depth = min(hop + 1, max_cascade_hops) if is_anomaly else 0
            data.severity = compute_severity(
                signal_strength, cascade_depth, n_cascaded, N
            )
            data.n_affected_nodes = n_cascaded
            data.cascade_depth = cascade_depth
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]
            data.severity = 0.0
            data.n_affected_nodes = 0
            data.cascade_depth = 0

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

    # Per-feature normalization (zero-mean, unit-variance)
    # Computed globally since train/test split happens downstream in benchmark.py
    all_x = torch.stack([d.x for d in dataset])          # [S, N, F, W]
    mean = all_x.mean(dim=(0, 1, 3), keepdim=True)       # [1, 1, F, 1]
    std = all_x.std(dim=(0, 1, 3), keepdim=True).clamp(min=1e-6)
    for d in dataset:
        d.x = (d.x - mean.squeeze(0)) / std.squeeze(0)
    logger.info("Applied per-feature normalization (mean=%.4f, std=%.4f)",
                mean.mean().item(), std.mean().item())

    return dataset, node_names, edge_index


# ---------------------------------------------------------------------------
# V2: SOTA Dataset Generation with Power-Law and Recurrence
# ---------------------------------------------------------------------------

def generate_scalable_dataset_v2(
    num_nodes: int = 100,
    ba_m: int = 3,
    num_samples: int = 5000,
    window_size: int = 12,
    num_features: int = 5,
    anomaly_ratio: float = 0.4,
    recurrence_ratio: float = 0.8,
    power_law_alpha: float = 2.0,
    noise_level: float = 0.1,
    max_cascade_hops: int = 6,
    use_realistic_pattern_weights: bool = True,
    seed: int = 42,
) -> Tuple[List[Data], List[str], torch.Tensor, List[Dict]]:
    """
    Generate a SOTA scalable simulated RCA dataset with:
    - Power-law root cause distribution (Pareto: top 10% = 80% of incidents)
    - 80% recurrence ratio (patterns repeat)
    - 8 varied temporal patterns including gray failures (39% of incidents!)

    Args:
        num_nodes:        N in {50, 100, 150, 200}
        ba_m:             BA attachment parameter (2-4)
        num_samples:      total windows
        window_size:      temporal window W
        num_features:     per-node features F
        anomaly_ratio:    fraction of anomalous windows
        recurrence_ratio: fraction of anomalies that are recurrences (default 0.8)
        power_law_alpha:  Zipf exponent for RC distribution (default 2.0)
        noise_level:      background noise sigma
        max_cascade_hops: max propagation depth
        use_realistic_pattern_weights: Use SOTA pattern weights (39% gray failures)
        seed:             reproducibility

    Returns:
        dataset:     List[Data] with .x, .edge_index, .y, .root_cause, .is_recurrent, .pattern_type
        node_names:  List[str]
        edge_index:  [2, E]
        metadata:    List[Dict] with is_recurrent, pattern_type, root_cause_idx
    """
    # Import SOTA modules
    from anomaly_distribution import PowerLawRootCauseDistribution
    from temporal_patterns import PATTERN_NAMES, PATTERN_WEIGHTS, PATTERN_DIFFICULTY, inject_anomaly_pattern

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    node_names, edge_index, P_base, G = generate_ba_topology(num_nodes, ba_m, seed)
    N = num_nodes

    # Initialize power-law distribution
    rc_dist = PowerLawRootCauseDistribution(
        num_nodes=N,
        alpha=power_law_alpha,
        top_k_weight=0.8,
        seed=seed,
    )

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
    metadata: List[Dict] = []

    # Prepare pattern sampling weights
    if use_realistic_pattern_weights:
        pattern_probs = np.array([PATTERN_WEIGHTS.get(p, 0.1) for p in PATTERN_NAMES])
        pattern_probs = pattern_probs / pattern_probs.sum()
        logger.info("Using realistic pattern weights: %s",
                   {p: f"{PATTERN_WEIGHTS.get(p, 0):.0%}" for p in PATTERN_NAMES})
    else:
        pattern_probs = np.ones(len(PATTERN_NAMES)) / len(PATTERN_NAMES)

    # Track pattern distribution for logging
    pattern_counts = {p: 0 for p in PATTERN_NAMES}
    recurrence_count = 0
    difficulty_counts = {"hard": 0, "medium": 0, "normal": 0, "easy": 0}

    for sample_idx in range(num_samples):
        is_anomaly = rng.rand() < anomaly_ratio

        X = torch.randn(window_size, N, num_features) * noise_level

        # ---- Distractor spikes (reduced strength for clear SNR) ----
        num_distractors = rng.randint(2, max(3, N // 15))
        for _ in range(num_distractors):
            t = rng.randint(0, window_size - 1)
            node = rng.randint(0, N)
            n_feats = rng.randint(1, max(2, num_features))
            feats = sorted(rng.choice(num_features, size=n_feats, replace=False))
            strength = 0.5 + rng.rand() * 1.5
            X[t, node, feats] += strength
            # 1-hop propagation (weak)
            if t + 1 < window_size:
                dst_mask = neighbor_mask[node]
                dst_idx = torch.nonzero(dst_mask, as_tuple=False).squeeze(-1)
                if dst_idx.numel() > 0:
                    feat_idx = torch.tensor(feats, dtype=torch.long)
                    X[t + 1][dst_idx.unsqueeze(1), feat_idx.unsqueeze(0)] += strength * 0.15

        root_cause_idx = -1
        is_recurrent = False
        pattern_type = "healthy"
        pattern_difficulty = "none"

        sample_meta = {
            "sample_idx": sample_idx,
            "is_anomaly": is_anomaly,
            "is_recurrent": False,
            "pattern_type": "healthy",
            "pattern_difficulty": "none",
            "root_cause_idx": -1,
        }

        if is_anomaly:
            # Sample root cause using power-law distribution with recurrence
            root_cause_idx, is_recurrent = rc_dist.sample_root_cause(
                recurrence_prob=recurrence_ratio
            )

            # Sample pattern type
            if is_recurrent:
                # Try to reuse pattern from history
                pattern_type, pattern_reused = rc_dist.sample_pattern_for_rc(
                    root_cause=root_cause_idx,
                    is_recurrent=is_recurrent,
                    available_patterns=PATTERN_NAMES,
                )
                if not pattern_reused:
                    # Fall back to weighted sampling
                    pattern_type = rng.choice(PATTERN_NAMES, p=pattern_probs)
            else:
                # Novel incident: weighted sampling
                pattern_type = rng.choice(PATTERN_NAMES, p=pattern_probs)

            pattern_difficulty = PATTERN_DIFFICULTY.get(pattern_type, "normal")

            # Feature subset affected
            n_affected = rng.randint(2, max(3, num_features))
            affected_feats = sorted(rng.choice(num_features, size=n_affected, replace=False))

            # Randomize pattern parameters (reduce strength for hard patterns)
            if pattern_difficulty == "hard":
                signal_strength = 1.5 + rng.rand() * 2.0  # Lower for gray failures
            else:
                signal_strength = 2.5 + rng.rand() * 3.0

            # Inject anomaly with temporal pattern and cascade propagation
            X = inject_anomaly_pattern(
                X=X,
                root_cause_idx=root_cause_idx,
                pattern_type=pattern_type,
                P_base=P_base,
                affected_feats=affected_feats,
                rng=rng,
                max_cascade_hops=max_cascade_hops,
                cascade_amplification=1.5 + rng.rand() * 0.5,
                cascade_decay=0.4 + rng.rand() * 0.2,
                strength=signal_strength,
            )

            # Background correlation (reduced)
            t_start = rng.randint(0, max(1, window_size - 4))
            for t in range(t_start, window_size):
                X[t] += torch.randn(N, num_features) * 0.05

            # Register incident for future recurrence
            rc_dist.register_incident(
                root_cause=root_cause_idx,
                pattern_type=pattern_type,
                affected_features=affected_feats,
                signal_strength=signal_strength,
                sample_idx=sample_idx,
            )

            sample_meta.update({
                "is_recurrent": is_recurrent,
                "pattern_type": pattern_type,
                "pattern_difficulty": pattern_difficulty,
                "root_cause_idx": root_cause_idx,
            })

            # Update stats
            pattern_counts[pattern_type] += 1
            difficulty_counts[pattern_difficulty] += 1
            if is_recurrent:
                recurrence_count += 1

        # Shape: [N, F, W]
        window_x = X.permute(1, 2, 0)

        data = Data(x=window_x, edge_index=edge_index)
        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [node_names[root_cause_idx]]
            data.is_recurrent = is_recurrent
            data.pattern_type = pattern_type
            data.pattern_difficulty = pattern_difficulty
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]
            data.is_recurrent = False
            data.pattern_type = "healthy"
            data.pattern_difficulty = "none"

        dataset.append(data)
        metadata.append(sample_meta)

    # Distribution stats
    anomalies = [d for d in dataset if d.y.item() == 1]
    healthy = [d for d in dataset if d.y.item() == 0]

    logger.info(
        "Dataset V2: %d total (%d anomalies, %d healthy)",
        len(dataset), len(anomalies), len(healthy),
    )

    if anomalies:
        rc_dist_stats = rc_dist.get_statistics()
        actual_recurrence_rate = recurrence_count / len(anomalies)
        logger.info(
            "Recurrence: %.1f%% (target: %.1f%%)",
            actual_recurrence_rate * 100, recurrence_ratio * 100,
        )
        logger.info(
            "Top 10%% RC weight (theoretical): %.1f%%",
            rc_dist_stats.get("top_10pct_theoretical_weight", 0) * 100,
        )
        logger.info("Pattern distribution: %s", pattern_counts)
        logger.info("Difficulty distribution: %s", difficulty_counts)

    return dataset, node_names, edge_index, metadata


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
    # V2 (SOTA) options
    parser.add_argument("--v2", action="store_true", help="Use SOTA V2 with power-law and recurrence")
    parser.add_argument("--recurrence-ratio", type=float, default=0.8, help="Fraction of recurrent anomalies (V2 only)")
    parser.add_argument("--power-law-alpha", type=float, default=2.0, help="Zipf exponent (V2 only)")
    args = parser.parse_args()

    if args.output is None:
        suffix = "_v2" if args.v2 else ""
        args.output = f"scalable_dataset_N{args.num_nodes}{suffix}.pt"

    if args.v2:
        # SOTA version with power-law and recurrence
        dataset, node_names, edge_index, metadata = generate_scalable_dataset_v2(
            num_nodes=args.num_nodes,
            ba_m=args.ba_m,
            num_samples=args.num_samples,
            window_size=args.window_size,
            anomaly_ratio=args.anomaly_ratio,
            recurrence_ratio=args.recurrence_ratio,
            power_law_alpha=args.power_law_alpha,
            seed=args.seed,
        )

        torch.save({
            "dataset": dataset,
            "node_names": node_names,
            "metadata": metadata,
            "version": "v2_sota",
        }, args.output)

        print(f"[V2 SOTA] Saved {len(dataset)} samples to {args.output}")
        print(f"Topology: N={args.num_nodes}, E={edge_index.size(1)//2} directed edges")
        print(f"Window shape: {dataset[0].x.shape}")

        # Pattern breakdown
        from collections import Counter
        anomalies = [m for m in metadata if m["is_anomaly"]]
        pattern_dist = Counter(m["pattern_type"] for m in anomalies)
        recurrent_count = sum(1 for m in anomalies if m["is_recurrent"])
        print(f"Anomalies: {len(anomalies)}, Recurrent: {recurrent_count} ({recurrent_count/len(anomalies)*100:.1f}%)")
        print(f"Pattern distribution: {dict(pattern_dist)}")

    else:
        # Original version (backward compatible)
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
