"""
Scaled Simulated RCA Dataset — 20-node microservice architecture.

Topology: realistic 3-tier K8S architecture with 20 services.
    Tier 1 (Gateways):    gateway-web, gateway-mobile, gateway-admin
    Tier 2 (Services):    auth, user-svc, order-svc, payment-svc, notification-svc,
                          search-svc, recommendation-svc, inventory-svc
    Tier 3 (Data/Infra):  postgres-primary, postgres-replica, redis-cache, redis-session,
                          rabbitmq, elasticsearch, minio, monitoring

Key properties vs the 5-node dataset:
    - GIB must now filter 20 nodes down to 1 → clear value from sparsification
    - Stochastic propagation: cascade strength varies per edge per sample
    - Deeper cascades: up to 4 hops (gateway → service → service → data)
    - Variable affected feature subsets per anomaly type
    - Multi-path propagation: same root cause can cascade via different routes
"""

import torch
from torch_geometric.data import Data
from typing import List
import numpy as np

# ── 20-node topology ──────────────────────────────────────────────────
NODE_NAMES_LARGE = [
    # Tier 1: Gateways (3)
    'gateway-web',        # 0
    'gateway-mobile',     # 1
    'gateway-admin',      # 2
    # Tier 2: Application services (8)
    'auth',               # 3
    'user-svc',           # 4
    'order-svc',          # 5
    'payment-svc',        # 6
    'notification-svc',   # 7
    'search-svc',         # 8
    'recommendation-svc', # 9
    'inventory-svc',      # 10
    # Tier 3: Data & Infrastructure (9)
    'postgres-primary',   # 11
    'postgres-replica',   # 12
    'redis-cache',        # 13
    'redis-session',      # 14
    'rabbitmq',           # 15
    'elasticsearch',      # 16
    'minio',              # 17
    'monitoring',         # 18
    'config-svc',         # 19
]

NAME_TO_IDX_LARGE = {name: i for i, name in enumerate(NODE_NAMES_LARGE)}
NUM_NODES_LARGE = len(NODE_NAMES_LARGE)

# Directed dependency edges: src → dst (src depends on dst)
EDGES_LARGE = [
    # Gateways → Services
    (0, 3), (0, 4), (0, 5), (0, 8),       # gateway-web → auth, user, order, search
    (1, 3), (1, 4), (1, 5), (1, 9),       # gateway-mobile → auth, user, order, recommendation
    (2, 3), (2, 4), (2, 7),               # gateway-admin → auth, user, notification
    # Services → Services (inter-service calls)
    (5, 6), (5, 10),                       # order → payment, inventory
    (6, 7),                                # payment → notification
    (8, 16),                               # search → elasticsearch
    (9, 4), (9, 16),                       # recommendation → user, elasticsearch
    (10, 5),                               # inventory has back-link to order (re-stock)
    # Services → Data layer
    (3, 14),                               # auth → redis-session
    (4, 11),                               # user-svc → postgres-primary
    (5, 11), (5, 15),                      # order → postgres-primary, rabbitmq
    (6, 11), (6, 15),                      # payment → postgres-primary, rabbitmq
    (7, 15), (7, 17),                      # notification → rabbitmq, minio
    (8, 13),                               # search → redis-cache
    (9, 13),                               # recommendation → redis-cache
    (10, 11),                              # inventory → postgres-primary
    # Data layer internals
    (11, 12),                              # postgres-primary → postgres-replica
    (15, 7),                               # rabbitmq delivers to notification (async)
    # Infrastructure
    (18, 11), (18, 13), (18, 15),          # monitoring scrapes postgres, redis, rabbitmq
    (19, 3), (19, 4), (19, 5),             # config-svc serves auth, user, order
]


def _build_edge_index() -> torch.Tensor:
    """Build bidirectional edge_index from EDGES_LARGE."""
    src, dst = [], []
    for s, d in EDGES_LARGE:
        src.extend([s, d])
        dst.extend([d, s])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_propagation_matrix(num_nodes: int) -> torch.Tensor:
    """
    Anomaly propagation matrix P: if node i is the root cause,
    which nodes get affected? Anomalies propagate UPSTREAM along
    the dependency graph (from cause to dependents).
    
    P[src, dst] = 1.0 means anomaly at src propagates to dst.
    """
    P = torch.zeros(num_nodes, num_nodes)
    # Reverse direction: if A depends on B, anomaly in B propagates to A
    for src, dst in EDGES_LARGE:
        P[dst, src] = 1.0  # anomaly at dst propagates to src
    return P


def generate_large_dataset(
    num_samples: int = 5000,
    window_size: int = 12,
    num_features: int = 5,
    anomaly_ratio: float = 0.4,
    noise_level: float = 0.1,
    propagation_strength: float = 0.7,
    seed: int = 42,
) -> List[Data]:
    """
    Generates a 20-node simulated RCA dataset.
    
    Key differences from the 5-node version:
        - Stochastic propagation: edge weights jittered per sample
        - Deeper cascades (up to 4 hops through the topology)
        - More distractor noise (proportional to graph size)
        - Variable anomaly types with different affected features
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    N = NUM_NODES_LARGE
    edge_index = _build_edge_index()
    P_base = _build_propagation_matrix(N)
    
    # Adjacency for distractor propagation
    A_raw = torch.zeros(N, N)
    for s, d in EDGES_LARGE:
        A_raw[s, d] = propagation_strength
        A_raw[d, s] = propagation_strength * 0.3  # weaker back-pressure
    for i in range(N):
        A_raw[i, i] = 1.0
    
    dataset = []
    
    for _ in range(num_samples):
        is_anomaly = rng.rand() < anomaly_ratio
        
        X = torch.zeros(window_size, N, num_features)
        X += torch.randn(window_size, N, num_features) * noise_level
        
        # Distractor spikes: more of them in a larger graph
        num_distractors = rng.randint(3, 8)
        for _ in range(num_distractors):
            rand_t = rng.randint(0, window_size - 1)
            rand_node = rng.randint(0, N)
            n_spike_feats = rng.randint(1, max(2, num_features))
            spike_feats = sorted(rng.choice(num_features, size=n_spike_feats, replace=False))
            spike_strength = 1.0 + rng.rand() * 3.0
            X[rand_t, rand_node, spike_feats] += spike_strength
            # Propagate 1 step to neighbors
            for dst in range(N):
                if A_raw[rand_node, dst].item() > 0 and dst != rand_node and rand_t + 1 < window_size:
                    X[rand_t + 1, dst, spike_feats] += spike_strength * 0.3
        
        root_cause_idx = -1
        if is_anomaly:
            root_cause_idx = rng.randint(0, N)
            
            # Feature subset affected by this anomaly
            n_affected = rng.randint(2, max(3, num_features))
            affected_feats = sorted(rng.choice(num_features, size=n_affected, replace=False))
            
            # Root cause: transient signal with exponential decay
            t_start = rng.randint(0, max(1, window_size - 4))
            signal_strength = 1.0 + rng.rand() * 2.5
            X[t_start, root_cause_idx, affected_feats] += signal_strength + torch.randn(len(affected_feats)) * 0.3
            decay = 0.3 + rng.rand() * 0.2
            for dt in range(1, window_size - t_start):
                remaining = signal_strength * (decay ** dt)
                if remaining > 0.05:
                    X[t_start + dt, root_cause_idx, affected_feats] += remaining + torch.randn(len(affected_feats)) * 0.1
            
            # Stochastic cascade: edge weights jittered per sample
            P_sample = P_base * (0.5 + rng.rand() * 1.0)  # scale whole matrix
            # Per-edge jitter 
            jitter = torch.from_numpy(rng.uniform(0.7, 1.3, size=P_sample.shape)).float()
            P_sample = P_sample * jitter
            
            active_alerts = torch.zeros(N)
            active_alerts[root_cause_idx] = signal_strength
            amplification = 1.5 + rng.rand() * 1.0  # 1.5-2.5× per hop
            
            # Multi-hop propagation (up to 4 hops in this topology)
            for t in range(t_start + 1, min(t_start + 5, window_size)):
                next_alerts = torch.matmul(P_sample.T, active_alerts)
                next_alerts = next_alerts * amplification
                # Stochastic drop: some paths don't fire every time
                drop_mask = torch.from_numpy(rng.binomial(1, 0.8, size=N)).float()
                next_alerts = next_alerts * drop_mask
                
                for node_id in range(N):
                    if next_alerts[node_id] > 0.1:
                        X[t, node_id, affected_feats] += next_alerts[node_id] + torch.randn(len(affected_feats)) * 0.2
                        other_feats = [f for f in range(num_features) if f not in affected_feats]
                        if other_feats:
                            X[t, node_id, other_feats] += next_alerts[node_id] * 0.15
                
                active_alerts = next_alerts
            
            # Background correlation during anomaly
            for t in range(t_start, window_size):
                X[t, :, :] += torch.randn(N, num_features) * 0.12
        
        # [nodes, features, time]
        window_x = X.permute(1, 2, 0)
        data = Data(x=window_x, edge_index=edge_index)
        
        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [NODE_NAMES_LARGE[root_cause_idx]]
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]
        
        dataset.append(data)
    
    return dataset


if __name__ == "__main__":
    print("Generating 20-node simulated RCA dataset (N=5000)...")
    dataset = generate_large_dataset(
        num_samples=5000, window_size=12, num_features=5,
        anomaly_ratio=0.4, seed=42,
    )
    
    torch.save(dataset, "simulated_large_dataset_windows.pt")
    
    anomalies = [d for d in dataset if d.y.item() == 1]
    healthy = [d for d in dataset if d.y.item() == 0]
    print(f"Dataset: {len(dataset)} total, {len(anomalies)} anomalies, {len(healthy)} healthy")
    
    from collections import Counter
    rc_dist = Counter([d.root_cause[0] for d in anomalies])
    print(f"Root cause distribution ({len(rc_dist)} classes):")
    for name, count in sorted(rc_dist.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    
    print(f"\nGraph: {NODE_NAMES_LARGE[0]}...{NODE_NAMES_LARGE[-1]}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
    print(f"Window shape: {dataset[0].x.shape}")
