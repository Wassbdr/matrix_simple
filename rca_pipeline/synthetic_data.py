import torch
from torch_geometric.data import Data
from typing import List, Dict
import numpy as np

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}

def generate_synthetic_dataset(
    num_samples: int = 1500,
    window_size: int = 6,
    num_features: int = 5,
    anomaly_ratio: float = 0.3,
    noise_level: float = 0.1,
    propagation_strength: float = 0.6
):
    """
    Generates a synthetic dataset rigorously following the mathematical formalization:
       X(t+1) = f(A, X(t)) + B*U(t) + noise
       
    The anomaly U(t) is a sparse exogenous injection. Over time, it propagates 
    via the adjacency matrix A to other nodes, rendering shallow models (like RF) 
    unable to find the root cause, as they only look at the final snapshot X(t).
    """
    
    # K8S Topology Adjacency Matrix
    # Frontend(0) -> Auth(1), Frontend(0) -> Backend(2)
    # Backend(2) -> Cache(3), Backend(2) -> Database(4)
    A = torch.zeros((5, 5))
    edges = [(0, 1), (0, 2), (2, 3), (2, 4)]
    
    # We add self-loops so history is preserved
    for i in range(5):
        A[i, i] = 1.0
        
    for src, dst in edges:
        A[src, dst] = propagation_strength
        A[dst, src] = propagation_strength * 0.5 # Back-pressure (e.g., db slows down backend)

    A_raw = A.clone()  # Store before normalization for distractor propagation

    # Normalize A (Random walk matrix)
    row_sum = A.sum(dim=1, keepdim=True)
    A = A / row_sum
    
    edge_index = torch.tensor([
        [0, 0, 2, 2, 1, 2, 3, 4],
        [1, 2, 3, 4, 0, 0, 2, 2]
    ], dtype=torch.long)
    
    # Propagation matrix P: anomaly travels UP the dependency chain
    P = torch.zeros((5, 5))
    anomaly_edges = [(1, 0), (2, 0), (3, 2), (4, 2)]
    for src, dst in anomaly_edges:
        P[src, dst] = 1.0
        
    dataset = []
    
    for _ in range(num_samples):
        is_anomaly = np.random.rand() < anomaly_ratio
        
        X = torch.zeros(window_size, 5, num_features)
        
        # Base healthy state noise
        X += torch.randn(window_size, 5, num_features) * noise_level
        
        # Distractor spikes that propagate 1 hop (mini-cascades)
        # These confuse flat learners since they resemble real cascades
        num_distractors = np.random.randint(2, 5)
        for _ in range(num_distractors):
            rand_t = np.random.randint(0, window_size - 1)
            rand_node = np.random.randint(0, 5)
            n_spike_feats = np.random.randint(1, num_features)
            spike_feats = sorted(np.random.choice(num_features, size=n_spike_feats, replace=False))
            spike_strength = 1.5 + np.random.rand() * 3.5
            X[rand_t, rand_node, spike_feats] += spike_strength
            # Propagate 1 step to neighbors via adjacency
            for dst in range(5):
                if A_raw[rand_node, dst].item() > 0 and dst != rand_node and rand_t + 1 < window_size:
                    X[rand_t + 1, dst, spike_feats] += spike_strength * 0.4

        root_cause_idx = -1
        if is_anomaly:
            root_cause_idx = np.random.randint(0, 5)

            # Feature subset: only some features react to this anomaly type
            n_affected = np.random.randint(2, max(3, num_features))
            affected_feats = sorted(np.random.choice(num_features, size=n_affected, replace=False))

            # Root cause: TRANSIENT signal (fades rapidly - exponential decay)
            t_start = np.random.randint(0, max(1, window_size - 3))
            signal_strength = 0.8 + np.random.rand() * 2.2  # Variable: 0.8-3.0
            X[t_start, root_cause_idx, affected_feats] += signal_strength + torch.randn(len(affected_feats)) * 0.3
            # Root cause signal decays exponentially (transient fault)
            decay = 0.3 + np.random.rand() * 0.2  # decay factor 0.3-0.5
            for dt in range(1, window_size - t_start):
                remaining = signal_strength * (decay ** dt)
                if remaining > 0.05:
                    X[t_start + dt, root_cause_idx, affected_feats] += remaining + torch.randn(len(affected_feats)) * 0.1

            # Cascade: propagates with STRONG amplification
            # Downstream nodes become dramatically louder than root cause
            active_alerts = torch.zeros(5)
            active_alerts[root_cause_idx] = signal_strength
            amplification = 1.8 + np.random.rand() * 0.7  # 1.8-2.5x per hop

            for t in range(t_start + 1, window_size):
                next_alerts = torch.matmul(P.T, active_alerts)
                next_alerts = next_alerts * amplification

                for node_id in range(5):
                    if next_alerts[node_id] > 0.1:
                        X[t, node_id, affected_feats] += next_alerts[node_id] + torch.randn(len(affected_feats)) * 0.3
                        # Bleed into other features (cascades are messy)
                        other_feats = [f for f in range(num_features) if f not in affected_feats]
                        if other_feats:
                            X[t, node_id, other_feats] += next_alerts[node_id] * 0.2

                active_alerts = next_alerts

            # Background correlation: all nodes get correlated noise during anomaly
            for t in range(t_start, window_size):
                X[t, :, :] += torch.randn(5, num_features) * 0.15
                
        # Permute to match expected format: [nodes, features, time]
        window_x = X.permute(1, 2, 0)
        
        data = Data(x=window_x, edge_index=edge_index)
        
        if is_anomaly:
            data.y = torch.tensor([1], dtype=torch.long)
            data.root_cause = [NODE_NAMES[root_cause_idx]]
        else:
            data.y = torch.tensor([0], dtype=torch.long)
            data.root_cause = [""]
            
        dataset.append(data)
        
    return dataset

if __name__ == "__main__":
    print("Generating strictly formalized synthetic RCA dataset (N=5000)...")
    dataset = generate_synthetic_dataset(num_samples=5000, window_size=12, num_features=5, anomaly_ratio=0.4, noise_level=0.1, propagation_strength=0.8)
    
    import torch
    torch.save(dataset, "synthetic_dataset_windows.pt")
    
    anomalies = [d for d in dataset if d.y.item() == 1]
    print(f"Dataset generated: 5000 total samples, {len(anomalies)} anomalies")
    
    from collections import Counter
    rc_dist = Counter([d.root_cause[0] for d in anomalies])
    print(f"Root cause distribution: {dict(rc_dist)}")
