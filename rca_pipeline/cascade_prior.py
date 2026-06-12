"""Propagation prior for cascade failures (inference-time, model-agnostic).

Documented failure mode (docs/SYNTHESE_FINALE.md §6.4): on multi-hop cascades
the root cause has almost no local metric anomaly — symptoms explode on its
neighbors (e.g. a delay injected on the orders → orders-db link makes
``orders``/``front-end`` scream while ``orders-db``, the actual root cause,
stays quiet). Local-evidence models (STA's selector in particular) then rank
the loud symptom nodes first; STA collapses to Avg@1 0.04 on that scenario.

The prior implemented here scores exactly the inverse pattern: *quiet nodes
with loud neighbors*. It is blended with model probabilities at evaluation
time (no retraining):

    p_final = (1 - gamma) * softmax(model_logits) + gamma * softmax(prior / tau)

Validation discipline: any gamma > 0 must be checked against the full
per-scenario breakdown — the boost must not degrade hard failures where the
root cause is itself the loudest node.
"""

from __future__ import annotations

import numpy as np
import torch


def node_anomaly_scores(x: torch.Tensor) -> np.ndarray:
    """Per-node anomaly intensity from a (normalized) window ``x`` [N, F, W].

    Mean absolute deviation over features and time, scaled to [0, 1] by the
    max across nodes. Inputs are expected z-scored (the benchmark normalizes
    with train-only stats); on raw inputs the relative ordering still holds.
    """
    a = x.detach().abs().mean(dim=(1, 2)).cpu().numpy().astype(np.float64)
    peak = a.max()
    return a / peak if peak > 0 else a


def undirected_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list:
    """Adjacency list using the union of in- and out-edges (cascades travel
    both with and against call direction)."""
    neigh = [set() for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        neigh[u].add(v)
        neigh[v].add(u)
    return [sorted(s) for s in neigh]


def propagation_scores(
    anomaly: np.ndarray,
    edge_index: torch.Tensor,
) -> np.ndarray:
    """Score quiet nodes whose neighborhood is anomalous.

    prior(v) = mean(anomaly of neighbors of v) * (1 - anomaly(v))

    Isolated nodes get 0. Output is in [0, 1].
    """
    n = len(anomaly)
    neigh = undirected_neighbors(edge_index, n)
    out = np.zeros(n, dtype=np.float64)
    for v in range(n):
        if not neigh[v]:
            continue
        neigh_a = float(np.mean([anomaly[u] for u in neigh[v]]))
        out[v] = neigh_a * (1.0 - float(anomaly[v]))
    return out


def apply_cascade_prior(
    logits: np.ndarray,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    gamma: float,
    tau: float = 0.25,
) -> np.ndarray:
    """Blend model logits with the propagation prior; returns probabilities.

    ``gamma=0`` returns plain softmax(logits). ``tau`` sharpens the prior
    distribution before mixing (priors live in [0, 1], much flatter than
    logits).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p_model = np.exp(z) / np.exp(z).sum()
    if gamma == 0.0:
        return p_model
    prior = propagation_scores(node_anomaly_scores(x), edge_index)
    q = prior / tau
    q = q - q.max()
    p_prior = np.exp(q) / np.exp(q).sum()
    return (1.0 - gamma) * p_model + gamma * p_prior
