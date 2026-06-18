"""End-to-end-ish cascade prior check on the real sock_shop N=14 topology.

Reproduces the documented failure (SYNTHESE_FINALE §6.4) — root cause
``orders-db`` metric-quiet, downstream neighbors ``orders``/``front-end``
loud — and verifies the prior blend recovers it to top-1. No training: the
prior is post-hoc, so this isolates the mechanism from model noise and runs
in CI without any dataset.
"""

import numpy as np
import torch

from rca_pipeline.cascade_prior import (
    apply_cascade_prior,
    node_anomaly_scores,
    propagation_scores,
)
from rca_pipeline.topologies import get_topology

F, W = 6, 12


def _cascade_window(idx, n):
    x = torch.zeros(n, F, W)
    x[idx["orders"]] = 4.0       # loud downstream symptom
    x[idx["front-end"]] = 3.0    # propagated upstream
    x[idx["orders-db"]] = 0.10   # the actual root cause barely moves
    torch.manual_seed(0)
    return x + 0.05 * torch.randn(n, F, W)


def test_prior_scores_quiet_root_cause_highest_on_sockshop():
    topo = get_topology("sock_shop")
    idx = topo.node_mapping
    ei = topo.edge_index()
    n = len(idx)
    x = _cascade_window(idx, n)

    prior = propagation_scores(node_anomaly_scores(x), ei)
    # orders-db is quiet with loud neighbors -> top of the prior.
    assert int(np.argmax(prior)) == idx["orders-db"]
    # The loud symptom node itself gets ~0 prior.
    assert prior[idx["orders"]] < prior[idx["orders-db"]]


def test_blend_recovers_root_cause_to_top1():
    topo = get_topology("sock_shop")
    idx = topo.node_mapping
    names = topo.node_names()
    ei = topo.edge_index()
    n = len(idx)
    x = _cascade_window(idx, n)

    # Local-evidence model ranks the loudest node first (STA's failure mode).
    model_logits = node_anomaly_scores(x) * 8.0
    rc = idx["orders-db"]

    def rc_rank(gamma):
        p = apply_cascade_prior(model_logits, x, ei, gamma=gamma, tau=0.1)
        order = np.argsort(-p)
        return int(np.where(order == rc)[0][0]) + 1

    assert rc_rank(0.0) > 1          # model alone misses it
    assert rc_rank(0.7) == 1         # blend recovers it
    # Monotone improvement as the prior gains weight.
    assert rc_rank(0.7) <= rc_rank(0.3) <= rc_rank(0.0)
    # And the chosen node is the documented root cause.
    p = apply_cascade_prior(model_logits, x, ei, gamma=0.7, tau=0.1)
    assert names[int(np.argmax(p))] == "orders-db"
