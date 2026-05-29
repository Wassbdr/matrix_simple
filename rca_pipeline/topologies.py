"""
rca_pipeline/topologies.py -- Service graph topologies for known demo apps.

Chaque topologie fournit :
  - node_mapping: {service_name: index} (ordre stable)
  - edges: liste de (src, dst) en noms de services (resolus via node_mapping)

Le GraphBuilder accepte un `topology` optionnel; sans argument il garde le
comportement historique (5 services de l'app interne).

Pour sock-shop, on encode la topologie weaveworksdemo (14 services) avec les
dépendances métier publiques : front-end -> catalogue/carts/orders/user/payment,
carts -> carts-db, catalogue -> catalogue-db, orders -> orders-db/shipping/
payment/user, user -> user-db/session-db, shipping -> rabbitmq -> queue-master.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ServiceTopology:
    name: str
    node_mapping: Dict[str, int]
    edges: List[Tuple[str, str]]

    def edge_index(self) -> torch.Tensor:
        idx = self.node_mapping
        src = [idx[s] for s, _ in self.edges]
        dst = [idx[d] for _, d in self.edges]
        return torch.tensor([src, dst], dtype=torch.long)

    def node_names(self) -> List[str]:
        return [n for n, _ in sorted(self.node_mapping.items(), key=lambda kv: kv[1])]


# --- App interne (rca-sandbox) : topologie historique ---------------------
CUSTOM_APP = ServiceTopology(
    name="custom_app",
    node_mapping={
        "frontend": 0,
        "auth": 1,
        "backend": 2,
        "cache": 3,
        "database": 4,
    },
    edges=[
        ("frontend", "auth"),
        ("frontend", "backend"),
        ("backend", "cache"),
        ("backend", "database"),
    ],
)


# --- Sock-shop (weaveworksdemo) ------------------------------------------
SOCK_SHOP = ServiceTopology(
    name="sock_shop",
    node_mapping={
        "front-end": 0,
        "catalogue": 1,
        "catalogue-db": 2,
        "carts": 3,
        "carts-db": 4,
        "orders": 5,
        "orders-db": 6,
        "user": 7,
        "user-db": 8,
        "session-db": 9,
        "payment": 10,
        "shipping": 11,
        "rabbitmq": 12,
        "queue-master": 13,
    },
    edges=[
        ("front-end", "catalogue"),
        ("front-end", "carts"),
        ("front-end", "orders"),
        ("front-end", "user"),
        ("front-end", "payment"),
        ("catalogue", "catalogue-db"),
        ("carts", "carts-db"),
        ("orders", "orders-db"),
        ("orders", "shipping"),
        ("orders", "payment"),
        ("orders", "user"),
        ("user", "user-db"),
        ("user", "session-db"),
        ("shipping", "rabbitmq"),
        ("rabbitmq", "queue-master"),
    ],
)


def _load_massive_n50() -> ServiceTopology:
    """Load the N=50 BA-like topology from the JSON shipped with k8s_massive.

    The JSON is produced by the generator that also emits massive-topology-N50.yaml,
    so this keeps the K8s deployment and the GNN edge_index strictly aligned.
    """
    import json
    from pathlib import Path
    p = Path(__file__).parent / "k8s_massive" / "topology_N50.json"
    if not p.exists():
        return None  # silently skip if generator artifact absent
    t = json.loads(p.read_text())
    names: List[str] = list(t["node_names"])
    mapping = {n: i for i, n in enumerate(names)}
    edges = [(names[src], names[dst]) for src, dst in t["edges"]]
    return ServiceTopology(name="massive_n50", node_mapping=mapping, edges=edges)


_MASSIVE_N50 = _load_massive_n50()

TOPOLOGIES: Dict[str, ServiceTopology] = {
    "custom_app": CUSTOM_APP,
    "sock_shop": SOCK_SHOP,
}
if _MASSIVE_N50 is not None:
    TOPOLOGIES["massive_n50"] = _MASSIVE_N50


def get_topology(name: str) -> ServiceTopology:
    key = name.lower().replace("-", "_")
    if key not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{name}'. Available: {sorted(TOPOLOGIES.keys())}"
        )
    return TOPOLOGIES[key]
