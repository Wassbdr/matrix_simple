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


TOPOLOGIES: Dict[str, ServiceTopology] = {
    "custom_app": CUSTOM_APP,
    "sock_shop": SOCK_SHOP,
}


def get_topology(name: str) -> ServiceTopology:
    key = name.lower().replace("-", "_")
    if key not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{name}'. Available: {sorted(TOPOLOGIES.keys())}"
        )
    return TOPOLOGIES[key]
