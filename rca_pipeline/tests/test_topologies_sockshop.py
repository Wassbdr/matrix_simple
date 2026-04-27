"""Topology configurability + sock-shop pod mapping regressions."""

from rca_pipeline.graph_builder import GraphBuilder
from rca_pipeline.topologies import get_topology

SOCKSHOP_PODS = {
    "front-end-55c969566c-stlj7": "front-end",
    "catalogue-6d8b8988d-gzdd5": "catalogue",
    "catalogue-db-5c796cfb68-5vrwd": "catalogue-db",
    "carts-f4f6c98b9-zpdfn": "carts",
    "carts-db-594b68dc85-ljk5r": "carts-db",
    "orders-685d756fb5-fllrv": "orders",
    "orders-db-67c8d5f459-7688v": "orders-db",
    "user-6776ddd957-s57rl": "user",
    "user-db-7dbb5fcd6f-xv9d8": "user-db",
    "session-db-67c8bfcb9c-qh8sc": "session-db",
    "payment-56bcfc857f-dbb4g": "payment",
    "shipping-5c7bc7f48f-rt6cb": "shipping",
    "rabbitmq-56df5b4654-zxh7x": "rabbitmq",
    "queue-master-5f4477db4c-9m824": "queue-master",
}


def _make_builder(topology_name: str) -> GraphBuilder:
    topo = get_topology(topology_name)
    gb = GraphBuilder.__new__(GraphBuilder)
    gb.csv_path = "unused"
    gb.topology = topo
    gb.node_mapping = dict(topo.node_mapping)
    gb.edge_index = topo.edge_index()
    return gb


def test_sock_shop_topology_shape():
    topo = get_topology("sock_shop")
    assert len(topo.node_mapping) == 14
    assert topo.edge_index().shape == (2, 15)


def test_sock_shop_pod_mapping_is_unambiguous():
    gb = _make_builder("sock_shop")
    topo = gb.topology
    inv = {v: k for k, v in topo.node_mapping.items()}
    for pod, expected in SOCKSHOP_PODS.items():
        idx = gb._map_pod_to_node(pod)
        assert idx >= 0, f"{pod} -> unmapped"
        assert inv[idx] == expected, f"{pod} -> {inv[idx]}, expected {expected}"


def test_custom_app_backward_compat():
    gb = _make_builder("custom_app")
    assert gb._map_pod_to_node("frontend-abc") == 0
    assert gb._map_pod_to_node("database-5") == 4
    assert gb._map_pod_to_node("cache-7") == 3
    assert gb._map_pod_to_node("random-other") == -1
