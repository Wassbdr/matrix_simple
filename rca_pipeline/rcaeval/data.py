"""Local compatibility layer exposing load_dataset for rca_pipeline/load_rcaeval.py."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _resolve_dataset_path(dataset_name: str) -> str:
    return "rca_pipeline/test_integration_data.pt"


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    path = _resolve_dataset_path(dataset_name)
    payload = torch.load(path, weights_only=False)

    dataset = payload.get("dataset", []) if isinstance(payload, dict) else payload
    node_names: List[str] = payload.get("node_names", []) if isinstance(payload, dict) else []

    incidents: List[Dict[str, Any]] = []
    for d in dataset:
        if not hasattr(d, "x"):
            continue
        ts = d.x.permute(2, 0, 1).cpu().numpy()  # [T, N, F]

        root_cause = "unknown"
        if hasattr(d, "root_cause") and d.root_cause:
            root_cause = d.root_cause[0]

        incidents.append(
            {
                "time_series": ts,
                "root_cause": root_cause,
                "is_recurrent": bool(getattr(d, "is_recurrent", False)),
                "pattern_type": getattr(d, "pattern_type", "unknown"),
                "severity": float(getattr(d, "severity", 1.0)),
            }
        )

    topology: Dict[str, List[str]] = {}
    if dataset:
        first = dataset[0]
        if hasattr(first, "edge_index"):
            ei = first.edge_index.cpu().numpy()
            topology = {name: [] for name in node_names}
            for src, dst in zip(ei[0], ei[1]):
                src_i, dst_i = int(src), int(dst)
                if 0 <= src_i < len(node_names) and 0 <= dst_i < len(node_names):
                    src_name = node_names[src_i]
                    dst_name = node_names[dst_i]
                    if dst_name not in topology[src_name]:
                        topology[src_name].append(dst_name)

    return {
        "node_names": node_names,
        "incidents": incidents,
        "topology": topology,
    }
