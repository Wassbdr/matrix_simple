"""Local compatibility layer exposing get_node_names for rca_pipeline/load_rcaeval.py."""

from __future__ import annotations

from typing import Any, Dict, List


def get_node_names(raw_data: Dict[str, Any]) -> List[str]:
    names = raw_data.get("node_names", [])
    if names:
        return names

    # Fallback to root_cause labels if node_names is absent.
    labels = sorted({inc.get("root_cause", "unknown") for inc in raw_data.get("incidents", [])})
    return labels
