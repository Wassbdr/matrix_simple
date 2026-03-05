"""
Shared constants and utilities for the RCA pipeline.
Single source of truth for node names, indices, and dataset splitting.
"""

from typing import Dict, List, Tuple
from torch_geometric.data import Data

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
NUM_NODES = len(NODE_NAMES)


def stratified_split(dataset: List[Data], train_ratio: float = 0.8) -> Tuple[List[Data], List[Data]]:
    """
    Deterministic train/test split.
    - Healthy windows: split chronologically.
    - Anomalous windows: split per root-cause type by ratio.

    Used by both train.py and benchmark.py to guarantee identical splits.
    """
    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in NAME_TO_IDX]

    h_split = int(len(healthy) * train_ratio)
    train_data = healthy[:h_split]
    test_data = healthy[h_split:]

    by_rc: Dict[str, list] = {}
    for d in anomalous:
        rc = d.root_cause[0]
        by_rc.setdefault(rc, []).append(d)

    for rc, items in by_rc.items():
        split = max(1, int(len(items) * train_ratio))
        train_data.extend(items[:split])
        test_data.extend(items[split:])

    return train_data, test_data
