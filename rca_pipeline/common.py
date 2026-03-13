"""
Shared constants and utilities for the RCA pipeline.
Single source of truth for node names, indices, and dataset splitting.
"""

import torch
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

# Default 5-node K8S topology (used by legacy scripts: train.py, benchmark.py)
NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
NUM_NODES = len(NODE_NAMES)


def load_windows_dataset(filepath: str) -> List[Data]:
    """Load a windowed dataset .pt file.

    Supports both:
    - New format: dict {'train': [...], 'test': [...], 'scaler': ...} — returns train+test combined.
    - Legacy format: plain List[Data].
    """
    payload = torch.load(filepath, weights_only=False)
    if isinstance(payload, dict) and 'train' in payload and 'test' in payload:
        return payload['train'] + payload['test']
    return payload


def load_dataset_payload(filepath: str):
    """Load a windowed dataset .pt file, returning (windows, node_names).

    node_names is a list of strings (e.g. ['frontend', 'auth', ...]).
    Falls back to the default 5-node NODE_NAMES for legacy list-format files.
    """
    payload = torch.load(filepath, weights_only=False)
    if isinstance(payload, dict) and 'train' in payload and 'test' in payload:
        windows = payload['train'] + payload['test']
        names = payload.get('node_names', NODE_NAMES)
        return windows, names
    return payload, NODE_NAMES


def stratified_split(
    dataset: List[Data],
    train_ratio: float = 0.8,
    name_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[List[Data], List[Data]]:
    """
    Deterministic train/test split.
    - Healthy windows: split chronologically.
    - Anomalous windows: split per root-cause type by ratio.

    Args:
        dataset: list of PyG Data objects.
        train_ratio: fraction for training.
        name_to_idx: optional mapping for dynamic topologies (N>5).
                     Falls back to the default 5-node NAME_TO_IDX.
    """
    idx_map = name_to_idx if name_to_idx is not None else NAME_TO_IDX

    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [d for d in dataset
                 if d.y is not None and d.y.item() == 1
                 and hasattr(d, 'root_cause') and len(d.root_cause) > 0
                 and d.root_cause[0] in idx_map]

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
