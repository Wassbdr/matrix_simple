"""
Shared constants and utilities for the RCA pipeline.
Single source of truth for node names, indices, and dataset splitting.
"""

import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

# Default 5-node K8S topology (used by legacy scripts: train.py, benchmark.py)
NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
NUM_NODES = len(NODE_NAMES)


def set_all_seeds(seed: int) -> None:
    """Reproducible RNG: Python, NumPy, PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_deterministic(deterministic: bool = True) -> None:
    """
    Best-effort determinism for PyTorch.

    Note: Some ops remain non-deterministic on GPU depending on versions/hardware.
    """
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this API.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def runtime_env_info() -> Dict[str, object]:
    """Small environment payload for JSON run outputs."""
    info: Dict[str, object] = {
        "python": None,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    try:
        import sys
        info["python"] = sys.version
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            info["cuda_version"] = torch.version.cuda
        except Exception:
            info["cuda_version"] = None
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            info["cudnn_version"] = None
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = None
    return info


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


def stratified_split_shuffle(
    dataset: List[Data],
    train_ratio: float = 0.8,
    name_to_idx: Optional[Dict[str, int]] = None,
    seed: int = 42,
    keep_at_least_one_in_test: bool = True,
) -> Tuple[List[Data], List[Data]]:
    """
    Stratified split with per-class shuffle.

    Notes:
    - This is safer than deterministic/chronological splits when the dataset list order
      is correlated with time or scenario batches.
    - We keep at least one sample in both splits for classes with >1 samples (if enabled),
      so ranking metrics remain evaluable per root-cause class.
    """
    idx_map = name_to_idx if name_to_idx is not None else NAME_TO_IDX
    rng = random.Random(seed)

    healthy = [d for d in dataset if d.y is None or d.y.item() == 0]
    anomalous = [
        d for d in dataset
        if d.y is not None and d.y.item() == 1
        and hasattr(d, "root_cause") and len(d.root_cause) > 0
        and d.root_cause[0] in idx_map
    ]

    rng.shuffle(healthy)
    h_split = int(len(healthy) * train_ratio)
    train_data = healthy[:h_split]
    test_data = healthy[h_split:]

    by_rc: Dict[str, List[Data]] = {}
    for d in anomalous:
        by_rc.setdefault(d.root_cause[0], []).append(d)

    for rc in sorted(by_rc.keys()):
        items = by_rc[rc]
        rng.shuffle(items)
        if len(items) <= 1:
            s = 1
        else:
            s = int(len(items) * train_ratio)
            if keep_at_least_one_in_test:
                s = max(1, min(s, len(items) - 1))
            else:
                s = max(1, s)
        train_data.extend(items[:s])
        test_data.extend(items[s:])

    return train_data, test_data
