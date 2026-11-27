"""
loaders/observit_loader.py - Charger des fenêtres PyG depuis le labo K8s observ-it.

Flux attendu :
  1) `dataset_builder.py` (Chaos + Locust + Prometheus) → CSV `dataset_<run_id>.csv`
  2) `graph_builder.GraphBuilder` → snapshots → fenêtres [N, F, W] + `root_cause`

Le format de sortie est aligné sur `RCAEvalLoader.load()` :
    (List[Data], node_names, edge_index)

où chaque `Data` a `.x` [N, F, W], `.edge_index`, `.root_cause` (liste de str), `.y`.
Avec `loki_url`, enrichit chaque fenêtre avec `.x_logs` [N, 8] via LokiClient.
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from rca_pipeline.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _merge_csv_paths(paths: list[str], out_path: str) -> None:
    dfs = sorted(paths)
    frames = []
    for p in dfs:
        frames.append(pd.read_csv(p))
    if not frames:
        raise ValueError("empty CSV list")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "pod"])
    df.to_csv(out_path, index=False)


def resolve_observit_csv(path: str) -> str:
    """
    Si `path` est un fichier .csv, le retourne tel quel.
    Si `path` est un répertoire, fusionne tous les `dataset_*.csv` dans un fichier temporaire.
    """
    if os.path.isfile(path):
        if not path.lower().endswith(".csv"):
            raise ValueError(f"Expected a .csv file, got: {path}")
        return path

    if os.path.isdir(path):
        pattern = os.path.join(path, "dataset_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No dataset_*.csv under {path}. Run dataset_builder.py first."
            )
        merged_path = os.path.join(path, "_merged_observit.csv")
        logger.info("Merging %d CSV files → %s", len(files), merged_path)
        _merge_csv_paths(files, merged_path)
        return merged_path

    raise FileNotFoundError(path)


def enrich_with_loki(
    windows: list[Data],
    node_names: list[str],
    loki_url: str,
    namespace: str = "sock-shop",
    service_label: str = "app",
    window_seconds: float = 24.0,
    batch_size: int = 50,
) -> list[Data]:
    """
    Attach `.x_logs` [N, 8] to each window using Loki log features.

    Queries Loki for each window’s time range (last_ts - window_seconds, last_ts).
    Falls back to zeros for services with no logs. Processes in batches to limit
    memory usage.
    """
    from rca_pipeline.log_feature_extractor import extract_log_features
    from rca_pipeline.loki_connector import LokiClient

    client = LokiClient(loki_url)
    N = len(node_names)
    enriched = 0

    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size]
        for w in batch:
            end_ts = float(w.time_step) if hasattr(w, "time_step") else 0.0
            start_ts = end_ts - window_seconds
            if end_ts == 0.0:
                w.x_logs = torch.zeros((N, 8), dtype=torch.float)
                continue

            try:
                log_map = client.fetch_lines_by_service(
                    start=start_ts,
                    end=end_ts,
                    services=node_names,
                    namespace=namespace,
                    service_label=service_label,
                )
            except Exception as exc:
                logger.warning("Loki fetch failed for ts=%.0f: %s", end_ts, exc)
                w.x_logs = torch.zeros((N, 8), dtype=torch.float)
                continue

            feats = np.stack([
                extract_log_features(log_map.get(svc), window_seconds=window_seconds)
                for svc in node_names
            ])
            w.x_logs = torch.from_numpy(feats)
            enriched += 1

    logger.info("Enriched %d/%d windows with Loki log features", enriched, len(windows))
    return windows


def load_observit(
    csv_path_or_dir: str,
    window_size: int = 12,
    topology_name: str | None = None,
    loki_url: str | None = None,
    loki_namespace: str = "sock-shop",
) -> tuple[list[Data], list[str], torch.Tensor]:
    """
    Build sliding windows from observ-it Prometheus export(s).

    Args:
        csv_path_or_dir: fichier CSV unique ou répertoire contenant dataset_*.csv
        window_size: longueur temporelle W (comme RCAEval)
        loki_url: si fourni, enrichit chaque fenêtre avec .x_logs [N, 8]
        loki_namespace: namespace Kubernetes à interroger dans Loki

    Returns:
        data_list, node_names (same order as GraphBuilder indices), edge_index
    """
    resolved = resolve_observit_csv(csv_path_or_dir)
    builder = GraphBuilder(resolved, topology_name=topology_name)
    snapshots = builder.build_snapshots()
    if not snapshots:
        raise RuntimeError(
            "No snapshots built - check CSV columns (timestamp, pod, features, label, root_cause)."
        )

    windows = builder.create_sliding_windows(snapshots, window_size)
    if not windows:
        raise RuntimeError(
            f"Not enough snapshots ({len(snapshots)}) for window_size={window_size}."
        )

    # Ne garder que les fenêtres avec cause racine connue (pour l’entraînement supervisé RCA)
    labeled = [w for w in windows if hasattr(w, "root_cause") and w.root_cause and w.root_cause[0]]
    if len(labeled) < len(windows):
        logger.info(
            "Dropped %d windows without root_cause (baseline-only)",
            len(windows) - len(labeled),
        )

    if not labeled:
        raise RuntimeError(
            "No labeled windows - ensure chaos experiments set root_cause during anomaly period."
        )

    node_names = builder.get_node_names()

    if loki_url:
        logger.info("Fetching Loki log features from %s (ns=%s)", loki_url, loki_namespace)
        window_seconds = window_size * 2.0
        labeled = enrich_with_loki(
            labeled, node_names, loki_url,
            namespace=loki_namespace,
            window_seconds=window_seconds,
        )

    return labeled, node_names, builder.edge_index.clone()
