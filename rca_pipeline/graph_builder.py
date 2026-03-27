import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Transforms Prometheus CSV metrics into PyTorch Geometric spatial-temporal graphs.
    """
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        
        # 5 microservices mapping to node indices
        self.node_mapping: Dict[str, int] = {
            "frontend": 0,
            "auth": 1,
            "backend": 2,
            "cache": 3,
            "database": 4
        }
        
        # Directed Dependencies: 
        # Frontend(0) -> Auth(1)
        # Frontend(0) -> Backend(2)
        # Backend(2) -> Cache(3)
        # Backend(2) -> Database(4)
        self.edge_index = torch.tensor([
            [0, 0, 2, 2],
            [1, 2, 3, 4]
        ], dtype=torch.long)
        
        self.features: List[str] = [
            "cpu_usage", "ram_usage", "network_tx", "network_rx", "error_rate"
        ]
        self.optional_features: List[str] = ["latency_p99"]

    def get_node_names(self) -> List[str]:
        """Return node names sorted by graph index."""
        return [n for n, _ in sorted(self.node_mapping.items(), key=lambda kv: kv[1])]

    def _map_pod_to_node(self, pod_name: str) -> int:
        """Map a pod name to a graph node index.
        
        Uses segment-based matching: splits on '-' and checks if any segment
        starts with the node key. Resolves ambiguity by preferring the longest
        matching key (e.g. 'database' over 'data').
        """
        pod_name_lower = str(pod_name).lower()
        segments = pod_name_lower.replace('_', '-').split('-')
        
        best_match_idx = -1
        best_match_len = 0
        for key, idx in self.node_mapping.items():
            for seg in segments:
                if seg == key or seg.startswith(key):
                    if len(key) > best_match_len:
                        best_match_len = len(key)
                        best_match_idx = idx
                    break
        return best_match_idx

    def build_snapshots(self) -> List[Data]:
        """Reads the CSV and builds a list of graph snapshots (one per timestamp)."""
        logger.info(f"Loading dataset from {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return []

        # Optional compatibility: include latency when present (e.g. prometheus_connector).
        active_features = list(self.features)
        for feat in self.optional_features:
            if feat in df.columns and feat not in active_features:
                active_features.append(feat)
        if len(active_features) != len(self.features):
            logger.info("Detected optional metrics in CSV: using features=%s", active_features)
        
        # Associate rows with node indices
        n_rows_before = len(df)
        df['node_idx'] = df['pod'].apply(self._map_pod_to_node)
        unmapped_rows = int((df['node_idx'] == -1).sum())
        if n_rows_before > 0:
            logger.info(
                "Pod mapping coverage: mapped=%d/%d (%.1f%%), dropped=%d",
                n_rows_before - unmapped_rows,
                n_rows_before,
                100.0 * (n_rows_before - unmapped_rows) / n_rows_before,
                unmapped_rows,
            )
        if unmapped_rows > 0:
            sample_unmapped = (
                df.loc[df['node_idx'] == -1, 'pod']
                .dropna()
                .astype(str)
                .unique()[:8]
                .tolist()
            )
            if sample_unmapped:
                logger.warning("Unmapped pod examples: %s", sample_unmapped)
        df = df[df['node_idx'] != -1]
        
        if df.empty:
            logger.warning("No valid pods mapped. CSV might be empty or missing correct names.")
            return []
            
        # Aggregate over multiple pod replicas per microservice using average
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0)
        grouped = df.groupby(['timestamp', 'node_idx'])[active_features].mean().reset_index()
        
        # Merge label and root_cause back
        labels_df = df.groupby('timestamp')[['label', 'root_cause']].first().reset_index()
        grouped = grouped.merge(labels_df, on='timestamp')
        
        snapshots: List[Data] = []
        timestamps = sorted(grouped['timestamp'].unique())
        
        for ts in timestamps:
            ts_data = grouped[grouped['timestamp'] == ts]
            x = torch.zeros((len(self.node_mapping), len(active_features)), dtype=torch.float)
            idxs = ts_data['node_idx'].to_numpy(dtype=np.int64)
            feats = ts_data[active_features].to_numpy(dtype=np.float32)
            if len(idxs) > 0:
                x[idxs] = torch.from_numpy(feats)
            snapshot = Data(x=x, edge_index=self.edge_index)
            snapshot.time_step = ts  # type: ignore
            
            # Extract label and root_cause
            lbl = ts_data['label'].max()
            snapshot.y = torch.tensor([lbl], dtype=torch.long)
            
            # Root cause string
            rc = str(ts_data['root_cause'].iloc[0]) if not pd.isna(ts_data['root_cause'].iloc[0]) else ""
            snapshot.root_cause = [rc]
            
            snapshots.append(snapshot)
            
        logger.info(f"Successfully processed {len(snapshots)} temporal graph snapshots.")
        return snapshots

    def create_sliding_windows(self, snapshots: List[Data], window_size: int) -> List[Data]:
        """
        Creates temporal sliding windows out of a sequential list of graphs.
        Splits snapshots into contiguous chunks based on time gaps to avoid 
        windows crossing different experiment boundaries.
        The resulting X tensor will have dimensions: [num_nodes, num_features, window_size].
        """
        if len(snapshots) < window_size:
            logger.warning(f"Not enough snapshots ({len(snapshots)}) to form a window of size {window_size}.")
            return []
            
        # 1. Identify contiguous chunks based on timestamp gaps
        # We assume regular intervals (e.g. ~2s). A jump > 10s indicates a new experiment
        chunks = []
        current_chunk = [snapshots[0]]
        
        for i in range(1, len(snapshots)):
            prev_ts = float(snapshots[i-1].time_step)
            curr_ts = float(snapshots[i].time_step)
            # If gap > 15s, split chunk
            if (curr_ts - prev_ts) > 15.0:
                chunks.append(current_chunk)
                current_chunk = [snapshots[i]]
            else:
                current_chunk.append(snapshots[i])
        chunks.append(current_chunk)
        
        logger.info(f"Split dataset into {len(chunks)} continuous chunks.")

        # NOTE: Normalization is NOT applied here to avoid data leakage.
        # The caller must normalize after train/test split using fit_scaler()
        # on training data only, then apply_scaler() on both splits.

        windows: List[Data] = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            if chunk_len < window_size:
                logger.warning(f"Chunk {chunk_idx} has {chunk_len} snaps (< {window_size}), skipping.")
                continue
                
            chunk_x = torch.stack([snap.x for snap in chunk])
            T_chunk = chunk_x.size(0)
            
            for i in range(T_chunk - window_size + 1):
                # Extract window: [window_size, num_nodes, num_features]
                window_x = chunk_x[i : i + window_size]
                # Permute to [num_nodes, num_features, window_size]
                permuted_window_x = window_x.permute(1, 2, 0)
                
                window_data = Data(x=permuted_window_x, edge_index=self.edge_index)
                
                # Inherit labels from the last snapshot
                last_snap = chunk[i + window_size - 1]
                if hasattr(last_snap, 'y'):
                    window_data.y = last_snap.y
                if hasattr(last_snap, 'root_cause'):
                    window_data.root_cause = last_snap.root_cause
                    
                windows.append(window_data)
                
        logger.info(f"Generated {len(windows)} sliding windows (size {window_size}).")
        return windows

    @staticmethod
    def fit_scaler(data_list: List[Data]) -> StandardScaler:
        """Fit a StandardScaler on a list of Data objects (training set only).
        x shape per sample: [num_nodes, num_features, window_size]."""
        all_x = torch.cat([d.x for d in data_list], dim=0)  # [total_nodes, F, W]
        # Flatten across nodes and timesteps to get per-feature statistics
        flat = all_x.permute(1, 0, 2).reshape(all_x.size(1), -1).T.numpy()  # [total_nodes*W, F]
        scaler = StandardScaler()
        scaler.fit(flat)
        logger.info(f"Scaler fit on {len(data_list)} samples (mean={scaler.mean_}, std={scaler.scale_})")
        return scaler

    @staticmethod
    def apply_scaler(data_list: List[Data], scaler: StandardScaler) -> List[Data]:
        """Apply a pre-fitted StandardScaler to a list of Data objects (in-place)."""
        mean = torch.tensor(scaler.mean_, dtype=torch.float).view(1, -1, 1)  # [1, F, 1]
        std = torch.tensor(scaler.scale_, dtype=torch.float).view(1, -1, 1)  # [1, F, 1]
        for d in data_list:
            d.x = (d.x - mean) / std
        return data_list

    def save_windows(self, windows: List[Data], filepath: str) -> None:
        """Saves the list of Data objects to a PyTorch file."""
        try:
            torch.save(windows, filepath)
            logger.info(f"Successfully saved {len(windows)} windowed graphs to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save windows to {filepath}: {e}")

if __name__ == "__main__":
    pass
