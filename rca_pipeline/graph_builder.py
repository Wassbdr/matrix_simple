import pandas as pd
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

    def _map_pod_to_node(self, pod_name: str) -> int:
        """Heuristically assign a pod name to a graph node index."""
        pod_name_lower = str(pod_name).lower()
        for key, idx in self.node_mapping.items():
            if key in pod_name_lower:
                return idx
        return -1

    def build_snapshots(self) -> List[Data]:
        """Reads the CSV and builds a list of graph snapshots (one per timestamp)."""
        logger.info(f"Loading dataset from {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return []
        
        # Associate rows with node indices
        df['node_idx'] = df['pod'].apply(self._map_pod_to_node)
        df = df[df['node_idx'] != -1]
        
        if df.empty:
            logger.warning("No valid pods mapped. CSV might be empty or missing correct names.")
            return []
            
        # Aggregate over multiple pod replicas per microservice using average
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0)
        grouped = df.groupby(['timestamp', 'node_idx'])[self.features].mean().reset_index()
        
        # Merge label and root_cause back
        labels_df = df.groupby('timestamp')[['label', 'root_cause']].first().reset_index()
        grouped = grouped.merge(labels_df, on='timestamp')
        
        snapshots: List[Data] = []
        timestamps = sorted(grouped['timestamp'].unique())
        
        for ts in timestamps:
            ts_data = grouped[grouped['timestamp'] == ts]
            x = torch.zeros((5, len(self.features)), dtype=torch.float)
            for _, row in ts_data.iterrows():
                idx = int(row['node_idx'])
                x[idx] = torch.tensor([row[feat] for feat in self.features], dtype=torch.float)
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

        # Extract all X matrices: [T, num_nodes, num_features] for global scaling
        all_x = torch.stack([snap.x for snap in snapshots])
        T, num_nodes, num_features = all_x.shape

        # Apply StandardScaler across the time and node dimensions globally
        scaler = StandardScaler()
        reshaped_x = all_x.view(-1, num_features).numpy()
        scaled_x = scaler.fit_transform(reshaped_x)
        all_x_scaled = torch.tensor(scaled_x, dtype=torch.float).view(T, num_nodes, num_features)
        
        # Map scaled features back directly on snapshots to maintain chunk boundaries
        for i, snap in enumerate(snapshots):
            snap.x = all_x_scaled[i]

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
                
        logger.info(f"Generated {len(windows)} sliding windows (size {window_size}) with standard scaling applied.")
        return windows

    def save_windows(self, windows: List[Data], filepath: str) -> None:
        """Saves the list of Data objects to a PyTorch file."""
        try:
            torch.save(windows, filepath)
            logger.info(f"Successfully saved {len(windows)} windowed graphs to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save windows to {filepath}: {e}")

if __name__ == "__main__":
    pass
