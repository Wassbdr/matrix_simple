import torch
from torch_geometric.data import Data
import logging
import sys
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_tensors(filepath: str) -> None:
    """Loads a PyTorch Geometric dataset and verifies tensor properties."""
    logger.info(f"Attempting to load dataset from {filepath}")
    try:
        dataset: List[Data] = torch.load(filepath, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load tensor dataset from {filepath}: {e}")
        sys.exit(1)

    if not dataset:
        logger.error("Dataset is empty.")
        sys.exit(1)

    # Prendre le premier tenseur
    data = dataset[0]
    logger.info("Successfully loaded dataset. Analyzing the first window... ")
    
    # 1. Dimension de la matrice des noeuds X
    x = data.x
    logger.info(f"Shape of X: {x.shape}")
    expected_nodes = 5
    expected_features = 5
    
    if len(x.shape) == 3:
        n_nodes, n_features, window_size = x.shape
        logger.info(f"Nodes: {n_nodes}, Features: {n_features}, Window Size: {window_size}")
        if n_nodes != expected_nodes:
            logger.warning(f"Expected {expected_nodes} nodes, found {n_nodes}")
        if n_features != expected_features:
            logger.warning(f"Expected {expected_features} features, found {n_features}")
    else:
        logger.error(f"Unexpected dimensionality for X. Expected 3D tensor, got {len(x.shape)}D.")
        
    # 2. Dimension et contenu de la matrice d'adjacence (edge_index)
    edge_index = data.edge_index
    logger.info(f"Shape of edge_index: {edge_index.shape}")
    if len(edge_index.shape) == 2 and edge_index.shape[0] == 2:
        logger.info(f"Number of edges: {edge_index.shape[1]}")
    else:
        logger.error("Unexpected dimensionality for edge_index.")
        
    logger.info(f"Adjacency Matrix (edge_index) content:\n{edge_index}")
    
    # 3. Presence de valeurs NaN ou Infinies
    has_nan = bool(torch.isnan(x).any())
    has_inf = bool(torch.isinf(x).any())
    
    if has_nan or has_inf:
        logger.error("Data X contains NaN or Inf values! This will explode gradients.")
    else:
        logger.info("No NaN or Inf values detected in X. Gradients should be stable.")

    # 4. Statistiques de base (moyenne, variance) d'une feature (ex: usage CPU)
    # L'usage CPU est la première feature (index 0)
    # Normalement StandardScaler met mu ~ 0 et var ~ 1
    cpu_feature = x[:, 0, :]
    mean_val = float(torch.mean(cpu_feature))
    var_val = float(torch.var(cpu_feature))
    
    logger.info(f"Statistics for Feature 0 (CPU Usage) in this window:")
    logger.info(f" - Mean: {mean_val:.4f}")
    logger.info(f" - Variance: {var_val:.4f}")
    
    if abs(mean_val) > 0.5 or abs(var_val - 1.0) > 0.5:
        logger.warning("Mean and Variance are far from 0 and 1. Was StandardScaler properly applied across the whole dataset?")
    else:
        logger.info("Mean and Variance are close to 0 and 1, confirming StandardScaler normalization.")

if __name__ == "__main__":
    # Example usage
    # verify_tensors("dataset_windows.pt")
    pass
