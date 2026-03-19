"""
Configuration centrale pour le pipeline RCA.

Ce fichier centralise tous les hyperparamètres, constantes et configurations
pour éviter la duplication et améliorer la maintenabilité.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration pour les modèles."""

    # Late Bottleneck (STGCN + GIB)
    late_bottleneck_hidden_dim: int = 32
    late_bottleneck_gib_temp: float = 0.5
    late_bottleneck_mask_floor: float = 0.01

    # STA (Select-then-Analyze)
    sta_hidden_dim: int = 32
    sta_k_select: int = 10
    sta_gumbel_temp: float = 1.0
    sta_temp_min: float = 0.1

    # Feature dimensions
    num_features: int = 5  # Default: metrics only. Set to 13 for full multi-modal
    window_size: int = 12

    # Multi-modal feature groups (start_idx, end_idx)
    # These define which indices correspond to which modality
    feature_groups: Dict = None

    def __post_init__(self):
        if self.feature_groups is None:
            if self.num_features == 13:
                # Full multi-modal: metrics + traces + logs
                self.feature_groups = {
                    "metrics": (0, 5),   # cpu, memory, latency_p99, error_rate, request_rate
                    "traces": (5, 9),    # span_count, avg_duration, error_spans, critical_path
                    "logs": (9, 13),     # log_volume, error_ratio, unique_errors, entropy
                }
            elif self.num_features == 9:
                # Metrics + one additional modality (traces or logs)
                self.feature_groups = {
                    "metrics": (0, 5),
                    "additional": (5, 9),
                }
            else:
                # Metrics only (default)
                self.feature_groups = {
                    "metrics": (0, self.num_features),
                }

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        if self.num_features == 13:
            return [
                # Metrics (0-4)
                "cpu_usage", "memory_ratio", "latency_p99", "error_rate", "request_rate",
                # Traces (5-8)
                "span_count", "avg_span_duration", "error_span_ratio", "critical_path_ratio",
                # Logs (9-12)
                "log_volume", "error_log_ratio", "unique_error_count", "log_entropy",
            ]
        elif self.num_features == 9:
            return [
                "cpu_usage", "memory_ratio", "latency_p99", "error_rate", "request_rate",
                "feature_5", "feature_6", "feature_7", "feature_8",
            ]
        else:
            return [
                "cpu_usage", "memory_ratio", "latency_p99", "error_rate", "request_rate",
            ][:self.num_features]


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement."""

    # Phase 1: Contrastive
    contrastive_epochs: int = 80
    contrastive_lr: float = 1e-3
    contrastive_temp_start: float = 0.2
    contrastive_temp_end: float = 0.05

    # Phase 2: Supervised
    supervised_epochs: int = 150
    supervised_lr: float = 5e-4

    # GIB Loss
    beta_start: float = 0.0
    beta_end: float = 1.0
    target_sparsity_base: float = 0.01  # Will be adjusted per num_nodes
    min_active_nodes: int = 1
    dispersion_weight: float = 0.25

    # General
    batch_size: int = 256
    weight_decay: float = 1e-4
    patience_early_stop: int = 20


@dataclass
class DataConfig:
    """Configuration pour les datasets."""

    # Default topologies
    node_names_small: List[str] = None
    node_names_large: List[str] = None

    # Generation params
    default_num_samples: int = 5000
    default_anomaly_ratio: float = 0.3
    noise_level: float = 0.1
    propagation_strength: float = 0.6

    # Barabási-Albert params
    ba_m: int = 3  # Edges to attach from new node

    def __post_init__(self):
        if self.node_names_small is None:
            self.node_names_small = [
                'frontend', 'auth', 'backend', 'cache', 'database'
            ]

        if self.node_names_large is None:
            self.node_names_large = [
                'gateway-web', 'gateway-mobile', 'gateway-admin',
                'auth', 'user-svc', 'order-svc', 'payment-svc',
                'notification-svc', 'search-svc', 'recommendation-svc',
                'inventory-svc', 'postgres-primary', 'postgres-replica',
                'redis-cache', 'redis-session', 'rabbitmq',
                'elasticsearch', 'minio', 'monitoring', 'config-svc'
            ]


@dataclass
class PathConfig:
    """Configuration pour les chemins de fichiers."""

    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    logs_dir: str = "logs"
    results_dir: str = "results"

    def ensure_dirs(self):
        """Créer les répertoires s'ils n'existent pas."""
        for dir_path in [self.checkpoint_dir, self.data_dir,
                        self.logs_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Configuration pour les expériences et benchmarks."""

    # Random seeds for reproducibility
    seeds: List[int] = None

    # Scale tests
    test_sizes: List[int] = None

    # Metrics
    k_values: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 7, 456, 789]

        if self.test_sizes is None:
            self.test_sizes = [20, 50, 100, 200]

        if self.k_values is None:
            self.k_values = [1, 2, 5]


# Configuration globale par défaut
DEFAULT_CONFIG = {
    'model': ModelConfig(),
    'training': TrainingConfig(),
    'data': DataConfig(),
    'paths': PathConfig(),
    'experiment': ExperimentConfig(),
}


def get_config(config_name: str = 'default') -> Dict:
    """
    Récupérer une configuration par nom.

    Args:
        config_name: Nom de la configuration ('default', 'fast', 'thorough')

    Returns:
        Dictionnaire de configuration
    """
    if config_name == 'default':
        return DEFAULT_CONFIG.copy()

    elif config_name == 'fast':
        # Configuration pour tests rapides
        config = DEFAULT_CONFIG.copy()
        config['training'].contrastive_epochs = 10
        config['training'].supervised_epochs = 20
        config['data'].default_num_samples = 500
        return config

    elif config_name == 'thorough':
        # Configuration pour expériences complètes
        config = DEFAULT_CONFIG.copy()
        config['training'].contrastive_epochs = 120
        config['training'].supervised_epochs = 200
        config['data'].default_num_samples = 10000
        return config

    else:
        raise ValueError(f"Configuration inconnue: {config_name}")


def update_target_sparsity_for_nodes(config: Dict, num_nodes: int) -> float:
    """
    Calculate target sparsity based on number of nodes.

    Args:
        config: Configuration dictionary
        num_nodes: Number of nodes in the graph

    Returns:
        Target sparsity value (1/num_nodes with minimum floor)
    """
    base_sparsity = config['training'].target_sparsity_base
    target = 1.0 / max(num_nodes, 1)
    return max(target, base_sparsity)