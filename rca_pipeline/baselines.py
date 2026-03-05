import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import List, Tuple
from torch_geometric.data import Data

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}


def get_ranking_metrics(y_true_list: List[int], y_pred_logits_list: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Top-1 Accuracy, Top-2 Accuracy, and Mean Reciprocal Rank (MRR).
    """
    top1, top2, mrr = 0.0, 0.0, 0.0
    for y_true, logits in zip(y_true_list, y_pred_logits_list):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        ranked = np.argsort(logits)[::-1]
        rank = np.where(ranked == y_true)[0][0] + 1
        if rank <= 1:
            top1 += 1
        if rank <= 2:
            top2 += 1
        mrr += 1.0 / rank

    n = len(y_true_list)
    if n == 0:
        return 0, 0, 0
    return top1 / n, top2 / n, mrr / n


def _get_anomalies(data_list: List[Data]) -> List[Data]:
    """Extract anomalous windows with valid root cause labels."""
    return [d for d in data_list
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and len(d.root_cause) > 0
            and d.root_cause[0] in NAME_TO_IDX]


# -------------------------------------------------------------
# Baseline 1: Pure Supervised GCN (No GIB, No Contrastive Loss)
# Batched training, residual GCN, proper temporal conv, LayerNorm
# -------------------------------------------------------------
class PureGCNBaseline(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, hidden_dim: int, num_classes: int = 5):
        super().__init__()
        self.num_nodes = num_nodes

        # Temporal: two Conv1d layers with same-padding + attention pooling
        self.temp_conv1 = nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1)
        self.temp_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.temp_norm = nn.BatchNorm1d(hidden_dim)
        self.temp_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )

        # Spatial: 2-layer GCN with residual + LayerNorm
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.spat_norm = nn.LayerNorm(hidden_dim)

        # Per-node scorer (same architecture as our STGCN NodeScorer)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [batch, V, F, W] or [V, F, W]
        Returns:
            logits: [batch, V] or [V]
        """
        squeezed = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True

        B, V, F_, W = x.shape
        # Temporal
        h = x.view(B * V, F_, W)
        h = F.relu(self.temp_conv1(h))
        h = F.relu(self.temp_conv2(h))
        # Attention pooling over time
        h_t = h.permute(0, 2, 1)  # [B*V, W, H]
        attn = F.softmax(self.temp_attn(h_t).squeeze(-1), dim=-1)  # [B*V, W]
        h = (h_t * attn.unsqueeze(-1)).sum(dim=1)  # [B*V, H]
        h = self.temp_norm(h)

        # Spatial (batched)
        H = h.size(1)
        if B > 1:
            offsets = torch.arange(B, device=h.device) * V
            batch_ei = edge_index.unsqueeze(0).expand(B, -1, -1) + offsets.view(-1, 1, 1)
            batch_ei = batch_ei.reshape(2, -1)
        else:
            batch_ei = edge_index
        residual = h
        h = F.relu(self.gcn1(h, batch_ei))
        h = self.gcn2(h, batch_ei)
        h = self.spat_norm(h + residual)

        # Per-node scoring
        h = h.view(B, V, H)
        logits = self.scorer(h).squeeze(-1)  # [B, V]

        if squeezed:
            logits = logits.squeeze(0)
        return logits


def train_pure_gcn(train_data: List[Data], test_data: List[Data],
                   epochs: int = 150, hidden_dim: int = 32, batch_size: int = 256,
                   seed: int = 42) -> Tuple[float, float, float]:
    torch.manual_seed(seed)
    train_anomalies = _get_anomalies(train_data)
    test_anomalies = _get_anomalies(test_data)

    if not train_anomalies or not test_anomalies:
        print(f"PureGCN: Not enough data (train={len(train_anomalies)}, test={len(test_anomalies)}).")
        return 0, 0, 0

    sample = train_anomalies[0]
    num_nodes, num_features = sample.x.size(0), sample.x.size(1)
    edge_index = sample.edge_index

    # Stack into tensors
    X_train = torch.stack([d.x for d in train_anomalies])  # [N, V, F, W]
    y_train = torch.tensor([NAME_TO_IDX[d.root_cause[0]] for d in train_anomalies], dtype=torch.long)

    model = PureGCNBaseline(num_nodes, num_features, hidden_dim, num_classes=num_nodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    n = len(train_anomalies)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb, edge_index)  # [B, V]
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    y_true, y_logits = [], []
    X_test = torch.stack([d.x for d in test_anomalies])
    y_test_labels = [NAME_TO_IDX[d.root_cause[0]] for d in test_anomalies]
    with torch.no_grad():
        # Batched inference
        for start in range(0, len(test_anomalies), batch_size):
            xb = X_test[start:start + batch_size]
            logits = model(xb, edge_index)  # [B, V]
            for i in range(logits.size(0)):
                y_true.append(y_test_labels[start + i])
                y_logits.append(logits[i].cpu().numpy())

    return get_ranking_metrics(y_true, y_logits)


# -------------------------------------------------------------
# Baseline 2: Random Forest (Flattened Spatial-Temporal Features)
# -------------------------------------------------------------
def train_random_forest(train_data: List[Data], test_data: List[Data],
                        seed: int = 42) -> Tuple[float, float, float]:
    train_anomalies = _get_anomalies(train_data)
    test_anomalies = _get_anomalies(test_data)

    if not train_anomalies or not test_anomalies:
        return 0, 0, 0

    X_train = np.array([d.x.numpy().flatten() for d in train_anomalies])
    y_train = np.array([NAME_TO_IDX[d.root_cause[0]] for d in train_anomalies])
    X_test = np.array([d.x.numpy().flatten() for d in test_anomalies])
    y_test = np.array([NAME_TO_IDX[d.root_cause[0]] for d in test_anomalies])

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced",
                                 max_depth=20, min_samples_leaf=5)
    clf.fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)
    logits = np.zeros((len(X_test), len(NODE_NAMES)))
    for i, c in enumerate(clf.classes_):
        logits[:, c] = y_pred_probs[:, i]

    return get_ranking_metrics(y_test, logits)


# -------------------------------------------------------------
# Baseline 3: MLP on Flattened Features (batched training)
# -------------------------------------------------------------
class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(train_data: List[Data], test_data: List[Data],
                        epochs: int = 150, hidden_dim: int = 128, batch_size: int = 256,
                        seed: int = 42) -> Tuple[float, float, float]:
    torch.manual_seed(seed)
    train_anomalies = _get_anomalies(train_data)
    test_anomalies = _get_anomalies(test_data)

    if not train_anomalies or not test_anomalies:
        return 0, 0, 0

    X_train = torch.stack([d.x.flatten() for d in train_anomalies])  # [N, V*F*W]
    y_train = torch.tensor([NAME_TO_IDX[d.root_cause[0]] for d in train_anomalies], dtype=torch.long)

    input_dim = X_train.size(1)
    model = MLPBaseline(input_dim, hidden_dim, num_classes=len(NODE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    n = len(train_anomalies)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    X_test = torch.stack([d.x.flatten() for d in test_anomalies])
    y_test_labels = [NAME_TO_IDX[d.root_cause[0]] for d in test_anomalies]
    y_true, y_logits = [], []
    with torch.no_grad():
        for start in range(0, len(test_anomalies), batch_size):
            xb = X_test[start:start + batch_size]
            logits = model(xb)
            for i in range(logits.size(0)):
                y_true.append(y_test_labels[start + i])
                y_logits.append(logits[i].cpu().numpy())

    return get_ranking_metrics(y_true, y_logits)
