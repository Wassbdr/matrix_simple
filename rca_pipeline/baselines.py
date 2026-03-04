import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Tuple
from torch_geometric.data import Data

NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']
NAME_TO_IDX = {name: i for i, name in enumerate(NODE_NAMES)}

# -------------------------------------------------------------
# Baseline 1: Pure Supervised GCN (No GIB, No Constrastive Loss)
# -------------------------------------------------------------
class PureGCNBaseline(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, hidden_dim: int, window_size: int, num_classes: int = 5):
        super(PureGCNBaseline, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        
        # Temporal convolution to reduce window_size -> 1
        self.temp_conv1 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.temp_conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=window_size)
        
        # Spatial convolution (Message Passing)
        self.spat_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.spat_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classification head: predicts which node is the root cause
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index):
        # x shape: [V, F, W]
        v, f, w = x.shape
        out = F.relu(self.temp_conv1(x))
        out = self.temp_conv2(out) # [V, hidden_dim, 1]
        out = out.squeeze(-1)      # [V, hidden_dim]
        
        # Spatial
        out = F.relu(self.spat_conv1(out, edge_index))
        out = F.relu(self.spat_conv2(out, edge_index)) # [V, hidden_dim]
        
        # Flatten and classify
        out = out.flatten() # [V * hidden_dim]
        logits = self.classifier(out) # [num_classes]
        return logits


def _get_anomalies(data_list: List[Data]) -> List[Data]:
    """Extract anomalous windows with valid root cause labels."""
    return [d for d in data_list
            if d.y is not None and d.y.item() == 1
            and hasattr(d, 'root_cause') and len(d.root_cause) > 0
            and d.root_cause[0] in NAME_TO_IDX]


def train_pure_gcn(train_data: List[Data], test_data: List[Data], epochs: int = 100, hidden_dim: int = 32):
    train_anomalies = _get_anomalies(train_data)
    test_anomalies = _get_anomalies(test_data)
    
    if not train_anomalies or not test_anomalies:
        print(f"PureGCN: Not enough anomalous data (train={len(train_anomalies)}, test={len(test_anomalies)}).")
        return 0, 0, 0
        
    sample = train_anomalies[0]
    num_nodes = sample.x.size(0)
    num_features = sample.x.size(1)
    window_size = sample.x.size(2)
    
    model = PureGCNBaseline(num_nodes, num_features, hidden_dim, window_size, num_classes=num_nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_anomalies:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            target = torch.tensor(NAME_TO_IDX[data.root_cause[0]], dtype=torch.long)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation Loop
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_anomalies:
            logits = model(data.x, data.edge_index)
            pred = torch.argmax(logits).item()
            y_true.append(NAME_TO_IDX[data.root_cause[0]])
            y_pred.append(pred)
            
    # Weighted metrics for multi-class
    p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return p, r, f1

# -------------------------------------------------------------
# Baseline 2: Random Forest (Flattened Spatial-Temporal Features)
# -------------------------------------------------------------
def train_random_forest(train_data: List[Data], test_data: List[Data]) -> Tuple[float, float, float]:
    train_anomalies = _get_anomalies(train_data)
    test_anomalies = _get_anomalies(test_data)
    
    if not train_anomalies or not test_anomalies:
        print(f"RandomForest: Not enough anomalous data (train={len(train_anomalies)}, test={len(test_anomalies)}).")
        return 0, 0, 0
        
    # Flatten Data.x [V, F, W] -> [V * F * W] 1D vector per sample
    X_train = np.array([d.x.numpy().flatten() for d in train_anomalies])
    y_train = np.array([NAME_TO_IDX[d.root_cause[0]] for d in train_anomalies])
    
    X_test = np.array([d.x.numpy().flatten() for d in test_anomalies])
    y_test = np.array([NAME_TO_IDX[d.root_cause[0]] for d in test_anomalies])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    p = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    r = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return p, r, f1
