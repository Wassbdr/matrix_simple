"""
Dual visualization: K8S Architecture + RCA Model Prediction.

Left panel  : Real Kubernetes service topology with service types.
Right panel : Same topology with root cause classifier probability heatmap.
Bottom      : Comparison of actual vs predicted root cause.
"""

import argparse
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from torch_geometric.data import Data
from model import SpatioTemporalRCA

# ── Constants ──────────────────────────────────────────────────────────
NODE_NAMES = ['frontend', 'auth', 'backend', 'cache', 'database']

# K8S service metadata
K8S_META = {
    'frontend':  {'type': 'Web Server',   'tech': 'Nginx',     'replicas': 2},
    'auth':      {'type': 'Auth Service', 'tech': 'FastAPI',   'replicas': 1},
    'backend':   {'type': 'API Server',   'tech': 'FastAPI',   'replicas': 2},
    'cache':     {'type': 'Cache',        'tech': 'Redis',     'replicas': 2},
    'database':  {'type': 'Database',     'tech': 'PostgreSQL','replicas': 1},
}

# Service type → color for K8S panel
K8S_COLORS = {
    'frontend':  '#4A90D9',  # Blue
    'auth':      '#9B59B6',  # Purple
    'backend':   '#E67E22',  # Orange
    'cache':     '#27AE60',  # Green
    'database':  '#2C3E50',  # Dark
}

# Shared layout for both panels
LAYOUT = {
    0: (0.5, 0.9),   # frontend
    1: (0.15, 0.5),   # auth
    2: (0.85, 0.5),   # backend
    3: (0.65, 0.1),   # cache
    4: (1.05, 0.1),   # database
}

EDGES = [(0, 1), (0, 2), (2, 3), (2, 4)]

EDGE_LABELS = {
    (0, 1): 'HTTP',
    (0, 2): 'HTTP',
    (2, 3): 'TCP',
    (2, 4): 'TCP',
}


def _build_graph():
    """Build a shared networkx DiGraph for both panels."""
    G = nx.DiGraph()
    for i, name in enumerate(NODE_NAMES):
        G.add_node(i, label=name)
    G.add_edges_from(EDGES)
    return G


def _draw_k8s_panel(ax, G):
    """Draw the K8S architecture panel (left)."""
    node_colors = [K8S_COLORS[NODE_NAMES[i]] for i in range(len(NODE_NAMES))]
    node_sizes = [3500] * len(NODE_NAMES)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, LAYOUT, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='white',
        linewidths=2.5,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, LAYOUT, ax=ax,
        width=2.5,
        alpha=0.5,
        arrowsize=20,
        edge_color='#7f8c8d',
        min_source_margin=18,
        min_target_margin=18,
        connectionstyle='arc3,rad=0.05',
    )

    # Labels: service name + technology
    labels = {}
    for i, name in enumerate(NODE_NAMES):
        meta = K8S_META[name]
        labels[i] = f"{name}\n({meta['tech']})"
    nx.draw_networkx_labels(G, LAYOUT, labels=labels, ax=ax,
                             font_size=9, font_weight='bold', font_color='white')

    # Edge labels (protocol)
    nx.draw_networkx_edge_labels(G, LAYOUT, edge_labels=EDGE_LABELS, ax=ax,
                                  font_size=7, font_color='#555', label_pos=0.3)

    # Legend
    legend_patches = []
    for name in NODE_NAMES:
        meta = K8S_META[name]
        patch = mpatches.Patch(color=K8S_COLORS[name],
                                label=f"{name}: {meta['type']} ({meta['replicas']}r)")
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc='lower left', fontsize=7,
              framealpha=0.8, fancybox=True)

    ax.set_title("Kubernetes Topology\n(rca-sandbox namespace)", fontsize=12,
                 fontweight='bold', pad=15)
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(-0.15, 1.1)
    ax.axis('off')


def _draw_rca_panel(ax, G, rc_probs, actual_rc, predicted_rc, correct):
    """Draw the RCA prediction panel (right)."""
    # Color nodes by classifier probability
    node_colors = [rc_probs[i] for i in range(len(NODE_NAMES))]
    node_sizes = [3000 + 3000 * rc_probs[i] for i in range(len(NODE_NAMES))]

    cmap = plt.cm.YlOrRd

    # Draw nodes
    nodes_plot = nx.draw_networkx_nodes(
        G, LAYOUT, ax=ax,
        node_color=node_colors,
        cmap=cmap,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='black',
        linewidths=2.5,
        vmin=0.0,
        vmax=1.0,
    )

    # Highlight predicted root cause node with a thick border
    pred_idx = NODE_NAMES.index(predicted_rc)
    nx.draw_networkx_nodes(
        G, LAYOUT, ax=ax,
        nodelist=[pred_idx],
        node_color=[rc_probs[pred_idx]],
        cmap=cmap,
        node_size=[node_sizes[pred_idx] + 500],
        alpha=1.0,
        edgecolors='#e74c3c' if correct else '#c0392b',
        linewidths=4,
        vmin=0.0,
        vmax=1.0,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, LAYOUT, ax=ax,
        width=2.0,
        alpha=0.4,
        arrowsize=18,
        edge_color='#7f8c8d',
        min_source_margin=18,
        min_target_margin=18,
        connectionstyle='arc3,rad=0.05',
    )

    # Labels: name + probability
    labels = {}
    for i, name in enumerate(NODE_NAMES):
        labels[i] = f"{name}\nP={rc_probs[i]:.3f}"
    nx.draw_networkx_labels(G, LAYOUT, labels=labels, ax=ax,
                             font_size=9, font_weight='bold')

    # Colorbar
    plt.colorbar(nodes_plot, ax=ax, shrink=0.6, pad=0.02,
                 label="Root Cause Probability")

    status = "✓ CORRECT" if correct else "✗ INCORRECT"
    ax.set_title(f"RCA Model Prediction\nPredicted: {predicted_rc.upper()} ({status})",
                 fontsize=12, fontweight='bold', pad=15,
                 color='#27ae60' if correct else '#e74c3c')
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(-0.15, 1.1)
    ax.axis('off')


def visualize_dual(model_path: str, dataset_path: str, output_path: str = "graph_heatmap.png"):
    """Generate the dual K8S + RCA visualization."""
    print(f"Loading checkpoint {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    print(f"Loading dataset {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    # Find first anomalous window
    anomaly_idx = -1
    for i, data in enumerate(dataset):
        if data.y is not None and data.y.item() == 1:
            anomaly_idx = i
            break

    if anomaly_idx == -1:
        print("Error: No anomalous window found in dataset.")
        return

    data = dataset[anomaly_idx]
    actual_rc = data.root_cause[0] if hasattr(data, 'root_cause') and len(data.root_cause) > 0 else "unknown"
    print(f"Selected anomalous window {anomaly_idx} with actual root cause: {actual_rc}")

    # Initialize model
    nodes = data.x.size(0)
    features = data.x.size(1)
    hidden_dim = checkpoint.get('hidden_dim',
        checkpoint['model_state_dict']['temporal_block.conv1.weight'].size(0))

    model = SpatioTemporalRCA(
        num_nodes=nodes,
        num_features=features,
        hidden_dim=hidden_dim,
        temporal_kernel=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get predictions
    with torch.no_grad():
        x = data.x.unsqueeze(0)
        edge_index = data.edge_index
        z, probs = model(x, edge_index)
        rc_logits = model.classify_root_cause(z)
        rc_probs = F.softmax(rc_logits, dim=-1).squeeze(0).numpy()

    predicted_rc = NODE_NAMES[rc_probs.argmax()]
    correct = (predicted_rc == actual_rc)

    print(f"Classifier probabilities: {dict(zip(NODE_NAMES, [f'{p:.4f}' for p in rc_probs]))}")
    print(f"Actual: {actual_rc} | Predicted: {predicted_rc} | {'✓' if correct else '✗'}")

    # ── Build figure ──────────────────────────────────────────────
    fig, (ax_k8s, ax_rca) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#fafafa')

    G = _build_graph()

    _draw_k8s_panel(ax_k8s, G)
    _draw_rca_panel(ax_rca, G, rc_probs, actual_rc, predicted_rc, correct)

    # Bottom annotation
    verdict_color = '#27ae60' if correct else '#e74c3c'
    verdict_symbol = '✓' if correct else '✗'
    fig.text(0.5, 0.02,
             f"Chaos injected on: {actual_rc.upper()}  →  Model predicted: {predicted_rc.upper()}  {verdict_symbol}",
             ha='center', fontsize=14, fontweight='bold', color=verdict_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=verdict_color, alpha=0.9))

    fig.suptitle("Root Cause Analysis — Infrastructure vs Model",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize K8S Architecture + RCA Prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset (.pt)")
    parser.add_argument("--output", type=str, default="graph_heatmap.png", help="Output image path")

    args = parser.parse_args()
    visualize_dual(args.checkpoint, args.dataset, args.output)
