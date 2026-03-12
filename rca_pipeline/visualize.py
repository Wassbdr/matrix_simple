"""
RCA Visualization: topology + root-cause probability heatmap.

Left panel  : Graph topology with node labels.
Right panel : Same topology colored by root-cause probability.
Bottom      : Actual vs predicted root cause verdict.

Supports any number of nodes (N) -- topology and layout are derived
dynamically from the dataset's edge_index.

Usage:
    # Late Bottleneck (5-node K8S)
    python visualize.py --checkpoint checkpoints/stgcn_rca_final.pt \
                        --dataset ../dataset_windows.pt

    # Select-then-Analyze (scalable simulated)
    python visualize.py --checkpoint checkpoints/sta_semi_sup.pt \
                        --dataset scalable_dataset_N100.pt --arch sta --k 10
"""

import argparse
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────

def _load_dataset(path: str):
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, dict):
        return payload["dataset"], payload.get("node_names")
    return payload, None


def _load_model(ckpt_path: str, arch: str, N: int, F_: int, hidden_dim: int, k: int):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    hd = ckpt.get("hidden_dim", hidden_dim)
    if arch == "late":
        from model import SpatioTemporalRCA
        model = SpatioTemporalRCA(num_features=F_, hidden_dim=hd, num_nodes=N)
    else:
        from model_sta import SelectThenAnalyze
        k_ = ckpt.get("k", k)
        model = SelectThenAnalyze(num_features=F_, hidden_dim=hd, num_nodes=N, k=k_)
    model.load_state_dict(state)
    model.eval()
    return model


def _get_probs(model, data, arch: str) -> np.ndarray:
    """Return [N] softmax probabilities for a single window."""
    with torch.no_grad():
        x = data.x.unsqueeze(0)
        ei = data.edge_index
        if arch == "late":
            N = data.x.size(0)
            x_flat = x.view(N, x.size(2), x.size(3))
            h_t = model.temporal_block(x_flat)
            h_s = model.spatial_block(h_t, ei)
            logits = model.gib.mlp(h_s).squeeze(-1)
        else:
            _, _, _, sel_logits = model(x, ei)
            logits = sel_logits.squeeze(0)
    return F.softmax(logits, dim=0).cpu().numpy()


def _build_graph(edge_index: torch.Tensor, node_names: list):
    """Build a networkx DiGraph from edge_index and node names."""
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(i, label=name)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        G.add_edge(s, d)
    return G


def _compute_layout(G, node_names: list):
    """Compute a spring layout, or use the fixed K8S layout for the 5-node case."""
    k8s_set = {"frontend", "auth", "backend", "cache", "database"}
    if set(node_names) == k8s_set:
        name_to_pos = {
            "frontend": (0.5, 0.9), "auth": (0.15, 0.5), "backend": (0.85, 0.5),
            "cache": (0.65, 0.1), "database": (1.05, 0.1),
        }
        return {i: name_to_pos[n] for i, n in enumerate(node_names)}
    return nx.spring_layout(G, seed=42, k=2.0 / max(1, len(node_names) ** 0.5))


# ── Drawing ──────────────────────────────────────────────────────────

def _draw_topology(ax, G, layout, node_names):
    """Left panel: plain topology with node labels."""
    N = len(node_names)
    colors = plt.cm.Set3(np.linspace(0, 1, max(N, 3)))[:N]
    nx.draw_networkx_nodes(G, layout, ax=ax, node_color=colors, node_size=2500,
                           alpha=0.9, edgecolors="white", linewidths=2)
    nx.draw_networkx_edges(G, layout, ax=ax, width=2, alpha=0.4, arrowsize=18,
                           edge_color="#7f8c8d", connectionstyle="arc3,rad=0.05")
    labels = {i: n for i, n in enumerate(node_names)}
    nx.draw_networkx_labels(G, layout, labels=labels, ax=ax, font_size=8,
                            font_weight="bold")
    ax.set_title("Graph Topology", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")


def _draw_heatmap(ax, G, layout, node_names, probs, predicted, correct):
    """Right panel: heatmap colored by root-cause probability."""
    N = len(node_names)
    cmap = plt.cm.YlOrRd
    sizes = [2000 + 2500 * probs[i] for i in range(N)]
    nodes_plot = nx.draw_networkx_nodes(
        G, layout, ax=ax, node_color=probs.tolist(), cmap=cmap,
        node_size=sizes, alpha=0.9, edgecolors="black", linewidths=2,
        vmin=0.0, vmax=1.0,
    )
    # Highlight predicted node
    pred_idx = node_names.index(predicted)
    border = "#27ae60" if correct else "#e74c3c"
    nx.draw_networkx_nodes(G, layout, ax=ax, nodelist=[pred_idx],
                           node_color=[probs[pred_idx]], cmap=cmap,
                           node_size=[sizes[pred_idx] + 400], alpha=1.0,
                           edgecolors=border, linewidths=3.5, vmin=0.0, vmax=1.0)
    nx.draw_networkx_edges(G, layout, ax=ax, width=1.8, alpha=0.35, arrowsize=16,
                           edge_color="#7f8c8d", connectionstyle="arc3,rad=0.05")
    labels = {i: f"{n}\nP={probs[i]:.3f}" for i, n in enumerate(node_names)}
    nx.draw_networkx_labels(G, layout, labels=labels, ax=ax, font_size=7,
                            font_weight="bold")
    plt.colorbar(nodes_plot, ax=ax, shrink=0.6, pad=0.02, label="Root Cause Probability")
    status = "CORRECT" if correct else "INCORRECT"
    ax.set_title(f"RCA Prediction: {predicted.upper()} ({status})",
                 fontsize=12, fontweight="bold", pad=12,
                 color="#27ae60" if correct else "#e74c3c")
    ax.axis("off")


# ── Entry point ──────────────────────────────────────────────────────

def visualize_dual(checkpoint: str, dataset_path: str, arch: str = "late",
                   hidden_dim: int = 32, k: int = 5,
                   output: str = "graph_heatmap.png"):
    dataset, payload_names = _load_dataset(dataset_path)
    sample = dataset[0]
    N = sample.x.size(0)
    F_ = sample.x.size(1)

    if payload_names:
        node_names = payload_names
    else:
        from common import NODE_NAMES
        node_names = NODE_NAMES
    assert len(node_names) == N

    model = _load_model(checkpoint, arch, N, F_, hidden_dim, k)

    # Find first anomalous window
    anomaly = None
    for d in dataset:
        if d.y is not None and d.y.item() == 1:
            anomaly = d
            break
    if anomaly is None:
        print("No anomalous window found in dataset.")
        return

    actual_rc = anomaly.root_cause[0] if hasattr(anomaly, "root_cause") else "unknown"
    probs = _get_probs(model, anomaly, arch)
    predicted_rc = node_names[int(np.argmax(probs))]
    correct = predicted_rc == actual_rc

    print(f"Actual: {actual_rc} | Predicted: {predicted_rc} | {'OK' if correct else 'WRONG'}")

    G = _build_graph(anomaly.edge_index, node_names)
    layout = _compute_layout(G, node_names)

    fig, (ax_topo, ax_rca) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#fafafa")

    _draw_topology(ax_topo, G, layout, node_names)
    _draw_heatmap(ax_rca, G, layout, node_names, probs, predicted_rc, correct)

    color = "#27ae60" if correct else "#e74c3c"
    symbol = "OK" if correct else "WRONG"
    fig.text(0.5, 0.02,
             f"Actual: {actual_rc.upper()}  |  Predicted: {predicted_rc.upper()}  [{symbol}]",
             ha="center", fontsize=13, fontweight="bold", color=color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor=color, alpha=0.9))
    fig.suptitle("Root Cause Analysis — Topology & Prediction",
                 fontsize=15, fontweight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RCA topology + prediction heatmap.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--arch", type=str, default="late", choices=["late", "sta"])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default="graph_heatmap.png")
    args = parser.parse_args()

    visualize_dual(args.checkpoint, args.dataset, args.arch, args.hidden_dim, args.k,
                   args.output)
