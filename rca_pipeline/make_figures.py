"""Generate the paper figures from the canonical benchmark JSON artifacts.

Reads the ``paper_benchmark_*.json`` files under ``results_transmission/`` and
emits vector PDFs (one per figure) into the chosen output directory. The figures
are designed to drop straight into ``docs/arxiv/preprint.tex`` /
``docs/rapport/rapport.tex`` via ``\\includegraphics``.

Figures:
    F1  scaling.pdf      Avg@1 and inference latency vs N (simulated BA graphs)
    F2  real.pdf         Avg@1 / MRR per RCAEval dataset (metrics-only)
    F3  observit.pdf     Avg@1 / MRR per model on the ObservIT cluster
    F4  multimodal.pdf   metrics-only vs +multimodal Avg@1 (the negative result)

Usage:
    python -m rca_pipeline.make_figures                       # defaults
    python -m rca_pipeline.make_figures --out docs/arxiv/figures
    python -m rca_pipeline.make_figures --sim results_transmission/paper_benchmark_sim_10seed.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)

# Stable colour per model family across all figures.
COLORS = {
    "STA": "#1f77b4",
    "Late-STGCN+GIB": "#2ca02c",
    "GATv2": "#ff7f0e",
    "Full-Graph GCN": "#9467bd",
    "Random Forest": "#8c564b",
    "MLP": "#7f7f7f",
}

# Simulated-benchmark model display-key prefixes → canonical label.
SIM_MODELS: List[Tuple[str, str]] = [
    ("STA Select-then-Analyze", "STA"),
    ("Late Bottleneck STGCN+GIB", "Late-STGCN+GIB"),
    ("Late Bottleneck GATv2", "GATv2"),
    ("Full-Graph GCN", "Full-Graph GCN"),
    ("Random Forest", "Random Forest"),
    ("MLP Baseline", "MLP"),
]
SIM_LATENCY_MODELS = {"STA", "Late-STGCN+GIB", "GATv2", "Full-Graph GCN"}


# ── IO helpers ─────────────────────────────────────────────────────────────--
def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as fh:
        return json.load(fh)


def _match(metrics: Dict[str, dict], prefix: str) -> Optional[dict]:
    """Return the metrics dict whose display key starts with ``prefix``."""
    for key, val in metrics.items():
        if key.startswith(prefix) and isinstance(val, dict):
            return val
    return None


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok]   {path}")


# ── Figures ──────────────────────────────────────────────────────────────────
def fig_scaling(data: dict, out_dir: str) -> None:
    runs = sorted(data["runs"], key=lambda r: r["N"])
    sizes = [r["N"] for r in runs]

    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for prefix, label in SIM_MODELS:
        ys, es = [], []
        for r in runs:
            m = _match(r["metrics"], prefix)
            ys.append(m["top1"] * 100 if m else float("nan"))
            es.append(m.get("top1_std", 0.0) * 100 if m else 0.0)
        ax_acc.errorbar(
            sizes, ys, yerr=es, marker="o", capsize=3, label=label,
            color=COLORS[label], linewidth=1.6,
        )

    ax_acc.set_xlabel("Number of services $N$")
    ax_acc.set_ylabel("Top-1 Avg@1 (%)")
    ax_acc.set_title("(a) Accuracy vs graph size")
    ax_acc.set_xticks(sizes)
    ax_acc.legend(fontsize=7, ncol=2)

    for prefix, label in SIM_MODELS:
        if label not in SIM_LATENCY_MODELS:
            continue
        ys, es = [], []
        for r in runs:
            m = _match(r["metrics"], prefix)
            lat = m.get("mean_latency_ms") if m else None
            ys.append(lat if lat is not None else float("nan"))
            es.append(m.get("mean_latency_ms_std", 0.0) if m else 0.0)
        ax_lat.errorbar(
            sizes, ys, yerr=es, marker="s", capsize=3, label=label,
            color=COLORS[label], linewidth=1.6,
        )
    ax_lat.set_xlabel("Number of services $N$")
    ax_lat.set_ylabel("Inference latency (ms)")
    ax_lat.set_title("(b) Inference latency vs graph size")
    ax_lat.set_xticks(sizes)
    ax_lat.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir, "scaling.pdf")


def _grouped_bars(ax, groups, series, ylabel, title) -> None:
    """series: list of (label, color, [(mean, std) per group])."""
    n_groups = len(groups)
    n_series = len(series)
    width = 0.8 / n_series
    x = list(range(n_groups))
    for i, (label, color, vals) in enumerate(series):
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        means = [v[0] * 100 for v in vals]
        stds = [v[1] * 100 for v in vals]
        ax.bar(offs, means, width=width, yerr=stds, capsize=3,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)


def fig_real(data: dict, out_dir: str) -> None:
    datasets = [d for d in ("trainticket", "sockshop", "online-boutique") if d in data]
    labels = {"trainticket": "TrainTicket", "sockshop": "SockShop",
              "online-boutique": "Online Boutique"}
    groups = [labels[d] for d in datasets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, metric, title in (
        (ax1, "avg1", "(a) Avg@1"),
        (ax2, "mrr", "(b) MRR"),
    ):
        series = []
        for model in ("STA", "GATv2"):
            vals = [(data[d][model][metric], data[d][model][f"{metric}_std"])
                    for d in datasets]
            series.append((model, COLORS[model], vals))
        _grouped_bars(ax, groups, series, f"{metric.upper()} (%)", title)
    fig.suptitle("RCAEval real-world datasets (metrics-only, 5 seeds)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "real.pdf")


def fig_observit(data: dict, out_dir: str) -> None:
    obs = data["observit"]
    models = [m for m in ("STA", "GATv2", "Late-STGCN+GIB") if m in obs]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    series = []
    for metric, mlabel in (("avg1", "Avg@1"), ("mrr", "MRR")):
        vals = [(obs[m][metric], obs[m][f"{metric}_std"]) for m in models]
        color = "#1f77b4" if metric == "avg1" else "#ff7f0e"
        series.append((mlabel, color, vals))
    _grouped_bars(ax, models, series, "Score (%)",
                  "ObservIT live cluster — sock-shop (5 seeds)")
    fig.tight_layout()
    _save(fig, out_dir, "observit.pdf")


def fig_multimodal(real: dict, mm: dict, out_dir: str) -> None:
    datasets = [d for d in ("trainticket", "online-boutique")
                if d in real and d in mm and "STA-MM" in mm[d]]
    labels = {"trainticket": "TrainTicket", "online-boutique": "Online Boutique"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.6),
                             squeeze=False)
    for ax, d in zip(axes[0], datasets):
        groups = ["STA", "GATv2"]
        mo = [(real[d]["STA"]["avg1"], real[d]["STA"]["avg1_std"]),
              (real[d]["GATv2"]["avg1"], real[d]["GATv2"]["avg1_std"])]
        mm_vals = [(mm[d]["STA-MM"]["avg1"], mm[d]["STA-MM"]["avg1_std"]),
                   (mm[d]["GATv2-MM"]["avg1"], mm[d]["GATv2-MM"]["avg1_std"])]
        series = [
            ("metrics-only", "#4c72b0", mo),
            ("+ multimodal", "#c44e52", mm_vals),
        ]
        _grouped_bars(ax, groups, series, "Avg@1 (%)", labels[d])
    fig.suptitle("Multimodal fusion vs metrics-only (negative result)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "multimodal.pdf")


# ── Entry point ────────────────────────────────────────────────────────────--
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures from benchmark JSON.")
    ap.add_argument("--results-dir", default="results_transmission")
    ap.add_argument("--out", default="docs/arxiv/figures")
    ap.add_argument("--sim", default=None, help="override sim JSON path")
    ap.add_argument("--real", default=None, help="override real JSON path")
    ap.add_argument("--observit", default=None, help="override observit JSON path")
    ap.add_argument("--real-mm", default=None, help="override real-MM JSON path")
    args = ap.parse_args()

    rd = args.results_dir
    sim_p = args.sim or os.path.join(rd, "paper_benchmark_sim.json")
    real_p = args.real or os.path.join(rd, "paper_benchmark_real.json")
    obs_p = args.observit or os.path.join(rd, "paper_benchmark_observit.json")
    mm_p = args.real_mm or os.path.join(rd, "paper_benchmark_real_mm.json")

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing figures to {args.out}/")

    sim = _load(sim_p)
    if sim:
        fig_scaling(sim, args.out)
    real = _load(real_p)
    if real:
        fig_real(real, args.out)
    obs = _load(obs_p)
    if obs:
        fig_observit(obs, args.out)
    mm = _load(mm_p)
    if real and mm:
        fig_multimodal(real, mm, args.out)


if __name__ == "__main__":
    main()
