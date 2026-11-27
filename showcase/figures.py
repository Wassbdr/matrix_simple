"""Build showcase/latency_vs_n.png, showcase/poster.png and showcase/showcase.json.

Every number is read from files produced by repository code:
  results/scaling_benchmark.json   (python -m rca_pipeline.scaling_benchmark)
  examples/sockshop_demo/          (python -m rca_pipeline.demo_incident build)
The incident is re-ranked live here (STA from the bundle, BARO on the same candidates).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from rca_pipeline.demo_incident import run_demo  # noqa: E402
from rca_pipeline.topologies import SOCK_SHOP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "showcase"
SCALING = Path(os.environ.get("SCALING_JSON", ROOT / "results" / "scaling_benchmark.json"))
DEMO_EVAL = ROOT / "results" / "sockshop_demo_eval.json"

BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
RED, GREEN, BLUE, ORANGE = "#f85149", "#3fb950", "#58a6ff", "#d29922"
COLORS = {"STA": GREEN, "FullGraphGCN": ORANGE, "LateGATv2": BLUE}
LABELS = {
    "STA": "STA (select k, then analyse)",
    "FullGraphGCN": "Full-graph GCN",
    "LateGATv2": "Full-graph GATv2",
}

LAYOUT = {
    "front-end": (0.0, 3.0),
    "catalogue": (-3.0, 2.0),
    "carts": (-1.5, 2.0),
    "user": (0.2, 2.0),
    "orders": (1.8, 2.0),
    "payment": (3.3, 2.0),
    "catalogue-db": (-3.0, 1.0),
    "carts-db": (-1.5, 1.0),
    "user-db": (-0.4, 1.0),
    "session-db": (0.8, 1.0),
    "orders-db": (1.6, 1.0),
    "shipping": (2.8, 1.0),
    "rabbitmq": (2.8, 0.0),
    "queue-master": (1.4, 0.0),
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )


def _hw(scaling) -> str:
    h = scaling["hardware"]
    return f"{h['cpu']}, {h.get('latency_threads', 1)} thread, batch 1, torch {h['torch']} CPU"


def latency_axes(ax, scaling, compact=False):
    for name in ("STA", "FullGraphGCN", "LateGATv2"):
        rows = [r for r in scaling["latency"] if name in r["models"]]
        if not rows:
            continue
        ns = [r["N"] for r in rows]
        ms = [r["models"][name]["latency"]["median_ms"] for r in rows]
        ax.plot(ns, ms, "o-", color=COLORS[name], lw=2.5, ms=5, label=LABELS[name])
        if not compact:
            ax.annotate(
                f"{ms[-1]:.1f} ms",
                (ns[-1], ms[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                color=COLORS[name],
                fontsize=11,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("services in the graph (N)")
    ax.set_ylabel("CPU inference, median ms")
    ax.grid(True, which="major", color="#30363d", lw=0.6)
    ax.legend(frameon=False, fontsize=9 if compact else 11, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def latency_chart(scaling, path: Path):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.08, 0.30, 0.55, 0.58])
    latency_axes(ax, scaling)
    fig.text(
        0.08,
        0.93,
        "CPU inference latency vs number of services",
        fontsize=18,
        weight="bold",
    )
    # accuracy table
    acc = scaling.get("accuracy", [])
    if acc:
        names = [m for m in ("STA", "FullGraphGCN", "LateGATv2") if m in acc[0]["models"]]
        y = 0.86
        fig.text(0.68, y, "Top-1 accuracy (mean of seeds)", fontsize=14, weight="bold")
        y -= 0.06
        fig.text(0.68, y, "N", color=MUTED)
        for j, m in enumerate(names):
            fig.text(
                0.74 + j * 0.085,
                y,
                m.replace("FullGraph", "Full").replace("LateGATv2", "GATv2"),
                color=COLORS[m],
                fontsize=11,
            )
        for r in acc:
            y -= 0.05
            fig.text(0.68, y, str(r["N"]))
            best = max(r["models"][m]["top1_mean"] for m in names)
            for j, m in enumerate(names):
                v = r["models"][m]["top1_mean"]
                fig.text(0.74 + j * 0.085, y, f"{v:.2f}", weight="bold" if v == best else "normal")
    seeds = ",".join(str(s) for s in scaling["config"]["seeds"])
    fig.text(
        0.08,
        0.14,
        f"Seeded synthetic Barabasi-Albert service graphs, dataset seed "
        f"{scaling['config']['dataset_seed']}, training seeds {seeds}.",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.08,
        0.10,
        f"Latency: untrained weights, one 12-step window, {_hw(scaling)}.",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.08,
        0.06,
        "Regenerate: python -m rca_pipeline.scaling_benchmark",
        color=MUTED,
        fontsize=11,
        family="DejaVu Sans Mono",
    )
    fig.savefig(path)
    plt.close(fig)


def poster(demo, scaling, demo_eval, path: Path):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(
        0.03,
        0.935,
        "Which service broke? STA ranks the root cause first",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.03,
        0.895,
        f"Synthetic Sock Shop {demo['failure_type']} incident (seed 42, "
        f"{len(demo['candidates'])} services). STA reads the model bundle; BARO ranks the same "
        "candidates.",
        color=MUTED,
        fontsize=12,
    )

    ax = fig.add_axes([0.01, 0.10, 0.56, 0.76])
    ax.axis("off")
    g = nx.DiGraph(SOCK_SHOP.edges)
    truth = demo["true_root_cause"]
    sta_top3 = demo["STA"]["top3"]
    nx.draw_networkx_edges(
        g, LAYOUT, ax=ax, edge_color="#30363d", width=1.6, arrows=True, arrowsize=12, node_size=1500
    )
    for n, (x, y) in LAYOUT.items():
        face = "#161b22"
        edge, lw = "#30363d", 1.5
        if n in sta_top3:
            face, edge, lw = "#12361f", GREEN, 2.5
        if n == truth:
            edge, lw = RED, 4
        ax.scatter([x], [y], s=1700, c=face, edgecolors=edge, linewidths=lw, zorder=3)
        ax.text(x, y - 0.36, n, ha="center", va="top", fontsize=10, color=FG, zorder=4)
        if n in sta_top3:
            ax.text(
                x,
                y,
                f"#{sta_top3.index(n) + 1}",
                ha="center",
                va="center",
                fontsize=13,
                weight="bold",
                color=GREEN,
                zorder=4,
            )
    ax.set_xlim(-3.8, 4.0)
    ax.set_ylim(-0.7, 3.5)
    ax.scatter([], [], s=150, c="#161b22", edgecolors=RED, linewidths=3, label="true root cause")
    ax.scatter([], [], s=150, c="#12361f", edgecolors=GREEN, linewidths=2, label="STA top 3")
    ax.legend(loc="lower left", frameon=False, fontsize=11, labelcolor=FG)

    # ranking columns
    x0 = 0.60
    for col, name in enumerate(("STA", "BARO")):
        r = demo[name]
        xc = x0 + col * 0.19
        color = GREEN if name == "STA" else ORANGE
        fig.text(xc, 0.83, name, fontsize=16, weight="bold", color=color)
        fig.text(
            xc,
            0.795,
            f"root cause at rank {r['true_rank']}  |  {r['inference_ms']:.2f} ms",
            fontsize=10,
            color=MUTED,
        )
        for i, s in enumerate(r["ranking"][:5]):
            hit = s == truth
            fig.text(
                xc,
                0.75 - i * 0.042,
                f"{i + 1}. {s}",
                fontsize=12,
                color=RED if hit else FG,
                weight="bold" if hit else "normal",
            )

    a = demo_eval["all"]
    fig.text(
        x0,
        0.50,
        f"Held-out Sock Shop incidents (n={a['n']}), same candidates:",
        fontsize=11,
        color=FG,
    )
    for row, m in enumerate(("avg1", "mrr")):
        label = "top-1" if m == "avg1" else "MRR  "
        fig.text(
            x0,
            0.465 - row * 0.035,
            f"{label}  STA {a['STA'][m]:.2f}  vs  BARO {a['BARO'][m]:.2f}",
            fontsize=11,
            color=FG,
            family="DejaVu Sans Mono",
        )

    ax2 = fig.add_axes([0.645, 0.12, 0.33, 0.25])
    latency_axes(ax2, scaling, compact=True)
    ax2.set_xlabel("services (N)", fontsize=10)
    ax2.set_ylabel("ms (median)", fontsize=10)
    fig.text(
        0.03,
        0.03,
        f"All numbers generated by showcase/make_showcase.sh.  Timings: {_hw(scaling)}.",
        color=MUTED,
        fontsize=10,
    )
    fig.savefig(path)
    plt.close(fig)


def showcase_json(demo, scaling, demo_eval) -> dict:
    hw = _hw(scaling)
    lat = {r["N"]: r["models"] for r in scaling["latency"]}
    n_max = max(lat)
    sta_ms = {n: m["STA"]["latency"]["median_ms"] for n, m in lat.items()}
    full_ms = {n: m["FullGraphGCN"]["latency"]["median_ms"] for n, m in lat.items()}
    acc = {r["N"]: r["models"] for r in scaling.get("accuracy", [])}
    seeds = ",".join(str(s) for s in scaling["config"]["seeds"])
    a, c = demo_eval["all"], demo_eval["cascade"]
    results = [
        {
            "label": f"CPU inference latency at N={n_max}: STA vs full-graph GCN",
            "value": f"{sta_ms[n_max]:.1f} ms vs {full_ms[n_max]:.1f} ms "
            f"(STA {sta_ms[min(lat)]:.1f} ms at N={min(lat)})",
            "command": "python -m rca_pipeline.scaling_benchmark",
            "setup": f"seeded BA graphs, untrained weights, batch 1, {hw}",
        },
    ]
    for n in sorted(acc):
        m = acc[n]
        results.append(
            {
                "label": f"Top-1 accuracy at N={n}, STA vs full-graph GCN",
                "value": f"{m['STA']['top1_mean']:.3f} vs {m['FullGraphGCN']['top1_mean']:.3f}",
                "command": "python -m rca_pipeline.scaling_benchmark",
                "setup": f"synthetic BA graphs, dataset seed {scaling['config']['dataset_seed']}, "
                f"training seeds {seeds}, {scaling['config']['epochs']} epochs, CPU",
            }
        )
    results += [
        {
            "label": "Sock Shop held-out incidents, top-1 STA vs BARO",
            "value": f"{a['STA']['avg1']:.3f} vs {a['BARO']['avg1']:.3f} (n={a['n']}); "
            f"cascades {c['STA']['avg1']:.3f} vs {c['BARO']['avg1']:.3f} (n={c['n']})",
            "command": "python -m rca_pipeline.demo_incident build",
            "setup": "synthetic Sock Shop topology (14 services), seed 42, CPU",
        },
        {
            "label": "Demo incident: rank of the true root cause",
            "value": f"STA rank {demo['STA']['true_rank']} ({demo['STA']['inference_ms']:.2f} ms), "
            f"BARO rank {demo['BARO']['true_rank']} ({demo['BARO']['inference_ms']:.2f} ms)",
            "command": "python -m rca_pipeline.demo_incident run",
            "setup": f"one held-out {demo['failure_type']} incident, root cause {demo['true_root_cause']}, "
            f"seed 42, {hw}",
        },
    ]
    return {
        "slug": "matrix_simple",
        "title": "STA: root-cause analysis for microservice incidents",
        "tagline": "Ranks the service that caused a microservice incident from its metrics, "
        "with CPU latency that stays flat as the system grows.",
        "problem": "When a microservice system degrades, alerts fire on many services at once and "
        "on-call engineers lose time finding the one that started it. Graph models that "
        "look at every service pair get slower as the system grows.",
        "approach": "Select-Then-Analyze (STA) first scores every service cheaply and keeps the k "
        "most suspect ones, then runs a spatio-temporal graph network on that k-node "
        "subgraph only, so cost is O(N + k^2) instead of O(N^2). The model ships as a "
        "self-describing bundle served by FastAPI, compared against the BARO statistical "
        "baseline on the same candidates.",
        "stack": [
            "Python",
            "PyTorch",
            "PyTorch Geometric",
            "FastAPI",
            "Docker Compose",
            "MLflow",
            "Prometheus",
            "Helm",
        ],
        "demo": {
            "gif": "showcase/demo.gif",
            "mp4": "showcase/demo.mp4",
            "poster": "showcase/poster.png",
            "kind": "synthetic",
            "caption": "docker-free local run: the FastAPI service ranks the services of a seeded "
            "synthetic Sock Shop incident via /predict, then the measured latency-vs-N "
            "chart of STA against full-graph models.",
        },
        "reproduce": [
            "pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && "
            "pip install -e '.[serve]' -c requirements/constraints-cpu.txt",
            "docker compose up -d && python scripts/demo_predict.py",
            "bash showcase/make_showcase.sh",
        ],
        "results": results,
        "limits": [
            "All showcase numbers come from synthetic, seeded data; real-trace numbers (RCAEval) "
            "need the external dataset and are not regenerated here.",
            "Latency is measured on CPU with untrained weights at batch 1; no GPU numbers.",
            "BARO needs no training and ties STA on local crash faults; STA's gain is "
            "largest on cascading failures and at larger N.",
        ],
        "role": "TODO",
        "repo": "https://github.com/Wassbdr/matrix_simple",
    }


README = ROOT / "README.md"
RESULTS_START, RESULTS_END = "<!-- results:start -->", "<!-- results:end -->"


def readme_results(demo, scaling, demo_eval) -> str:
    """Key-results block for the README, generated from the same JSON as the poster."""
    lat = {r["N"]: r["models"] for r in scaling["latency"]}
    acc = {r["N"]: r for r in scaling.get("accuracy", [])}
    hw = _hw(scaling)
    n_max = max(lat)
    sta, full = lat[n_max]["STA"]["latency"], lat[n_max]["FullGraphGCN"]["latency"]
    lines = [
        f"- **CPU latency stays flat:** at N={n_max} services STA answers in "
        f"{sta['median_ms']:.1f} ms vs {full['median_ms']:.1f} ms for the full-graph GCN "
        f"({full['median_ms'] / sta['median_ms']:.0f}x). Setup: seeded BA graphs, batch 1, "
        f"1 thread, {hw}.",
        "",
        "| N services | k | top-1 STA | top-1 full-graph GCN | top-1 full-graph GATv2 "
        "| random | STA ms | GCN ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(models, name):
        m = models.get(name)
        return f"{m['top1_mean']:.2f} ± {m['top1_std']:.2f}" if m else "not run"

    for n in sorted(set(acc) | set(lat)):
        r = acc.get(n)
        ms = lat.get(n, {})
        cells = [
            str(n),
            str(r["k"]) if r else "",
            fmt(r["models"], "STA") if r else "",
            fmt(r["models"], "FullGraphGCN") if r else "",
            fmt(r["models"], "LateGATv2") if r else "",
            f"{r['random_top1']:.3f}" if r else "",
            f"{ms['STA']['latency']['median_ms']:.1f}" if ms else "",
            f"{ms['FullGraphGCN']['latency']['median_ms']:.1f}" if ms else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    seeds = sorted({s for r in acc.values() for s in r["seeds"]})
    epochs = sorted({r["epochs"] for r in acc.values()})
    lines += [
        "",
        f"Accuracy: synthetic Barabási-Albert graphs, dataset seed "
        f"{scaling['config']['dataset_seed']}, training seeds {', '.join(map(str, seeds))} "
        f"(mean ± std), {', '.join(map(str, epochs))} epochs, trained on CPU. "
        "Command: `python -m rca_pipeline.scaling_benchmark`.",
        "",
    ]
    a, c, h = demo_eval["all"], demo_eval["cascade"], demo_eval["hard"]
    lines += [
        f"- **STA vs BARO, same candidates** (synthetic Sock Shop, 14 services, seed 42): top-1 "
        f"{a['STA']['avg1']:.2f} vs {a['BARO']['avg1']:.2f} over {a['n']} held-out incidents, "
        f"{c['STA']['avg1']:.2f} vs {c['BARO']['avg1']:.2f} on the {c['n']} cascading ones. "
        f"BARO needs no training and ties STA ({h['BARO']['avg1']:.2f} vs {h['STA']['avg1']:.2f}) "
        f"on the {h['n']} local crash ('hard') faults.",
        f"- **Demo incident** (`{demo['true_root_cause']}`, {demo['failure_type']}): STA ranks it "
        f"#{demo['STA']['true_rank']} in {demo['STA']['inference_ms']:.2f} ms, BARO ranks it "
        f"#{demo['BARO']['true_rank']}. Command: `make demo`.",
    ]
    return "\n".join(lines)


def write_readme_results(block: str) -> None:
    text = README.read_text()
    start, end = text.index(RESULTS_START), text.index(RESULTS_END)
    README.write_text(text[: start + len(RESULTS_START)] + "\n" + block + "\n" + text[end:])


def main():
    _style()
    scaling = json.loads(SCALING.read_text())
    demo_eval = json.loads(DEMO_EVAL.read_text())["metrics"]
    demo = run_demo()
    latency_chart(scaling, OUT / "latency_vs_n.png")
    poster(demo, scaling, demo_eval, OUT / "poster.png")
    (OUT / "showcase.json").write_text(
        json.dumps(showcase_json(demo, scaling, demo_eval), indent=2) + "\n"
    )
    write_readme_results(readme_results(demo, scaling, demo_eval))
    print(
        "wrote showcase/latency_vs_n.png, showcase/poster.png, showcase/showcase.json, README results"
    )


if __name__ == "__main__":
    main()
