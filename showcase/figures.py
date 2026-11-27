"""Draw the showcase from the public incident ranks and measured scaling data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reference import ROOT, baro_ranks

from rca_pipeline.data import DEMO_CASE

OUT = Path(__file__).resolve().parent
ROLE = "Sole developer. Built during my ML research internship at Devoteam (AIOps), where it is used internally."
ROOT_CAUSE = "checkoutservice"
BG, FG, MUTED = "#0d1117", "#e6edf3", "#a3adbc"
GREEN, BLUE = "#40c463", "#58a6ff"


def latency_chart(scaling: dict, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for method, label, color in (("STA", "STA", GREEN), ("FullGraphGCN", "Full-graph GCN", BLUE)):
        points = [
            (row["N"], row["models"][method]["latency"]["median_ms"])
            for row in scaling["latency"] if method in row["models"]
        ]
        nodes, times = zip(*points)
        ax.plot(nodes, times, color=color, label=label, marker="o", linewidth=2.5)
    ax.set(xlabel="Services", ylabel="Median forward time (ms)")
    ax.set_title(f"CPU latency with STA capped at {scaling['latency_config']['sta_k']} candidates", loc="left", pad=18)
    ax.grid(alpha=0.15)
    ax.legend(facecolor=BG, labelcolor=FG, loc="upper left")
    fig.text(0.12, 0.05, hardware(scaling), color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.94, top=0.87, bottom=0.18)
    fig.savefig(destination, facecolor=BG)
    plt.close(fig)


def hardware(scaling: dict) -> str:
    return f"{scaling['hardware']['cpu']}, one CPU thread, batch one"


def poster(sta_rank: int, baro_rank: int, destination: Path) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    fig.text(0.075, 0.86, "RCAEval / ONLINE BOUTIQUE", color=BLUE, fontsize=16, weight="bold")
    fig.text(0.075, 0.72, "Find the service that failed.", color=FG, fontsize=43, weight="bold")
    fig.text(0.075, 0.60, "A CPU fault was injected into the checkout service.",
             color=MUTED, fontsize=17)
    for x, title, value, color in [(0.075, "STA", sta_rank, GREEN), (0.53, "BARO", baro_rank, BLUE)]:
        fig.text(x, 0.38, title, color=color, fontsize=20, weight="bold")
        fig.text(x, 0.22, f"Rank {value}", color=FG, fontsize=54, weight="bold")
    fig.text(0.075, 0.095, f"Public case: {DEMO_CASE}  |  Injected service: {ROOT_CAUSE}",
             color=MUTED, fontsize=14)
    fig.savefig(destination, facecolor=BG)
    plt.close(fig)


def write_showcase(sta_rank: int, baro_rank: int, scaling: dict) -> None:
    path = OUT / "showcase.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    data = {
        "slug": "matrix_simple",
        "title": "STA root-cause ranking",
        "tagline": "Rank likely root-cause services from a metrics export.",
        "problem": "Many services show symptoms when a microservice fails. On-call engineers need to find where the incident started.",
        "approach": "Select-Then-Analyze measures pre/post metric changes, then runs a deep scorer only on the most suspect services.",
        "stack": ["Python", "PyTorch", "PyTorch Geometric", "FastAPI", "Docker"],
        "demo": {
            "gif": "showcase/demo.gif", "mp4": "showcase/demo.mp4", "poster": "showcase/poster.png",
            "kind": "public",
            "caption": "The installed CLI ranks an RCAEval Online Boutique CPU fault and compares it with the cached RCAEval BARO prediction.",
        },
        "reproduce": ["make install INSTALL_TARGET='.[data]'", "make demo", "make showcase"],
        "results": [
            {
                "label": "Held-out RCAEval Online Boutique case: rank of the injected service",
                "value": f"STA {sta_rank}, BARO {baro_rank}",
                "command": "make demo",
                "setup": f"{DEMO_CASE}; pooled RE1 checkpoint, training seed 42; cached RCAEval BARO",
            },
            {
                "label": "Fitted synthetic CPU latency slopes (STA vs full-graph GCN)",
                "value": f"{scaling['latency_slope_ms_per_1000_nodes']['STA']:.2f} vs "
                         f"{scaling['latency_slope_ms_per_1000_nodes']['FullGraphGCN']:.2f} ms per 1,000 services",
                "command": "make scaling",
                "setup": hardware(scaling),
            },
        ],
        "limits": [
            "The showcase is one public case, not an estimate of the packaged checkpoint's accuracy.",
            "The metrics-only comparison cannot run MicroRCA on the same inputs.",
            "STA latency grows with the number of services and is slower than the GCN at small sizes.",
        ],
        "role": old.get("role", ROLE),
        "repo": "https://github.com/Wassbdr/matrix_simple",
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    plt.rcParams.update({"text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 14})
    ranked = json.loads((OUT / "build" / "sta.json").read_text())
    sta_rank = next(item["rank"] for item in ranked if item["service"] == ROOT_CAUSE)
    baro_rank = baro_ranks().index(ROOT_CAUSE) + 1
    scaling = json.loads((ROOT / "results" / "scaling_benchmark.json").read_text())
    latency_chart(scaling, OUT / "latency_vs_n.png")
    poster(sta_rank, baro_rank, OUT / "poster.png")
    write_showcase(sta_rank, baro_rank, scaling)


if __name__ == "__main__":
    main()
