"""
Per-scenario aggregation for ObservIT cluster results.

Reads the canonical ObservIT JSON artifacts and produces a
scenario x model breakdown table (markdown) with Avg@1 mean +/- std
per chaos scenario. Maps root-cause node names to chaos scenarios
and failure types (hard / cascade / gray).

Usage:
    python -m rca_pipeline.analyze_per_scenario \\
        --observit results_transmission/paper_benchmark_observit.json \\
        --observit-mm results_transmission/paper_benchmark_observit_mm.json \\
        --output docs/PER_SCENARIO_BREAKDOWN.md
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

CHAOS_SCENARIOS = [
    ("catalogue",     "stress-cpu-catalogue",                       "hard"),
    ("carts",         "stress-mem-carts",                           "hard"),
    ("orders-db",     "network-delay-orders-to-orders-db",          "cascade"),
    ("catalogue-db",  "network-loss-catalogue-to-catalogue-db",     "gray"),
    ("user",          "pod-kill-user",                              "hard"),
]


def mean_std(values: List[float]) -> tuple:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def aggregate_per_rc(per_rc_list: List[Dict[str, float]]) -> Dict[str, tuple]:
    if not per_rc_list:
        return {}
    rcs = list(per_rc_list[0].keys())
    out = {}
    for rc in rcs:
        vals = [seed[rc] for seed in per_rc_list if rc in seed]
        out[rc] = mean_std(vals)
    return out


def load_observit(path: Path) -> Dict[str, Dict[str, tuple]]:
    with open(path) as f:
        data = json.load(f)
    obs = data["observit"]
    result = {}
    for model in ("STA", "GATv2", "Late-STGCN+GIB"):
        key = f"{model}_per_root_cause"
        if key in obs:
            result[model] = aggregate_per_rc(obs[key])
    return result


def render_table(metrics_only: Dict[str, Dict[str, tuple]],
                 mm: Dict[str, Dict[str, tuple]]) -> str:
    lines = []
    lines.append("# ObservIT cluster - per-scenario breakdown")
    lines.append("")
    lines.append("Source: `results_transmission/paper_benchmark_observit{,_mm}.json`.")
    lines.append("Five chaos scenarios from `scripts/run_sockshop_campaign.sh`, "
                 "five seeds each (42, 123, 7, 1337, 2025).")
    lines.append("Values are Avg@1 mean +/- std across 5 seeds.")
    lines.append("")
    lines.append("## Scenario x Model matrix")
    lines.append("")
    header = "| Scenario | Failure Type | Root Cause | STA | GATv2 | Late-STGCN+GIB |"
    sep    = "|---|---|---|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for rc, scenario, ftype in CHAOS_SCENARIOS:
        row = [f"`{scenario}`", f"**{ftype}**", f"`{rc}`"]
        for model in ("STA", "GATv2", "Late-STGCN+GIB"):
            if model in metrics_only and rc in metrics_only[model]:
                m, s = metrics_only[model][rc]
                row.append(f"{m:.3f} +/- {s:.3f}")
            else:
                row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Aggregated by failure type")
    lines.append("")
    lines.append("| Failure type | Scenarios | STA mean Avg@1 | GATv2 mean Avg@1 | Late-STGCN+GIB mean Avg@1 |")
    lines.append("|---|---|---:|---:|---:|")
    for ftype in ("hard", "cascade", "gray"):
        scenarios = [(rc, sc) for rc, sc, ft in CHAOS_SCENARIOS if ft == ftype]
        scen_list = ", ".join(sc for _, sc in scenarios)
        cells = [f"**{ftype}**", scen_list]
        for model in ("STA", "GATv2", "Late-STGCN+GIB"):
            if model not in metrics_only:
                cells.append("n/a")
                continue
            means = [metrics_only[model][rc][0]
                     for rc, _ in scenarios
                     if rc in metrics_only[model]]
            if means:
                cells.append(f"{sum(means)/len(means):.3f}")
            else:
                cells.append("n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Observations")
    lines.append("")
    # Compute observations from the matrix
    obs = []
    sta = metrics_only.get("STA", {})
    gatv2 = metrics_only.get("GATv2", {})
    gray_rc = "catalogue-db"
    if gray_rc in sta and gray_rc in gatv2:
        sta_gray, gatv2_gray = sta[gray_rc][0], gatv2[gray_rc][0]
        obs.append(f"- **Gray failure (`{gray_rc}` / network-loss)**: "
                   f"STA Avg@1 = {sta_gray:.3f}, GATv2 Avg@1 = {gatv2_gray:.3f}. "
                   "This is the hardest scenario for STA's hard top-k selector when "
                   "the signal sits in a sub-threshold metric.")
    cascade_rc = "orders-db"
    if cascade_rc in sta and cascade_rc in gatv2:
        sta_c, gatv2_c = sta[cascade_rc][0], gatv2[cascade_rc][0]
        obs.append(f"- **Cascade failure (`{cascade_rc}` / network-delay)**: "
                   f"STA Avg@1 = {sta_c:.3f}, GATv2 Avg@1 = {gatv2_c:.3f}. "
                   "Multi-hop propagation; GATv2's soft attention can spread weight "
                   "along the cascade chain.")
    hard_scens = [(rc, sc) for rc, sc, ft in CHAOS_SCENARIOS if ft == "hard"]
    hard_sta = [sta[rc][0] for rc, _ in hard_scens if rc in sta]
    hard_gatv2 = [gatv2[rc][0] for rc, _ in hard_scens if rc in gatv2]
    if hard_sta and hard_gatv2:
        obs.append(f"- **Hard failures** (CPU stress, memory stress, pod-kill): "
                   f"STA mean = {sum(hard_sta)/len(hard_sta):.3f}, "
                   f"GATv2 mean = {sum(hard_gatv2)/len(hard_gatv2):.3f}. "
                   "Both models handle isolated, high-signal anomalies well.")
    for line in obs:
        lines.append(line)
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observit", required=True, type=Path,
                        help="Path to paper_benchmark_observit.json")
    parser.add_argument("--observit-mm", type=Path,
                        help="Path to paper_benchmark_observit_mm.json (optional)")
    parser.add_argument("--output", type=Path,
                        help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    metrics_only = load_observit(args.observit)
    mm = load_observit(args.observit_mm) if args.observit_mm else {}

    md = render_table(metrics_only, mm)
    if args.output:
        args.output.write_text(md)
        print(f"Wrote {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
