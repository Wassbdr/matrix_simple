#!/usr/bin/env python3
"""Compare two baseline JSON files and print metric deltas.

Supports both baseline_official.json and baseline_official_compact.json formats.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

PLUS_MINUS_RE = re.compile(r"^\s*([+-]?\d*\.?\d+)(?:\s*[±\+/-]\s*([+-]?\d*\.?\d+))?\s*$")


def parse_value(value: Any) -> Tuple[float | None, float | None]:
    """Parse values like '0.9544±0.008' or numeric values.

    Returns (mean, std) where std can be None.
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    if not isinstance(value, str):
        return None, None

    text = value.strip().replace("+/-", "±")
    match = PLUS_MINUS_RE.match(text)
    if not match:
        return None, None
    mean = float(match.group(1))
    std = float(match.group(2)) if match.group(2) is not None else None
    return mean, std


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_benchmark(payload: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return benchmark map in canonical shape: N -> model -> fields."""
    if "benchmark_top1_mrr_latency" in payload:
        return payload["benchmark_top1_mrr_latency"]
    return payload.get("benchmark_official", {})


def get_metric(model_data: Dict[str, Any], metric_name: str) -> float | None:
    mean, _ = parse_value(model_data.get(metric_name))
    return mean


def pct_points(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta * 100:+.2f} pp"


def as_float(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.4f}"


def compare(old_payload: Dict[str, Any], new_payload: Dict[str, Any]) -> Dict[str, Any]:
    old_bench = extract_benchmark(old_payload)
    new_bench = extract_benchmark(new_payload)

    report: Dict[str, Any] = {
        "summary": {},
        "per_size": {},
    }

    sizes = sorted(set(old_bench.keys()) | set(new_bench.keys()))

    for size in sizes:
        old_models = old_bench.get(size, {})
        new_models = new_bench.get(size, {})
        models = sorted(set(old_models.keys()) | set(new_models.keys()))

        size_report: Dict[str, Any] = {}
        for model in models:
            old_row = old_models.get(model, {})
            new_row = new_models.get(model, {})

            old_top1 = get_metric(old_row, "top1")
            new_top1 = get_metric(new_row, "top1")
            old_mrr = get_metric(old_row, "mrr")
            new_mrr = get_metric(new_row, "mrr")
            old_lat = get_metric(old_row, "latency_ms")
            new_lat = get_metric(new_row, "latency_ms")

            size_report[model] = {
                "old": {
                    "top1": old_top1,
                    "mrr": old_mrr,
                    "latency_ms": old_lat,
                },
                "new": {
                    "top1": new_top1,
                    "mrr": new_mrr,
                    "latency_ms": new_lat,
                },
                "delta": {
                    "top1": (None if old_top1 is None or new_top1 is None else new_top1 - old_top1),
                    "mrr": (None if old_mrr is None or new_mrr is None else new_mrr - old_mrr),
                    "latency_ms": (None if old_lat is None or new_lat is None else new_lat - old_lat),
                },
            }

        report["per_size"][size] = size_report

    old_eval = old_payload.get("evaluate", old_payload.get("evaluate_metrics", {}))
    new_eval = new_payload.get("evaluate", new_payload.get("evaluate_metrics", {}))

    eval_fields = ["precision_weighted", "recall_weighted", "f1_weighted"]
    eval_delta = {}
    for field in eval_fields:
        old_v = old_eval.get(field)
        new_v = new_eval.get(field)
        eval_delta[field] = None if old_v is None or new_v is None else float(new_v) - float(old_v)

    report["summary"]["evaluate_delta"] = eval_delta
    report["summary"]["old_checkpoint"] = old_payload.get("checkpoint")
    report["summary"]["new_checkpoint"] = new_payload.get("checkpoint")

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 90)
    print("BASELINE COMPARISON (new - old)")
    print("=" * 90)

    for size, rows in report["per_size"].items():
        print(f"\n{size}")
        print("-" * 90)
        print(f"{'Model':40} {'Top1':>12} {'MRR':>12} {'Latency(ms)':>14}")
        print("-" * 90)
        for model, data in rows.items():
            d = data["delta"]
            print(
                f"{model[:40]:40} {pct_points(d['top1']):>12} {as_float(d['mrr']):>12} {as_float(d['latency_ms']):>14}"
            )

    eval_delta = report["summary"]["evaluate_delta"]
    print("\nEVALUATE DELTAS")
    print("-" * 90)
    print(f"Precision(weighted): {as_float(eval_delta['precision_weighted'])}")
    print(f"Recall(weighted):    {as_float(eval_delta['recall_weighted'])}")
    print(f"F1(weighted):        {as_float(eval_delta['f1_weighted'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two baseline JSON files.")
    parser.add_argument("--old", required=True, help="Path to old baseline JSON")
    parser.add_argument("--new", required=True, help="Path to new baseline JSON")
    parser.add_argument("--out", help="Optional output JSON path for machine-readable diff")
    args = parser.parse_args()

    old_payload = load_json(Path(args.old))
    new_payload = load_json(Path(args.new))

    report = compare(old_payload, new_payload)
    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote diff JSON: {out_path}")


if __name__ == "__main__":
    main()
