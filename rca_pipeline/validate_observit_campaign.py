#!/usr/bin/env python3
"""
Validate ObservIT campaign completeness against SOTA-oriented criteria.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp",
    "pod",
    "cpu_usage",
    "ram_usage",
    "network_tx",
    "network_rx",
    "error_rate",
    "label",
    "root_cause",
}


def _load_csvs(input_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(input_dir, "dataset_*.csv")))


def _summarize(paths: List[str]) -> Dict[str, object]:
    all_rows = 0
    required_missing: Dict[str, List[str]] = {}
    root_causes = Counter()
    failure_types = Counter()
    scenarios = Counter()
    labeled_rows = 0
    latency_present_files = 0

    for p in paths:
        df = pd.read_csv(p)
        all_rows += len(df)
        missing = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
        if missing:
            required_missing[p] = missing
            continue

        if "latency_p99" in df.columns:
            latency_present_files += 1

        an = df[df["label"] == 1]
        labeled_rows += len(an)

        for rc, n in an["root_cause"].fillna("").astype(str).value_counts().to_dict().items():
            if rc.strip():
                root_causes[rc] += int(n)

        if "failure_type" in an.columns:
            for ft, n in an["failure_type"].fillna("unknown").astype(str).value_counts().to_dict().items():
                failure_types[ft] += int(n)
        if "scenario" in an.columns:
            for sc, n in an["scenario"].fillna("unknown").astype(str).value_counts().to_dict().items():
                scenarios[sc] += int(n)

    return {
        "files": len(paths),
        "rows_total": int(all_rows),
        "rows_labeled": int(labeled_rows),
        "missing_columns": required_missing,
        "root_causes": dict(root_causes),
        "failure_types": dict(failure_types),
        "scenarios": dict(scenarios),
        "latency_p99_files": latency_present_files,
    }


def _evaluate(summary: Dict[str, object]) -> Dict[str, object]:
    missing_columns = summary["missing_columns"]  # type: ignore[index]
    root_causes: Dict[str, int] = summary["root_causes"]  # type: ignore[assignment]
    failure_types: Dict[str, int] = summary["failure_types"]  # type: ignore[assignment]
    rows_labeled: int = summary["rows_labeled"]  # type: ignore[assignment]

    checks = {
        "schema_ok": len(missing_columns) == 0,
        "enough_labeled_rows": rows_labeled >= 200,
        "root_cause_coverage>=3": len(root_causes) >= 3,
        "failure_type_coverage>=2": len(failure_types) >= 2,
    }
    checks["overall_pass"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ObservIT dataset campaign quality")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing dataset_*.csv")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    paths = _load_csvs(args.input_dir)
    if not paths:
        raise SystemExit(f"No dataset_*.csv found in {args.input_dir}")

    summary = _summarize(paths)
    checks = _evaluate(summary)

    print("\n=== ObservIT Campaign Validation ===")
    print(json.dumps({"summary": summary, "checks": checks}, indent=2, ensure_ascii=False))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "checks": checks}, f, indent=2, ensure_ascii=False)
        print(f"\nValidation JSON saved to {args.output_json}")

    if not checks["overall_pass"]:
        raise SystemExit("Campaign is not yet SOTA-complete. See checks above.")


if __name__ == "__main__":
    main()
