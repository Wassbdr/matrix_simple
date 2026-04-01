"""Tests for comparability matrix generation."""

import json
from pathlib import Path

from rca_pipeline.comparability_matrix import build_matrix


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_matrix_extracts_protocol_rows(tmp_path: Path):
    repo = tmp_path
    _write(
        repo / "rca_pipeline/benchmark_real_long_prepublish.json",
        {
            "trainticket": {
                "config": {
                    "seeds": [42, 123, 7],
                    "max_cases": 90,
                    "enable_drift": True,
                    "normalized": False,
                    "split": {"mode": "varying_split_by_run_seed"},
                    "hyperparams": {"epochs": 100},
                },
                "STA": {"avg1": 0.9},
            }
        },
    )
    _write(
        repo / "rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
        {
            "sockshop": {
                "config": {
                    "use_mm": True,
                    "seeds": [42, 123, 7],
                    "max_cases": 90,
                    "enable_drift": True,
                    "normalized": False,
                    "split": {"mode": "varying_split_by_run_seed"},
                    "hyperparams": {"epochs": 100},
                }
            }
        },
    )

    matrix = build_matrix(repo)
    rows = matrix["protocol_rows"]

    assert len(rows) == 2
    row0 = next(r for r in rows if r["dataset"] == "trainticket" and r["mode"] == "metrics_only")
    assert row0["seeds"] == "42,123,7"
    assert row0["split_mode"] == "varying_split_by_run_seed"

    row1 = next(r for r in rows if r["dataset"] == "sockshop" and r["mode"] == "full_mm_strict")
    assert row1["mm_status"] == "skipped_with_reason"
