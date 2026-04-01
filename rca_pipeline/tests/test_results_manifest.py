"""Unit tests for canonical results manifest generation."""

import json
from pathlib import Path

from rca_pipeline.results_manifest import (
    ArtifactSpec,
    _build_reconciliation_rows,
    _extract_simulated_snapshot,
    _parse_scalar_metric,
    build_manifest,
)


def test_parse_scalar_metric_handles_numeric_and_pm_strings():
    assert _parse_scalar_metric(0.25) == 0.25
    assert _parse_scalar_metric("0.9795±0.007") == 0.9795
    assert _parse_scalar_metric("0.9544+/-0.008") == 0.9544


def test_reconciliation_rows_detect_numeric_deltas():
    canonical = {
        "N20": {
            "STA Select-then-Analyze (k=5)::top1": 0.7691,
            "STA Select-then-Analyze (k=5)::mrr": 0.8230,
        }
    }
    legacy = {
        "N20": {
            "STA Select-then-Analyze (k=5)::top1": 0.9795,
            "STA Select-then-Analyze (k=5)::mrr": 0.9833,
        }
    }
    rows = _build_reconciliation_rows(canonical, legacy, source_path="legacy.json")
    assert len(rows) == 2
    top1_row = next(r for r in rows if r["metric"] == "top1")
    assert top1_row["delta_canonical_minus_legacy"] < 0


def test_extract_simulated_snapshot_reads_run_metrics():
    blob = {
        "runs": [
            {
                "N": 20,
                "metrics": {
                    "STA Select-then-Analyze (k=5)": {"top1": 0.7, "mrr": 0.8},
                    "Full-Graph GCN (N=20)": {"top1": 0.9, "mrr": 0.95},
                },
            }
        ]
    }
    snap = _extract_simulated_snapshot(blob)
    assert "N20" in snap
    assert snap["N20"]["STA Select-then-Analyze::top1"] == 0.7


def test_build_manifest_with_minimal_artifact(tmp_path: Path):
    repo = tmp_path
    p = repo / "rca_pipeline" / "benchmark_full_prepublish.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"runs": []}), encoding="utf-8")

    manifest = build_manifest(
        repo,
        artifacts=[
            ArtifactSpec(
                path="rca_pipeline/benchmark_full_prepublish.json",
                generation_command="python -m rca_pipeline.benchmark --output-json rca_pipeline/benchmark_full_prepublish.json",
            )
        ],
    )
    rows = manifest["artifacts"]
    assert len(rows) == 1
    assert rows[0]["exists"] is True
    assert "sha256" in rows[0]
