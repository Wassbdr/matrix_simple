#!/usr/bin/env python3
"""Build a canonical results manifest with provenance and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    generation_command: str
    optional: bool = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_scalar_metric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if "±" in cleaned:
            cleaned = cleaned.split("±", 1)[0]
        if "+/-" in cleaned:
            cleaned = cleaned.split("+/-", 1)[0]
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_simulated_snapshot(blob: Dict) -> Dict[str, Dict[str, float]]:
    runs = blob.get("runs", [])
    out: Dict[str, Dict[str, float]] = {}
    model_keys = [
        "STA Select-then-Analyze",
        "Late Bottleneck STGCN+GIB",
        "Late Bottleneck GATv2",
        "Full-Graph GCN",
    ]
    for run in runs:
        n = run.get("N")
        metrics = run.get("metrics", {})
        if n is None or not isinstance(metrics, dict):
            continue
        bucket: Dict[str, float] = {}
        for prefix in model_keys:
            match_key = next((k for k in metrics.keys() if k.startswith(prefix)), None)
            if match_key is None:
                continue
            top1 = _parse_scalar_metric(metrics[match_key].get("top1"))
            mrr = _parse_scalar_metric(metrics[match_key].get("mrr"))
            if top1 is not None:
                bucket[f"{prefix}::top1"] = top1
            if mrr is not None:
                bucket[f"{prefix}::mrr"] = mrr
        out[f"N{int(n)}"] = bucket
    return out


def _extract_legacy_snapshot(blob: Dict) -> Dict[str, Dict[str, float]]:
    root = blob.get("benchmark_official")
    if root is None:
        root = blob.get("benchmark_top1_mrr_latency")
    if not isinstance(root, dict):
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for n_key, models in root.items():
        if not isinstance(models, dict):
            continue
        bucket: Dict[str, float] = {}
        for model_name, vals in models.items():
            if not isinstance(vals, dict):
                continue
            top1 = _parse_scalar_metric(vals.get("top1"))
            mrr = _parse_scalar_metric(vals.get("mrr"))
            if top1 is not None:
                bucket[f"{model_name}::top1"] = top1
            if mrr is not None:
                bucket[f"{model_name}::mrr"] = mrr
        out[str(n_key)] = bucket
    return out


def _normalized_key(raw_key: str) -> str:
    if raw_key.startswith("STA Select-then-Analyze"):
        return "STA Select-then-Analyze"
    if raw_key.startswith("Late Bottleneck STGCN+GIB"):
        return "Late Bottleneck STGCN+GIB"
    if raw_key.startswith("Late Bottleneck GATv2"):
        return "Late Bottleneck GATv2"
    if raw_key.startswith("Full-Graph GCN"):
        return "Full-Graph GCN"
    return raw_key


def _normalize_snapshot(snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for n_key, bucket in snapshot.items():
        n_out: Dict[str, float] = {}
        for full_key, val in bucket.items():
            if "::" not in full_key:
                continue
            model, metric = full_key.split("::", 1)
            n_out[f"{_normalized_key(model)}::{metric}"] = float(val)
        out[str(n_key)] = n_out
    return out


def _build_reconciliation_rows(
    canonical_snapshot: Dict[str, Dict[str, float]],
    legacy_snapshot: Dict[str, Dict[str, float]],
    source_path: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    c_norm = _normalize_snapshot(canonical_snapshot)
    l_norm = _normalize_snapshot(legacy_snapshot)
    for n_key in sorted(set(c_norm) & set(l_norm)):
        c_bucket = c_norm[n_key]
        l_bucket = l_norm[n_key]
        for key in sorted(set(c_bucket) & set(l_bucket)):
            model, metric = key.split("::", 1)
            c_val = float(c_bucket[key])
            l_val = float(l_bucket[key])
            rows.append(
                {
                    "source": source_path,
                    "N": n_key,
                    "model": model,
                    "metric": metric,
                    "canonical": c_val,
                    "legacy": l_val,
                    "delta_canonical_minus_legacy": c_val - l_val,
                }
            )
    return rows


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def build_manifest(repo_root: Path, artifacts: Optional[List[ArtifactSpec]] = None) -> Dict[str, object]:
    if artifacts is None:
        artifacts = [
            ArtifactSpec(
                "rca_pipeline/benchmark_full_prepublish.json",
                "python -m rca_pipeline.benchmark --output-json rca_pipeline/benchmark_full_prepublish.json",
            ),
            ArtifactSpec(
                "rca_pipeline/benchmark_real_long_prepublish.json",
                "python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7 --epochs 100 --max-cases 90 --sta-contrastive-warmup 0 --output-json rca_pipeline/benchmark_real_long_prepublish.json",
            ),
            ArtifactSpec(
                "rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
                "python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7 --epochs 100 --max-cases 90 --sta-contrastive-warmup 0 --use-mm --mm-strict --output-json rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
            ),
            ArtifactSpec(
                "rca_pipeline/ablation_prepublish.json",
                "python -m rca_pipeline.benchmark_ablation --output-json rca_pipeline/ablation_prepublish.json",
            ),
            ArtifactSpec(
                "results_transmission/warmup_real_trainticket_w0.json",
                "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42,123,7 --epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 0 --output-json results_transmission/warmup_real_trainticket_w0.json",
            ),
            ArtifactSpec(
                "results_transmission/warmup_real_trainticket_w10.json",
                "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42,123,7 --epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 10 --allow-real-warmup --output-json results_transmission/warmup_real_trainticket_w10.json",
            ),
            ArtifactSpec(
                "results_transmission/smoke_pre_submission_check.json",
                "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42 --epochs 1 --max-cases 10 --sta-contrastive-warmup 0 --output-json results_transmission/smoke_pre_submission_check.json",
            ),
            ArtifactSpec(
                "results_transmission/comparability_matrix.json",
                "python -m rca_pipeline.comparability_matrix --repo-root . --output-md docs/COMPARABILITY_MATRIX.md --output-json results_transmission/comparability_matrix.json",
            ),
            ArtifactSpec(
                "docs/COMPARABILITY_MATRIX.md",
                "python -m rca_pipeline.comparability_matrix --repo-root . --output-md docs/COMPARABILITY_MATRIX.md --output-json results_transmission/comparability_matrix.json",
            ),
            ArtifactSpec(
                "rca_pipeline/baseline_official.json",
                "legacy archive reference only",
                optional=True,
            ),
            ArtifactSpec(
                "rca_pipeline/baseline_official_compact.json",
                "legacy archive reference only",
                optional=True,
            ),
        ]

    artifact_rows: List[Dict[str, object]] = []
    canonical_snapshot: Dict[str, Dict[str, float]] = {}
    for spec in artifacts:
        p = repo_root / spec.path
        exists = p.exists()
        row: Dict[str, object] = {
            "path": spec.path,
            "optional": bool(spec.optional),
            "exists": bool(exists),
            "generation_command": spec.generation_command,
        }
        if exists:
            row["size_bytes"] = int(p.stat().st_size)
            row["sha256"] = _sha256_file(p)
        artifact_rows.append(row)

        if spec.path == "rca_pipeline/benchmark_full_prepublish.json" and exists:
            canonical_snapshot = _extract_simulated_snapshot(json.loads(p.read_text(encoding="utf-8")))

    reconciliation_rows: List[Dict[str, object]] = []
    for legacy_path in ["rca_pipeline/baseline_official.json", "rca_pipeline/baseline_official_compact.json"]:
        p = repo_root / legacy_path
        if not p.exists():
            continue
        legacy_snapshot = _extract_legacy_snapshot(json.loads(p.read_text(encoding="utf-8")))
        reconciliation_rows.extend(
            _build_reconciliation_rows(canonical_snapshot, legacy_snapshot, source_path=legacy_path)
        )

    return {
        "manifest_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source": "rca_pipeline/benchmark_full_prepublish.json",
        "git": {
            "commit": _git_commit(repo_root),
            "dirty_worktree": _git_dirty(repo_root),
        },
        "artifacts": artifact_rows,
        "simulated_snapshot": canonical_snapshot,
        "reconciliation_table": reconciliation_rows,
        "notes": [
            "Use this manifest as the single source of truth for soutenance metrics.",
            "If reconciliation_table is non-empty, legacy and canonical values diverge and must be explained.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical benchmark manifest + checksums")
    parser.add_argument("--repo-root", type=str, default=".", help="Path to repository root")
    parser.add_argument(
        "--output",
        type=str,
        default="results_transmission/canonical_results_manifest.json",
        help="Output JSON path for canonical manifest",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(repo_root)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_hash = _sha256_file(output_path)
    sha_path = output_path.with_suffix(output_path.suffix + ".sha256")
    sha_path.write_text(f"{manifest_hash}  {output_path.name}\n", encoding="utf-8")

    print(f"Wrote manifest: {output_path}")
    print(f"Wrote checksum: {sha_path}")


if __name__ == "__main__":
    main()
