#!/usr/bin/env python3
"""Build a protocol comparability matrix for soutenance artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rca_pipeline.metrics_rcaeval import RCAEVAL_PUBLISHED_BASELINES


@dataclass(frozen=True)
class RunRow:
    dataset: str
    mode: str
    seeds: str
    epochs: str
    max_cases: str
    split_mode: str
    drift: str
    normalized: str
    mm_status: str
    protocol_source: str


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_seeds(cfg: Dict) -> str:
    seeds = cfg.get("seeds")
    if isinstance(seeds, list):
        return ",".join(str(int(s)) for s in seeds)
    return "unknown"


def _infer_mm_status(ds_blob: Dict) -> str:
    cfg = ds_blob.get("config", {}) if isinstance(ds_blob, dict) else {}
    status = cfg.get("mm_dataset_status")
    if isinstance(status, str) and status:
        return status

    # Backward-compatible inference for older artifacts without mm_dataset_status.
    has_mm_model = any(k.endswith("-MM") for k in ds_blob.keys())
    if has_mm_model:
        return "comparable"
    if bool(cfg.get("use_mm")):
        return "skipped_with_reason"
    return "not_applicable"


def _extract_row(dataset: str, mode: str, ds_blob: Dict, defaults: Optional[Dict[str, str]] = None) -> RunRow:
    cfg = ds_blob.get("config", {}) if isinstance(ds_blob, dict) else {}
    split = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}

    seeds = _format_seeds(cfg)
    epochs = str(cfg.get("hyperparams", {}).get("epochs", "unknown"))
    max_cases_raw = cfg.get("max_cases", "unknown")
    max_cases = "all" if max_cases_raw is None else str(max_cases_raw)
    split_mode = str(split.get("mode", "unknown"))
    drift = "on" if bool(cfg.get("enable_drift", False)) else "off"
    normalized = "yes" if bool(cfg.get("normalized", False)) else "no"
    mm_status = _infer_mm_status(ds_blob)
    protocol_source = "artifact_config"

    if defaults is not None:
        if seeds == "unknown":
            seeds = defaults.get("seeds", seeds)
            protocol_source = "fallback_inferred"
        if epochs == "unknown":
            epochs = defaults.get("epochs", epochs)
            protocol_source = "fallback_inferred"
        if max_cases == "unknown":
            max_cases = defaults.get("max_cases", max_cases)
            protocol_source = "fallback_inferred"
        if split_mode == "unknown":
            split_mode = defaults.get("split_mode", split_mode)
            protocol_source = "fallback_inferred"

    return RunRow(
        dataset=dataset,
        mode=mode,
        seeds=seeds,
        epochs=epochs,
        max_cases=max_cases,
        split_mode=split_mode,
        drift=drift,
        normalized=normalized,
        mm_status=mm_status,
        protocol_source=protocol_source,
    )


def build_matrix(repo_root: Path) -> Dict[str, object]:
    long_metrics = _load_json(repo_root / "rca_pipeline/benchmark_real_long_prepublish.json") or {}
    long_mm = _load_json(repo_root / "rca_pipeline/benchmark_real_long_mm_strict_prepublish.json") or {}

    long_defaults = {
        "seeds": "42,123,7",
        "epochs": "100",
        "max_cases": "90",
        "split_mode": "varying_split_by_run_seed",
    }

    rows: List[RunRow] = []
    for ds_name, ds_blob in long_metrics.items():
        if not isinstance(ds_blob, dict) or ds_name.startswith("_"):
            continue
        rows.append(_extract_row(ds_name, "metrics_only", ds_blob, defaults=long_defaults))

    for ds_name, ds_blob in long_mm.items():
        if not isinstance(ds_blob, dict) or ds_name.startswith("_"):
            continue
        rows.append(_extract_row(ds_name, "full_mm_strict", ds_blob, defaults=long_defaults))

    baseline_rows = [
        {
            "method": method,
            "avg1": float(vals["avg1"]),
            "avg3": float(vals["avg3"]),
            "avg5": float(vals["avg5"]),
            "mrr": float(vals["mrr"]),
            "ndcg3": float(vals["ndcg3"]),
            "ndcg5": float(vals["ndcg5"]),
        }
        for method, vals in sorted(RCAEVAL_PUBLISHED_BASELINES.items())
    ]

    # Isoperimetric BARO: re-run on identical cases / candidate nodes / per-seed split /
    # metric functions (baselines_rcaeval_ourprotocol.json), produced by baseline_baro.py.
    iso = _load_json(repo_root / "results_transmission/baselines_rcaeval_ourprotocol.json") or {}
    iso_rows: List[Dict] = []
    for ds_name in ("trainticket", "sockshop", "online-boutique"):
        blob = iso.get(ds_name)
        if not isinstance(blob, dict):
            continue
        baro = blob.get("BARO_ourprotocol", {})
        sta = blob.get("STA_ours", {})
        gat = blob.get("GATv2_ours", {})
        baro1 = baro.get("avg1")
        ours_vals = [v for v in (sta.get("avg1"), gat.get("avg1")) if v is not None]
        best_ours = max(ours_vals) if ours_vals else None
        if baro1 is None or best_ours is None:
            verdict = "n/a"
        elif best_ours - baro1 > 0.03:
            verdict = "ours_better"
        elif best_ours - baro1 < -0.03:
            verdict = "baro_better"
        else:
            verdict = "within_noise"
        iso_rows.append({
            "dataset": ds_name,
            "baro_avg1": baro1,
            "baro_avg1_std": baro.get("avg1_std"),
            "baro_mrr": baro.get("mrr"),
            "sta_avg1": sta.get("avg1"),
            "gatv2_avg1": gat.get("avg1"),
            "verdict": verdict,
        })

    metrics_verdict = "comparable" if iso_rows else "partially_comparable"
    comparability_findings = [
        {
            "claim": "Ours vs BARO (metrics-only), isoperimetric (our protocol)",
            "evidence_in_repo": (
                "BARO re-run on identical cases, candidate node set, per-seed split and "
                "metric functions (results_transmission/baselines_rcaeval_ourprotocol.json). "
                "Under identical conditions BARO is far stronger than its published 0.71: "
                "ours wins on trainticket, ties on sockshop, loses on online-boutique - no "
                "uniform SOTA claim."
            ) if iso_rows else "Baseline values available, but no per-dataset split/preprocessing signatures.",
            "verdict": metrics_verdict,
            "required_for_full_isoperimeter": [] if iso_rows else [
                "Published baseline split protocol per dataset",
                "Published preprocessing/normalization details",
                "Case filtering policy alignment (max_cases, exclusions)",
            ],
        },
        {
            "claim": "Ours vs *published* RCAEval baseline numbers (BARO 0.71 etc.)",
            "evidence_in_repo": "Published constants have no per-dataset split/preprocessing signature; superseded by the isoperimetric row above.",
            "verdict": "partially_comparable",
            "required_for_full_isoperimeter": [
                "Published baseline split protocol per dataset",
                "Published preprocessing/normalization details",
            ],
        },
        {
            "claim": "Ours MM strict vs published baselines",
            "evidence_in_repo": "Published baselines in repo are not multimodal and have no MM protocol signature.",
            "verdict": "not_isoperimeter",
            "required_for_full_isoperimeter": [
                "Published multimodal baseline references with matching modality scope",
                "Dataset-level MM comparability status for all compared datasets",
            ],
        },
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "metrics_only": "rca_pipeline/benchmark_real_long_prepublish.json",
            "full_mm_strict": "rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
        },
        "protocol_rows": [row.__dict__ for row in rows],
        "published_baselines": baseline_rows,
        "isoperimetric_baro": iso_rows,
        "comparability_findings": comparability_findings,
    }


def render_markdown(matrix: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# COMPARABILITY MATRIX")
    lines.append("")
    lines.append(f"Generated at: {matrix['generated_at_utc']}")
    lines.append("")
    lines.append("## Protocol Signature (Ours)")
    lines.append("")
    lines.append("| Dataset | Mode | Seeds | Epochs | Max cases | Split mode | Drift | Normalized | MM status | Protocol source |")
    lines.append("|---|---|---|---:|---|---|---|---|---|---|")
    for row in matrix["protocol_rows"]:
        lines.append(
            "| {dataset} | {mode} | {seeds} | {epochs} | {max_cases} | {split_mode} | {drift} | {normalized} | {mm_status} | {protocol_source} |".format(
                **row
            )
        )

    lines.append("")
    lines.append("## Published Baselines Available In Repository")
    lines.append("")
    lines.append("| Method | Avg@1 | Avg@3 | Avg@5 | MRR | NDCG@3 | NDCG@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in matrix["published_baselines"]:
        lines.append(
            f"| {row['method']} | {row['avg1']:.2f} | {row['avg3']:.2f} | {row['avg5']:.2f} | {row['mrr']:.2f} | {row['ndcg3']:.2f} | {row['ndcg5']:.2f} |"
        )

    iso = matrix.get("isoperimetric_baro") or []
    if iso:
        lines.append("")
        lines.append("## Isoperimetric BARO (our protocol)")
        lines.append("")
        lines.append(
            "BARO re-run on the *same* cases, candidate node set, per-seed split and metric "
            "functions as our models (see `baseline_baro.py`). Avg@1 (%)."
        )
        lines.append("")
        lines.append("| Dataset | BARO (ours) | BARO (published) | STA (ours) | GATv2 (ours) | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---|")

        def _pct(v):
            return f"{v * 100:.1f}" if isinstance(v, (int, float)) else "n/a"

        pub_baro = next((b["avg1"] for b in matrix["published_baselines"] if b["method"] == "BARO"), None)
        for r in iso:
            lines.append(
                f"| {r['dataset']} | {_pct(r['baro_avg1'])} | {_pct(pub_baro)} | "
                f"{_pct(r['sta_avg1'])} | {_pct(r['gatv2_avg1'])} | {r['verdict']} |"
            )
        lines.append("")
        lines.append(
            "> Under identical conditions BARO is far stronger than its published 0.71. "
            "Our models win on trainticket, are within noise on sockshop, and **lose on "
            "online-boutique** - we make no uniform SOTA claim over BARO."
        )

    lines.append("")
    lines.append("## Isoperimeter Verdict")
    lines.append("")
    lines.append("| Claim | Evidence In Repo | Verdict | Missing For Full Isoperimeter |")
    lines.append("|---|---|---|---|")
    for finding in matrix["comparability_findings"]:
        missing = "; ".join(finding["required_for_full_isoperimeter"])
        lines.append(
            f"| {finding['claim']} | {finding['evidence_in_repo']} | {finding['verdict']} | {missing} |"
        )

    lines.append("")
    lines.append("## Decision Use")
    lines.append("")
    lines.append("- Use this matrix to qualify every baseline claim in slides/report.")
    lines.append("- A claim marked not_isoperimeter must be framed as indicative, not conclusive SOTA evidence.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparability matrix from benchmark artifacts")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root path")
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/COMPARABILITY_MATRIX.md",
        help="Markdown output path (repo-relative)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results_transmission/comparability_matrix.json",
        help="JSON output path (repo-relative)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_md = (repo_root / args.output_md).resolve()
    out_json = (repo_root / args.output_json).resolve()

    matrix = build_matrix(repo_root)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(matrix), encoding="utf-8")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
