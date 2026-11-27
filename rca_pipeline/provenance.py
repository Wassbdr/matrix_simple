"""Results manifest v2: SHA-256 for every committed result artifact, with provenance.

    python -m rca_pipeline.provenance build    # rewrite results_transmission/canonical_results_manifest.json
    python -m rca_pipeline.provenance verify   # exit 1 on any hash mismatch or missing file (make verify-manifest)

Each artifact has a ``status``:

* ``regenerated``  - rebuilt on CPU from the clean commit in ``source_commit``
* ``generated``    - produced by code added in this branch (command in ``generation_command``)
* ``unchanged``    - same bytes as declared in the historical v1 manifest
* ``edited``       - committed file whose bytes changed after the v1 manifest; the v1
                     bytes were found in git history and restored under ``historical/``
* ``historical``   - frozen copy kept only so the v1 manifest can be checked

Artifacts that cannot be regenerated here are listed in ``missing`` with the reason;
they have no hash and are never reported as present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

MANIFEST = "results_transmission/canonical_results_manifest.json"
HISTORICAL_V1 = "results_transmission/historical/canonical_results_manifest_v1.json"
REGEN_COMMIT = "07b7bc1"  # main at branch point, clean worktree

ARTIFACTS: List[Dict[str, str]] = [
    {
        "path": "results/scaling_benchmark.json",
        "status": "generated",
        "generation_command": "make scaling (three invocations, listed in `invocations`)",
    },
    {
        "path": "results/sockshop_demo_eval.json",
        "status": "generated",
        "generation_command": "python -m rca_pipeline.demo_incident build",
    },
    {
        "path": "examples/sockshop_demo/incident.json",
        "status": "generated",
        "generation_command": "python -m rca_pipeline.demo_incident build",
    },
    {
        "path": "examples/sockshop_demo/bundle/model.pt",
        "status": "generated",
        "generation_command": "python -m rca_pipeline.demo_incident build",
    },
    {
        "path": "examples/sockshop_demo/bundle/scaler.pt",
        "status": "generated",
        "generation_command": "python -m rca_pipeline.demo_incident build",
    },
    {
        "path": "examples/sockshop_demo/bundle/metadata.json",
        "status": "generated",
        "generation_command": "python -m rca_pipeline.demo_incident build",
    },
    {
        "path": "results/regenerated/benchmark_full.json",
        "status": "regenerated",
        "replaces": "rca_pipeline/benchmark_full_prepublish.json",
        "source_commit": REGEN_COMMIT,
        "generation_command": "git worktree add ../regen 07b7bc1 && cd ../regen && "
        "python -m rca_pipeline.benchmark --output-json rca_pipeline/benchmark_full_prepublish.json",
    },
    {
        "path": "results/regenerated/ablation.json",
        "status": "regenerated",
        "replaces": "rca_pipeline/ablation_prepublish.json",
        "source_commit": REGEN_COMMIT,
        "generation_command": "git worktree add ../regen 07b7bc1 && cd ../regen && "
        "python -m rca_pipeline.benchmark_ablation --output-json rca_pipeline/ablation_prepublish.json",
    },
    {
        "path": "results_transmission/warmup_real_trainticket_w0.json",
        "status": "unchanged",
        "generation_command": "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42,123,7 "
        "--epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 0 "
        "--output-json results_transmission/warmup_real_trainticket_w0.json",
    },
    {
        "path": "results_transmission/warmup_real_trainticket_w10.json",
        "status": "unchanged",
        "generation_command": "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42,123,7 "
        "--epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 10 --allow-real-warmup "
        "--output-json results_transmission/warmup_real_trainticket_w10.json",
    },
    {
        "path": "results_transmission/smoke_pre_submission_check.json",
        "status": "unchanged",
        "generation_command": "python -m rca_pipeline.benchmark_real --dataset trainticket --seeds 42 "
        "--epochs 1 --max-cases 10 --sta-contrastive-warmup 0 "
        "--output-json results_transmission/smoke_pre_submission_check.json",
    },
    {
        "path": "results_transmission/comparability_matrix.json",
        "status": "edited",
        "notes": "edited in d6ff3b8 (isoperimetric BARO) after the v1 manifest; "
        "v1 bytes restored from 00d220c under results_transmission/historical/",
        "generation_command": "python -m rca_pipeline.comparability_matrix --repo-root . "
        "--output-md docs/COMPARABILITY_MATRIX.md --output-json results_transmission/comparability_matrix.json "
        "(needs the RCAEval prepublish runs listed in `missing`)",
    },
    {
        "path": "docs/COMPARABILITY_MATRIX.md",
        "status": "edited",
        "notes": "edited in d6ff3b8 and 55ea6a5 after the v1 manifest; "
        "v1 bytes restored from 00d220c under results_transmission/historical/",
        "generation_command": "see results_transmission/comparability_matrix.json",
    },
    {
        "path": "results_transmission/historical/comparability_matrix_00d220c.json",
        "status": "historical",
        "generation_command": "git show 00d220c:results_transmission/comparability_matrix.json",
    },
    {
        "path": "results_transmission/historical/COMPARABILITY_MATRIX_00d220c.md",
        "status": "historical",
        "generation_command": "git show 00d220c:docs/COMPARABILITY_MATRIX.md",
    },
    {
        "path": HISTORICAL_V1,
        "status": "historical",
        "generation_command": f"git show 07b7bc1:{MANIFEST}",
    },
]

MISSING: List[Dict[str, str]] = [
    {
        "path": "rca_pipeline/benchmark_real_long_prepublish.json",
        "reason": "never committed (matched by .gitignore `rca_pipeline/benchmark*.json`), absent from "
        "every branch and from `git log --all`; regeneration needs the RCAEval RE2 dataset "
        "(external download, not redistributed here) and ~100 epochs x 3 seeds x 3 systems.",
        "generation_command": "python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7 "
        "--epochs 100 --max-cases 90 --sta-contrastive-warmup 0 "
        "--output-json rca_pipeline/benchmark_real_long_prepublish.json",
    },
    {
        "path": "rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
        "reason": "same as benchmark_real_long_prepublish.json (RCAEval data required).",
        "generation_command": "python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7 "
        "--epochs 100 --max-cases 90 --sta-contrastive-warmup 0 --use-mm --mm-strict "
        "--output-json rca_pipeline/benchmark_real_long_mm_strict_prepublish.json",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def build(repo: Path) -> Dict:
    rows = []
    for spec in ARTIFACTS:
        p = repo / spec["path"]
        if not p.exists():
            raise SystemExit(f"artifact missing, cannot build manifest: {spec['path']}")
        row = {**spec, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if p.suffix == ".json":
            invocations = json.loads(p.read_text(encoding="utf-8")).get("invocations")
            if isinstance(invocations, list) and invocations:
                row["invocations"] = invocations
        rows.append(row)
    return {
        "manifest_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "built_from_commit": _git(repo, "rev-parse", "HEAD") or None,
        "supersedes": HISTORICAL_V1,
        "hardware_for_regenerated_and_generated": "CPU only (no GPU); see each JSON's hardware block",
        "artifacts": rows,
        "missing": MISSING,
        "verify": "make verify-manifest",
    }


def verify(repo: Path, manifest_path: str = MANIFEST) -> int:
    mpath = repo / manifest_path
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    failures = 0
    sha_file = mpath.with_suffix(mpath.suffix + ".sha256")
    if sha_file.exists():
        expected = sha_file.read_text().split()[0]
        ok = expected == sha256_file(mpath)
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'}  {manifest_path} (self-checksum)")
    for row in manifest["artifacts"]:
        p = repo / row["path"]
        if not p.exists():
            print(f"BAD  {row['path']}: file missing")
            failures += 1
            continue
        actual = sha256_file(p)
        ok = actual == row["sha256"]
        failures += not ok
        print(
            f"{'OK ' if ok else 'BAD'}  {row['path']} [{row['status']}]"
            + ("" if ok else f"\n     expected {row['sha256']}\n     actual   {actual}")
        )
    for row in manifest.get("missing", []):
        print(f"--   {row['path']}: not present (declared missing)")
    print(
        f"\n{len(manifest['artifacts']) - failures}/{len(manifest['artifacts'])} artifacts match"
        + (f", {failures} failure(s)" if failures else "")
    )
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    if args.cmd == "verify":
        return verify(repo)
    manifest = build(repo)
    out = repo / MANIFEST
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (repo / (MANIFEST + ".sha256")).write_text(f"{sha256_file(out)}  {out.name}\n")
    print(f"wrote {MANIFEST} ({len(manifest['artifacts'])} artifacts, {len(MISSING)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
