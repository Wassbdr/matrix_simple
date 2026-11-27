#!/usr/bin/env python3
"""
validation_protocol.py - Fixed, repeatable validation runs for RCAEval.

Runs a small matrix of configs and writes JSON outputs under rca_pipeline/validation_runs/.
Then prints a compact summary table (Avg@1, MRR, NDCG@3) per dataset/method.

Sockshop: only ``metrics_only`` is executed; MM configs write a stub JSON (skipped) - see
docs/PAPER_BENCHMARKS.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "validation_runs"


@dataclass(frozen=True)
class RunCfg:
    name: str
    extra_args: List[str]


@dataclass(frozen=True)
class ProtocolPreset:
    name: str
    seeds: str
    epochs: str
    max_cases: str


PRESETS: Dict[str, ProtocolPreset] = {
    # Fast sanity check (quick iteration)
    "smoke": ProtocolPreset("smoke", "42,123", "20", "20"),
    # Reference comparison - 5 seeds for more stable CIs
    "long": ProtocolPreset("long", "42,123,7,1337,2025", "100", "0"),
    # Extended: 10 seeds for paper-grade statistical power
    "paper": ProtocolPreset("paper", "42,123,7,1337,2025,314,555,888,999,4242", "100", "0"),
}


RUNS: List[RunCfg] = [
    RunCfg("metrics_only", []),
    RunCfg("traces_only", ["--use-mm", "--mm-traces-only", "--mm-strict"]),
    RunCfg("trace_prior", ["--use-mm", "--mm-trace-prior", "--mm-strict"]),
    RunCfg("full_mm_strict", ["--use-mm", "--mm-strict"]),
]

# Multimodal runs are only defined for datasets with reliable logs/traces in our protocol.
# sockshop is excluded from MM (see docs/PAPER_BENCHMARKS.md).
MM_RUN_NAMES = frozenset({"traces_only", "trace_prior", "full_mm_strict"})


def _build_mm_skipped_stub(dataset_name: str) -> Dict:
    reason = (
        "Multimodal RCAEval runs are not part of the paper protocol for sockshop "
        "(trace coverage insufficient for strict MM); compare trainticket and "
        "online-boutique for MM."
    )
    return {
        dataset_name: {
            "_skipped": True,
            "_reason": reason,
            "config": {
                "use_mm": True,
                "mm_strict": True,
                "mm_dataset_status": "skipped_with_reason",
                "mm_dataset_reason": reason,
            },
        }
    }


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, cwd=str(ROOT.parent), check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _extract_metrics(blob: Dict, model_key: str) -> Optional[Tuple[float, float, float]]:
    if model_key not in blob:
        return None
    m = blob[model_key]
    return float(m.get("avg1", 0.0)), float(m.get("mrr", 0.0)), float(m.get("ndcg3", 0.0))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    datasets = ["trainticket", "online-boutique", "sockshop"]
    preset_name = os.environ.get("RCA_VAL_PRESET", "smoke").lower()
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown RCA_VAL_PRESET='{preset_name}'. Choose from: {list(PRESETS.keys())}")
    preset = PRESETS[preset_name]

    # Explicit env overrides remain available.
    seeds = os.environ.get("RCA_VAL_SEEDS", preset.seeds)
    epochs = os.environ.get("RCA_VAL_EPOCHS", preset.epochs)
    max_cases = os.environ.get("RCA_VAL_MAX_CASES", preset.max_cases)
    drift = os.environ.get("RCA_VAL_DRIFT", "1") == "1"
    sta_warmup = os.environ.get("RCA_VAL_STA_WARMUP", "0")

    print(
        f"[validation_protocol] preset={preset_name} seeds={seeds} epochs={epochs} "
        f"max_cases={max_cases} drift={int(drift)} sta_warmup={sta_warmup}"
    )

    results: Dict[str, Dict[str, Dict[str, Optional[Tuple[float, float, float]]]]] = {}

    for ds in datasets:
        results[ds] = {}
        for cfg in RUNS:
            out_json = OUT_DIR / f"{ds}__{cfg.name}.json"
            if ds == "sockshop" and cfg.name in MM_RUN_NAMES:
                stub = _build_mm_skipped_stub(ds)
                out_json.write_text(json.dumps(stub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(
                    f"\n=== SKIP {ds} :: {cfg.name} "
                    f"(MM not in scope for sockshop; stub written to {out_json.name}) ==="
                )
                results[ds][cfg.name] = {"STA": None, "GATv2": None}
                continue

            cmd = [
                sys.executable,
                "rca_pipeline/benchmark_real.py",
                "--dataset",
                ds,
                "--epochs",
                str(epochs),
                "--seeds",
                str(seeds),
                "--sta-contrastive-warmup",
                str(sta_warmup),
                "--output-json",
                str(out_json),
            ]
            if int(max_cases) > 0:
                cmd.extend(["--max-cases", str(max_cases)])
            if drift:
                cmd.append("--enable-drift")
            cmd.extend(cfg.extra_args)

            print(f"\n=== RUN {ds} :: {cfg.name} ===")
            _run(cmd)

            blob = _load_json(out_json).get(ds, {})
            sta_key = "STA-MM" if any(a == "--use-mm" for a in cfg.extra_args) else "STA"
            gat_key = "GATv2-MM" if any(a == "--use-mm" for a in cfg.extra_args) else "GATv2"
            results[ds][cfg.name] = {
                "STA": _extract_metrics(blob, sta_key),
                "GATv2": _extract_metrics(blob, gat_key),
            }

    print("\n\n==================== SUMMARY (Avg@1 / MRR / NDCG@3) ====================")
    for ds in datasets:
        print(f"\nDATASET: {ds}")
        for cfg in RUNS:
            r = results[ds].get(cfg.name, {})
            sta = r.get("STA")
            gat = r.get("GATv2")
            def fmt(x):
                if x is None:
                    return "n/a"
                return f"{x[0]:.3f}/{x[1]:.3f}/{x[2]:.3f}"
            print(f"  {cfg.name:14s}  STA={fmt(sta):18s}  GATv2={fmt(gat):18s}")


if __name__ == "__main__":
    main()
