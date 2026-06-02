#!/usr/bin/env bash
# run_paper_benchmark.sh — Publication-grade benchmark suite.
#
# Runs three benchmarks sequentially:
#   1. Simulated (Barabasi-Albert N=20/50/100, 5 seeds, deterministic)
#   2. RCAEval metrics-only (3 datasets, 5 seeds, deterministic)
#   3. RCAEval multimodal strict (3 datasets, 5 seeds, deterministic)
#
# Outputs to results_transmission/paper_*.json (tracked in git).
# Runtime estimate: 4–6 h on CPU.
#
# Usage:
#   bash scripts/run_paper_benchmark.sh [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate 2>/dev/null || true

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

OUT_DIR="results_transmission"
LOG_FILE="$OUT_DIR/paper_benchmark_$(date +%Y%m%d_%H%M%S).log"
SIM_OUT="$OUT_DIR/paper_benchmark_sim.json"
REAL_OUT="$OUT_DIR/paper_benchmark_real.json"
MM_OUT="$OUT_DIR/paper_benchmark_real_mm.json"

SEEDS="42,123,7,1337,2025"
SIM_SUP_EPOCHS=150
SIM_CONTRASTIVE_EPOCHS=80
SIM_WARMUP_EPOCHS=20
SIM_ANNEAL_EPOCHS=60
SIM_BASELINE_EPOCHS=150
REAL_EPOCHS=100

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=========================================="
log "STA RCA — Paper Benchmark Suite"
log "  seeds        : $SEEDS"
log "  sim epochs   : sup=$SIM_SUP_EPOCHS contrastive=$SIM_CONTRASTIVE_EPOCHS"
log "                 warmup=$SIM_WARMUP_EPOCHS anneal=$SIM_ANNEAL_EPOCHS"
log "  real epochs  : $REAL_EPOCHS"
log "  deterministic: yes"
log "  outputs      : $OUT_DIR/"
log "=========================================="

if [[ $DRY_RUN -eq 1 ]]; then
  log "DRY-RUN: commands printed but not executed."
fi

# ── 1. SIMULATED BENCHMARK ────────────────────────────────────────────────────
log ""
log "=== PHASE 1/3: Simulated benchmark (N=20,50,100 × 5 seeds) ==="
T0=$SECONDS

SIM_CMD=(
  python -m rca_pipeline.benchmark
  --sizes 20,50,100
  --seeds "$SEEDS"
  --num-samples 3000
  --supervised-epochs  $SIM_SUP_EPOCHS
  --late-contrastive-epochs $SIM_CONTRASTIVE_EPOCHS
  --sta-warmup-epochs  $SIM_WARMUP_EPOCHS
  --sta-k-anneal-epochs $SIM_ANNEAL_EPOCHS
  --baseline-epochs    $SIM_BASELINE_EPOCHS
  --deterministic
  --output-json "$SIM_OUT"
)

log "CMD: ${SIM_CMD[*]}"
if [[ $DRY_RUN -eq 0 ]]; then
  "${SIM_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "Phase 1 done in $(( SECONDS - T0 ))s → $SIM_OUT"
fi

# ── 2. REAL BENCHMARK (metrics-only) ─────────────────────────────────────────
log ""
log "=== PHASE 2/3: RCAEval real benchmark — metrics-only (5 seeds) ==="
T0=$SECONDS

REAL_CMD=(
  python -m rca_pipeline.benchmark_real
  --dataset all
  --seeds "$SEEDS"
  --epochs $REAL_EPOCHS
  --sta-contrastive-warmup 0
  --enable-drift
  --deterministic
  --output-json "$REAL_OUT"
)

log "CMD: ${REAL_CMD[*]}"
if [[ $DRY_RUN -eq 0 ]]; then
  "${REAL_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "Phase 2 done in $(( SECONDS - T0 ))s → $REAL_OUT"
fi

# ── 3. REAL BENCHMARK (multimodal strict) ────────────────────────────────────
log ""
log "=== PHASE 3/3: RCAEval real benchmark — multimodal strict (5 seeds) ==="
T0=$SECONDS

MM_CMD=(
  python -m rca_pipeline.benchmark_real
  --dataset all
  --seeds "$SEEDS"
  --epochs $REAL_EPOCHS
  --sta-contrastive-warmup 0
  --enable-drift
  --deterministic
  --use-mm --mm-strict
  --output-json "$MM_OUT"
)

log "CMD: ${MM_CMD[*]}"
if [[ $DRY_RUN -eq 0 ]]; then
  "${MM_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "Phase 3 done in $(( SECONDS - T0 ))s → $MM_OUT"
fi

# ── SUMMARY ──────────────────────────────────────────────────────────────────
log ""
log "=== ALL PHASES COMPLETE ==="
log ""

if [[ $DRY_RUN -eq 0 ]]; then
  python - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from pathlib import Path

files = {
    "Simulated":        "results_transmission/paper_benchmark_sim.json",
    "Real metrics-only":"results_transmission/paper_benchmark_real.json",
    "Real MM strict":   "results_transmission/paper_benchmark_real_mm.json",
}

for label, path in files.items():
    p = Path(path)
    if not p.exists():
        print(f"[MISSING] {label}: {path}")
        continue
    d = json.loads(p.read_text())
    print(f"\n── {label} ──")

    # Simulated: list per-N best Top-1
    if "runs" in d:
        for run in d["runs"]:
            N = run["N"]
            best_t1, best_model = 0.0, ""
            mlp_t1 = 0.0
            for k, v in run["metrics"].items():
                if isinstance(v, dict) and "top1" in v:
                    if v["top1"] > best_t1:
                        best_t1, best_model = v["top1"], k
                    if "MLP" in k:
                        mlp_t1 = v["top1"]
            print(f"  N={N:3d}: best={best_model[:35]:<35} Top-1={best_t1:.4f}  MLP={mlp_t1:.4f}")
    else:
        # Real: show Avg@1 and MRR per dataset/model
        for ds, dv in d.items():
            if not isinstance(dv, dict):
                continue
            models = [k for k in dv if k not in ("config", "drift_filter", "env", "deterministic")]
            for m in models:
                if isinstance(dv[m], dict) and "avg1" in dv[m]:
                    avg1 = dv[m]["avg1"]
                    std  = dv[m].get("avg1_std", 0)
                    mrr  = dv[m].get("mrr", 0)
                    print(f"  {ds:20s} {m:12s}  Avg@1={avg1:.4f}±{std:.4f}  MRR={mrr:.4f}")
PYEOF
fi

log ""
log "Log : $LOG_FILE"
log "Done."
