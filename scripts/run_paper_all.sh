#!/usr/bin/env bash
##############################################################################
# run_paper_all.sh - End-to-end publication-grade benchmark pipeline.
#
# Runs all benchmarks needed to reproduce the published numbers, in the
# canonical order, with identical hyperparameters across phases:
#
#   Phase 1   Simulated Barabasi-Albert (N=20,50,100)
#   Phase 2   RCAEval real, metrics-only (TrainTicket, SockShop, OnlineBoutique)
#   Phase 3   RCAEval real, multimodal strict (metrics + traces, sockshop excluded)
#   Phase 4   Ablation study (GIB, training protocol, k-sensitivity, warmup)
#   Phase 5A  ObservIT sock-shop, metrics-only       (cluster-derived)
#   Phase 5B  ObservIT sock-shop, multimodal + Loki  (cluster-derived)
#   Phase 6   Regenerate canonical manifest + SHA-256 + comparability matrix
#   Phase 7   Print summary tables
#
# Usage:
#   bash scripts/run_paper_all.sh [options]
#
# Options:
#   --skip-sim          Skip phase 1
#   --skip-real         Skip phase 2
#   --skip-mm           Skip phase 3
#   --skip-ablation     Skip phase 4
#   --skip-observit     Skip phase 5 (both A and B)
#   --skip-observit-mm  Skip phase 5B only (multimodal Loki run)
#   --observit-data DIR ObservIT CSV directory (default: datasets/sockshop_runs_full)
#   --loki-ns NS        Loki namespace (default: loki)
#   --loki-svc SVC      Loki service name (default: loki)
#   --no-loki-pf        Skip auto port-forwarding Loki (assume already up)
#   --dry-run           Print commands without executing
#   --seeds LIST        Override seeds (default: 42,123,7,1337,2025)
#   --help              Show this help
#
# Output:
#   results_transmission/paper_benchmark_sim.json
#   results_transmission/paper_benchmark_real.json
#   results_transmission/paper_benchmark_real_mm.json
#   results_transmission/paper_benchmark_ablation.json
#   results_transmission/paper_benchmark_observit.json
#   results_transmission/paper_benchmark_observit_mm.json
#   results_transmission/paper_benchmark_<timestamp>.log
#   results_transmission/paper_benchmark.sha256
#   results_transmission/canonical_results_manifest.json (regenerated)
#
# Runtime estimate (CPU only):
#   Phase 1 ~3.0 h    Phase 2 ~0.5 h    Phase 3 ~0.5 h
#   Phase 4 ~0.5 h    Phase 5 ~0.3 h    Phase 6-7 ~1 min
#   Total   ~5 h
##############################################################################

set -uo pipefail

# ── argv parsing ────────────────────────────────────────────────────────────
SKIP_SIM=0; SKIP_REAL=0; SKIP_MM=0; SKIP_ABLATION=0
SKIP_OBSERVIT=0; SKIP_OBSERVIT_MM=0
OBSERVIT_DATA="datasets/sockshop_runs_full"
LOKI_NS="loki"
LOKI_SVC="loki"
NO_LOKI_PF=0
DRY_RUN=0
SEEDS="42,123,7,1337,2025"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sim)        SKIP_SIM=1 ;;
    --skip-real)       SKIP_REAL=1 ;;
    --skip-mm)         SKIP_MM=1 ;;
    --skip-ablation)   SKIP_ABLATION=1 ;;
    --skip-observit)   SKIP_OBSERVIT=1 ;;
    --skip-observit-mm) SKIP_OBSERVIT_MM=1 ;;
    --observit-data)   OBSERVIT_DATA="$2"; shift ;;
    --loki-ns)         LOKI_NS="$2"; shift ;;
    --loki-svc)        LOKI_SVC="$2"; shift ;;
    --no-loki-pf)      NO_LOKI_PF=1 ;;
    --dry-run)         DRY_RUN=1 ;;
    --seeds)           SEEDS="$2"; shift ;;
    --help|-h)         sed -n '3,40p' "$0"; exit 0 ;;
    *)                 echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# ── paths and constants ─────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

OUT_DIR="results_transmission"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$OUT_DIR/paper_benchmark_${TS}.log"

SIM_OUT="$OUT_DIR/paper_benchmark_sim.json"
REAL_OUT="$OUT_DIR/paper_benchmark_real.json"
MM_OUT="$OUT_DIR/paper_benchmark_real_mm.json"
ABL_OUT="$OUT_DIR/paper_benchmark_ablation.json"
OBS_OUT="$OUT_DIR/paper_benchmark_observit.json"
OBSMM_OUT="$OUT_DIR/paper_benchmark_observit_mm.json"
SHA_OUT="$OUT_DIR/paper_benchmark.sha256"

# Hyperparameters (must stay aligned across phases for paper-grade comparability)
SIM_SUP_EPOCHS=150
SIM_CONTRASTIVE_EPOCHS=80
SIM_WARMUP_EPOCHS=20
SIM_ANNEAL_EPOCHS=60
SIM_BASELINE_EPOCHS=150
REAL_EPOCHS=100
ABL_EPOCHS=100
OBS_EPOCHS=100
OBS_WINDOW=12

# State
LOKI_PF_PID=""
START_GLOBAL=$SECONDS
PHASE_FAILS=()

mkdir -p "$OUT_DIR"

# ── helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" | tee -a "$LOG" >&2; }
hdr()  { log ""; log "============================================================"; log "$*"; log "============================================================"; }

cleanup() {
  if [[ -n "$LOKI_PF_PID" ]] && kill -0 "$LOKI_PF_PID" 2>/dev/null; then
    log "Cleanup: stopping Loki port-forward (PID=$LOKI_PF_PID)"
    kill "$LOKI_PF_PID" 2>/dev/null || true
    wait "$LOKI_PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run() {
  log "CMD: $*"
  if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
  "$@" 2>&1 | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

start_loki_pf() {
  if [[ $NO_LOKI_PF -eq 1 ]]; then
    log "Loki: --no-loki-pf set, skipping port-forward"
    return 0
  fi
  log "Loki: starting port-forward $LOKI_NS/$LOKI_SVC -> 127.0.0.1:3100"
  if ! command -v kubectl >/dev/null 2>&1; then
    err "kubectl not in PATH; install or pass --no-loki-pf"
    return 1
  fi
  kubectl port-forward -n "$LOKI_NS" "svc/$LOKI_SVC" 3100:3100 \
      --address=127.0.0.1 >/tmp/loki_pf_${TS}.log 2>&1 &
  LOKI_PF_PID=$!
  # Wait until Loki is ready (max 30s)
  local i=0
  while (( i < 30 )); do
    if curl -fsS http://127.0.0.1:3100/ready >/dev/null 2>&1; then
      log "Loki: ready (PID=$LOKI_PF_PID)"
      return 0
    fi
    sleep 1; ((i++))
  done
  err "Loki port-forward did not become ready within 30s"
  return 1
}

stop_loki_pf() {
  if [[ -n "$LOKI_PF_PID" ]] && kill -0 "$LOKI_PF_PID" 2>/dev/null; then
    kill "$LOKI_PF_PID" 2>/dev/null || true
    wait "$LOKI_PF_PID" 2>/dev/null || true
    log "Loki: port-forward stopped"
    LOKI_PF_PID=""
  fi
}

# ── pre-flight checks ───────────────────────────────────────────────────────
hdr "PRE-FLIGHT CHECKS"

# Python + deps
if ! python -c "import torch, torch_geometric, numpy, pandas, sklearn" 2>/dev/null; then
  err "Python deps missing. Run: pip install -e \".[dev]\""
  exit 1
fi
log "Python deps OK"
log "  python: $(python --version 2>&1)"
log "  torch:  $(python -c 'import torch;print(torch.__version__)')"
log "  pyg:    $(python -c 'import torch_geometric;print(torch_geometric.__version__)')"

# Lint + tests guard (catch broken state before 5 hours of compute)
hdr "GUARD: lint + tests"
if ! run python -m ruff check rca_pipeline; then
  err "Ruff failed - aborting (fix lint before paper run)"
  exit 1
fi
if ! run python -m pytest -q rca_pipeline/tests; then
  err "Tests failed - aborting"
  exit 1
fi

# RCAEval data presence (needed for phases 2, 3, 4)
if [[ $SKIP_REAL -eq 0 || $SKIP_MM -eq 0 || $SKIP_ABLATION -eq 0 ]]; then
  if [[ ! -d "rcaeval/data" ]]; then
    err "rcaeval/data/ missing. Phases 2/3/4 require RCAEval data."
    err "Either skip them (--skip-real --skip-mm --skip-ablation) or install RCAEval data."
    exit 1
  fi
  log "RCAEval data: rcaeval/data/ present"
fi

# ObservIT data + cluster presence (needed for phase 5)
if [[ $SKIP_OBSERVIT -eq 0 ]]; then
  if [[ ! -d "$OBSERVIT_DATA" ]]; then
    err "ObservIT dataset directory missing: $OBSERVIT_DATA"
    err "Either skip (--skip-observit) or override (--observit-data DIR)"
    exit 1
  fi
  csv_count=$(find "$OBSERVIT_DATA" -maxdepth 1 -name 'dataset_*.csv' 2>/dev/null | wc -l)
  if [[ $csv_count -lt 1 ]]; then
    err "No dataset_*.csv files in $OBSERVIT_DATA"
    exit 1
  fi
  log "ObservIT data: $csv_count CSV(s) found in $OBSERVIT_DATA"

  if [[ $SKIP_OBSERVIT_MM -eq 0 && $NO_LOKI_PF -eq 0 ]]; then
    if ! command -v kubectl >/dev/null 2>&1; then
      err "kubectl required for phase 5B (Loki port-forward). Use --no-loki-pf or --skip-observit-mm."
      exit 1
    fi
    if ! kubectl config current-context >/dev/null 2>&1; then
      err "kubectl context not set. Use --skip-observit-mm or --no-loki-pf or fix kubeconfig."
      exit 1
    fi
    log "kubectl context: $(kubectl config current-context)"
  fi
fi

# Display run plan
hdr "RUN PLAN"
log "Output dir   : $OUT_DIR/"
log "Log file     : $LOG"
log "Seeds        : $SEEDS"
log "Sim epochs   : sup=$SIM_SUP_EPOCHS contrastive=$SIM_CONTRASTIVE_EPOCHS warmup=$SIM_WARMUP_EPOCHS anneal=$SIM_ANNEAL_EPOCHS baseline=$SIM_BASELINE_EPOCHS"
log "Real epochs  : $REAL_EPOCHS"
log "Phases       : sim=$([[ $SKIP_SIM -eq 0 ]] && echo Y || echo N)  real=$([[ $SKIP_REAL -eq 0 ]] && echo Y || echo N)  mm=$([[ $SKIP_MM -eq 0 ]] && echo Y || echo N)  ablation=$([[ $SKIP_ABLATION -eq 0 ]] && echo Y || echo N)  observit=$([[ $SKIP_OBSERVIT -eq 0 ]] && echo Y || echo N)  observit-mm=$([[ $SKIP_OBSERVIT -eq 0 && $SKIP_OBSERVIT_MM -eq 0 ]] && echo Y || echo N)"
log "Deterministic: yes (all phases)"
log "STA warmup   : 0 (real-data guardrail; see README)"
log "Drift filter : ADWIN enabled (delta=0.002, cooldown=30)"

# ── PHASE 1: simulated ──────────────────────────────────────────────────────
if [[ $SKIP_SIM -eq 0 ]]; then
  hdr "PHASE 1/7: Simulated benchmark (BA N=20,50,100 × 5 seeds)"
  T0=$SECONDS
  if run python -m rca_pipeline.benchmark \
        --sizes 20,50,100 \
        --seeds "$SEEDS" \
        --num-samples 3000 \
        --supervised-epochs       $SIM_SUP_EPOCHS \
        --late-contrastive-epochs $SIM_CONTRASTIVE_EPOCHS \
        --sta-warmup-epochs       $SIM_WARMUP_EPOCHS \
        --sta-k-anneal-epochs     $SIM_ANNEAL_EPOCHS \
        --baseline-epochs         $SIM_BASELINE_EPOCHS \
        --deterministic \
        --output-json "$SIM_OUT"; then
    log "Phase 1 done in $((SECONDS - T0))s → $SIM_OUT"
  else
    err "Phase 1 FAILED"; PHASE_FAILS+=("phase1-sim")
  fi
else
  log "Phase 1 skipped"
fi

# ── PHASE 2: RCAEval metrics-only ───────────────────────────────────────────
if [[ $SKIP_REAL -eq 0 ]]; then
  hdr "PHASE 2/7: RCAEval real - metrics-only (5 seeds)"
  T0=$SECONDS
  if run python -m rca_pipeline.benchmark_real \
        --dataset all \
        --seeds "$SEEDS" \
        --epochs $REAL_EPOCHS \
        --sta-contrastive-warmup 0 \
        --enable-drift \
        --deterministic \
        --output-json "$REAL_OUT"; then
    log "Phase 2 done in $((SECONDS - T0))s → $REAL_OUT"
  else
    err "Phase 2 FAILED"; PHASE_FAILS+=("phase2-real")
  fi
else
  log "Phase 2 skipped"
fi

# ── PHASE 3: RCAEval multimodal strict ──────────────────────────────────────
if [[ $SKIP_MM -eq 0 ]]; then
  hdr "PHASE 3/7: RCAEval real - multimodal strict (5 seeds)"
  T0=$SECONDS
  if run python -m rca_pipeline.benchmark_real \
        --dataset all \
        --seeds "$SEEDS" \
        --epochs $REAL_EPOCHS \
        --sta-contrastive-warmup 0 \
        --enable-drift \
        --deterministic \
        --use-mm --mm-strict \
        --output-json "$MM_OUT"; then
    log "Phase 3 done in $((SECONDS - T0))s → $MM_OUT"
  else
    err "Phase 3 FAILED"; PHASE_FAILS+=("phase3-mm")
  fi
else
  log "Phase 3 skipped"
fi

# ── PHASE 4: ablation ───────────────────────────────────────────────────────
if [[ $SKIP_ABLATION -eq 0 ]]; then
  hdr "PHASE 4/7: Ablation study (GIB, training protocol, k-sensitivity)"
  T0=$SECONDS
  if run python -m rca_pipeline.benchmark_ablation \
        --output-json "$ABL_OUT"; then
    log "Phase 4 done in $((SECONDS - T0))s → $ABL_OUT"
  else
    err "Phase 4 FAILED"; PHASE_FAILS+=("phase4-ablation")
  fi
else
  log "Phase 4 skipped"
fi

# ── PHASE 5A: ObservIT metrics-only ─────────────────────────────────────────
if [[ $SKIP_OBSERVIT -eq 0 ]]; then
  hdr "PHASE 5A/7: ObservIT sock-shop - metrics-only"
  T0=$SECONDS
  if run python -m rca_pipeline.benchmark_real \
        --observit-csv      "$OBSERVIT_DATA" \
        --observit-window   $OBS_WINDOW \
        --observit-topology sock_shop \
        --seeds             "$SEEDS" \
        --epochs            $OBS_EPOCHS \
        --sta-contrastive-warmup 0 \
        --enable-drift \
        --deterministic \
        --include-late-gib \
        --output-json "$OBS_OUT"; then
    log "Phase 5A done in $((SECONDS - T0))s → $OBS_OUT"
  else
    err "Phase 5A FAILED"; PHASE_FAILS+=("phase5a-observit")
  fi

  # ── PHASE 5B: ObservIT multimodal (Loki) ─────────────────────────────────
  if [[ $SKIP_OBSERVIT_MM -eq 0 ]]; then
    hdr "PHASE 5B/7: ObservIT sock-shop - multimodal (metrics + Loki logs)"
    T0=$SECONDS
    if start_loki_pf; then
      if run python -m rca_pipeline.benchmark_real \
            --observit-csv      "$OBSERVIT_DATA" \
            --observit-window   $OBS_WINDOW \
            --observit-topology sock_shop \
            --seeds             "$SEEDS" \
            --epochs            $OBS_EPOCHS \
            --sta-contrastive-warmup 0 \
            --enable-drift \
            --deterministic \
            --include-late-gib \
            --use-mm --mm-logs-only \
            --loki-url       http://127.0.0.1:3100 \
            --loki-namespace sock-shop \
            --output-json "$OBSMM_OUT"; then
        log "Phase 5B done in $((SECONDS - T0))s → $OBSMM_OUT"
      else
        err "Phase 5B FAILED"; PHASE_FAILS+=("phase5b-observit-mm")
      fi
      stop_loki_pf
    else
      err "Phase 5B FAILED (loki port-forward)"; PHASE_FAILS+=("phase5b-loki-pf")
    fi
  else
    log "Phase 5B skipped (--skip-observit-mm)"
  fi
else
  log "Phase 5 skipped"
fi

# ── PHASE 6: regenerate canonical artifacts ─────────────────────────────────
hdr "PHASE 6/7: Regenerating canonical manifest + SHA-256 + comparability matrix"

if [[ $DRY_RUN -eq 0 ]]; then
  # SHA-256 over all paper outputs that exist
  : > "$SHA_OUT"
  for f in "$SIM_OUT" "$REAL_OUT" "$MM_OUT" "$ABL_OUT" "$OBS_OUT" "$OBSMM_OUT"; do
    [[ -f "$f" ]] && sha256sum "$f" >> "$SHA_OUT"
  done
  log "SHA-256 written: $SHA_OUT"
  cat "$SHA_OUT" | tee -a "$LOG"

  # Regenerate canonical manifest
  if run python -m rca_pipeline.results_manifest \
        --repo-root . \
        --output "$OUT_DIR/canonical_results_manifest.json"; then
    log "Canonical manifest regenerated"
    sha256sum "$OUT_DIR/canonical_results_manifest.json" \
      > "$OUT_DIR/canonical_results_manifest.json.sha256"
  else
    err "Canonical manifest regeneration FAILED"
  fi

  # Regenerate comparability matrix
  if run python -m rca_pipeline.comparability_matrix \
        --repo-root . \
        --output-md docs/COMPARABILITY_MATRIX.md \
        --output-json "$OUT_DIR/comparability_matrix.json"; then
    log "Comparability matrix regenerated"
  else
    err "Comparability matrix regeneration FAILED"
  fi
fi

# ── PHASE 7: summary ────────────────────────────────────────────────────────
hdr "PHASE 7/7: Summary"

if [[ $DRY_RUN -eq 0 ]]; then
  python - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

OUTS = {
    "Phase 1 - Simulated":         "results_transmission/paper_benchmark_sim.json",
    "Phase 2 - Real metrics-only": "results_transmission/paper_benchmark_real.json",
    "Phase 3 - Real MM strict":    "results_transmission/paper_benchmark_real_mm.json",
    "Phase 4 - Ablation":          "results_transmission/paper_benchmark_ablation.json",
    "Phase 5A - ObservIT":         "results_transmission/paper_benchmark_observit.json",
    "Phase 5B - ObservIT MM":      "results_transmission/paper_benchmark_observit_mm.json",
}

for label, path in OUTS.items():
    p = Path(path)
    if not p.exists():
        print(f"\n── {label} ── (MISSING)")
        continue
    print(f"\n── {label} ──")
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f"  parse error: {e}")
        continue

    # Simulated layout
    if isinstance(d, dict) and "runs" in d:
        for run in d["runs"]:
            N = run.get("N", "?")
            best_t1, best_model, mlp_t1 = 0.0, "", 0.0
            for k, v in run.get("metrics", {}).items():
                if isinstance(v, dict) and "top1" in v:
                    if v["top1"] > best_t1:
                        best_t1, best_model = v["top1"], k
                    if "MLP" in k:
                        mlp_t1 = v["top1"]
            print(f"  N={N:3} best={best_model[:40]:<40} Top-1={best_t1:.4f}  MLP={mlp_t1:.4f}")
        continue

    # Ablation layout (A_N20, B_N20, ...)
    if any(k.startswith(("A_", "B_", "C", "D")) for k in d.keys() if isinstance(k, str)):
        for sec, secv in d.items():
            if not isinstance(secv, dict):
                continue
            for k, v in secv.items():
                if isinstance(v, dict) and "top1" in v:
                    print(f"  {sec:8} {k[:50]:<50} Top-1={v['top1']:.4f}")
        continue

    # Real / observit layout
    for ds, dv in d.items():
        if not isinstance(dv, dict):
            continue
        for m in ("STA","GATv2","STA-MM","GATv2-MM","Late-GIB","STA_drift_filtered","GATv2_drift_filtered"):
            if m in dv and isinstance(dv[m], dict) and "avg1" in dv[m]:
                avg1 = dv[m]["avg1"]
                std  = dv[m].get("avg1_std", 0.0)
                mrr  = dv[m].get("mrr", 0.0)
                ci   = dv[m].get("ci95", {}).get("avg1", {})
                n    = ci.get("n", "?")
                rel  = ci.get("seed_level_bootstrap_reliable", "?")
                print(f"  {ds:18} {m:22}  Avg@1={avg1:.4f}±{std:.4f}  MRR={mrr:.4f}  n={n} reliable={rel}")
PYEOF
fi

# ── final status ────────────────────────────────────────────────────────────
hdr "DONE"
log "Total runtime: $((SECONDS - START_GLOBAL))s"

if [[ ${#PHASE_FAILS[@]} -eq 0 ]]; then
  log "All requested phases SUCCEEDED."
  log ""
  log "Next step (commit publication artifacts):"
  log "  git add results_transmission/paper_benchmark_*.json \\"
  log "          results_transmission/paper_benchmark.sha256 \\"
  log "          results_transmission/canonical_results_manifest.json \\"
  log "          results_transmission/canonical_results_manifest.json.sha256 \\"
  log "          results_transmission/comparability_matrix.json \\"
  log "          docs/COMPARABILITY_MATRIX.md"
  log "  git commit -m \"Add publication-grade benchmark results (5 seeds, deterministic)\""
  exit 0
else
  err "FAILED PHASES: ${PHASE_FAILS[*]}"
  err "Inspect log: $LOG"
  exit 1
fi
