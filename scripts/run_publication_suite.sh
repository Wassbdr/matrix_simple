#!/usr/bin/env bash
# run_publication_suite.sh — Single entry point that produces every result
# cited in the paper, in publication-grade configuration.
#
# Phases (each idempotent, skipped if output already exists unless --force):
#   0. Pre-flight  : env check, deps, audit, ruff, pytest
#   1. Simulated   : Barabasi-Albert N=20/50/100, 5 seeds, deterministic
#   2. RCAEval     : real-world metrics-only, 3 datasets x 5 seeds
#   3. RCAEval MM  : multimodal strict, 3 datasets x 5 seeds (sockshop excluded)
#   4. ObservIT    : sock-shop cluster, metrics-only (requires kubectl + dataset)
#   5. ObservIT MM : sock-shop cluster + Loki logs (requires Loki port-forward)
#   6. Post        : SHA256 + canonical manifest + summary
#
# Runtime: ~5–6 h on CPU (skip phases 4–5 if cluster unavailable: ~4 h).
#
# Usage:
#   bash scripts/run_publication_suite.sh                  # full run
#   bash scripts/run_publication_suite.sh --dry-run        # print commands only
#   bash scripts/run_publication_suite.sh --skip-observit  # local-only (no cluster)
#   bash scripts/run_publication_suite.sh --force          # rerun even if outputs exist
#   bash scripts/run_publication_suite.sh --only 1,2       # run subset of phases

set -uo pipefail

# ── CLI parsing ──────────────────────────────────────────────────────────────
DRY_RUN=0
FORCE=0
SKIP_OBSERVIT=0
ONLY=""
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --force)         FORCE=1 ;;
    --skip-observit) SKIP_OBSERVIT=1 ;;
    --only)          shift; ONLY="$1" ;;
    --only=*)        ONLY="${arg#*=}" ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \?//' | head -30
      exit 0 ;;
  esac
done

# ── Setup ────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

OUT_DIR="results_transmission"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUT_DIR/logs_publication_$TS"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"

SEEDS="42,123,7,1337,2025"
SIM_OUT="$OUT_DIR/paper_benchmark_sim.json"
REAL_OUT="$OUT_DIR/paper_benchmark_real.json"
MM_OUT="$OUT_DIR/paper_benchmark_real_mm.json"
OBS_OUT="$OUT_DIR/paper_benchmark_observit.json"
OBS_MM_OUT="$OUT_DIR/paper_benchmark_observit_mm.json"
SHA_OUT="$OUT_DIR/paper_benchmark.sha256"
MANIFEST_OUT="$OUT_DIR/canonical_results_manifest.json"

OBSERVIT_DATA="${OBSERVIT_DATA:-datasets/sockshop_runs_full}"
LOKI_URL="${LOKI_URL:-http://localhost:3100}"
LOKI_NS="${LOKI_NS:-sock-shop}"

# ── Logging helpers ──────────────────────────────────────────────────────────
RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
log()  { echo "${BLU}[$(date '+%H:%M:%S')]${RST} $*" | tee -a "$MAIN_LOG"; }
ok()   { echo "${GRN}OK${RST}    $*"   | tee -a "$MAIN_LOG"; }
warn() { echo "${YLW}WARN${RST}  $*"   | tee -a "$MAIN_LOG"; }
err()  { echo "${RED}FAIL${RST}  $*"   | tee -a "$MAIN_LOG"; }

run_phase_p() { [[ -z "$ONLY" ]] || [[ ",$ONLY," == *",$1,"* ]]; }
should_run()  { [[ $FORCE -eq 1 ]] || [[ ! -s "$1" ]]; }

run_cmd() {
  local label="$1"; shift
  log "${BLU}CMD${RST} ($label): $*"
  if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
  "$@" 2>&1 | tee -a "$LOG_DIR/${label}.log" "$MAIN_LOG"
  return ${PIPESTATUS[0]}
}

print_summary() {
  echo ""
  log "============================================================"
  log "PUBLICATION SUITE SUMMARY (timestamp: $TS)"
  log "============================================================"
  for f in "$SIM_OUT" "$REAL_OUT" "$MM_OUT" "$OBS_OUT" "$OBS_MM_OUT"; do
    if [[ -s "$f" ]]; then
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      ok "$(printf '%-55s' "$f") $(numfmt --to=iec --suffix=B "$sz" 2>/dev/null || echo "${sz}B")"
    else
      warn "$(printf '%-55s' "$f") missing"
    fi
  done
  log "Logs   : $LOG_DIR/"
  log "SHA256 : $SHA_OUT"
}

trap 'err "Interrupted at line $LINENO"; print_summary; exit 130' INT TERM

log "============================================================"
log "GNN-RCA — Publication Result Suite"
log "  timestamp     : $TS"
log "  seeds         : $SEEDS"
log "  output dir    : $OUT_DIR/"
log "  logs          : $LOG_DIR/"
log "  dry-run       : $DRY_RUN"
log "  force         : $FORCE"
log "  skip-observit : $SKIP_OBSERVIT"
log "  only          : ${ONLY:-all}"
log "============================================================"

# ── PHASE 0: PRE-FLIGHT ──────────────────────────────────────────────────────
if run_phase_p 0; then
  log ""; log "── PHASE 0/6: Pre-flight checks ──"

  # Python deps
  if ! python -c "import torch, torch_geometric, numpy, pandas, sklearn" 2>/dev/null; then
    err "Required Python deps missing. Run: pip install -e '.[dev]'"
    exit 1
  fi
  ok "Python deps importable"

  # Lint
  if [[ $DRY_RUN -eq 0 ]]; then
    if python -m ruff check rca_pipeline >"$LOG_DIR/00_ruff.log" 2>&1; then
      ok "ruff: clean"
    else
      err "ruff failed — see $LOG_DIR/00_ruff.log"; exit 1
    fi
    if python -m pytest -q rca_pipeline/tests >"$LOG_DIR/00_pytest.log" 2>&1; then
      ok "pytest: $(grep -oE '[0-9]+ passed' "$LOG_DIR/00_pytest.log" | head -1)"
    else
      err "pytest failed — see $LOG_DIR/00_pytest.log"; exit 1
    fi
    if bash scripts/pre_publish_audit.sh >"$LOG_DIR/00_audit.log" 2>&1; then
      ok "pre-publish audit: passed"
    else
      warn "audit reported issues — see $LOG_DIR/00_audit.log"
    fi
  fi

  # Record runtime env
  python - >"$LOG_DIR/00_env.json" <<'PY'
import json, platform, sys
try:
    import torch, torch_geometric
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
except Exception as e:
    info = {"error": str(e)}
print(json.dumps(info, indent=2))
PY
  ok "env captured: $LOG_DIR/00_env.json"
fi

# ── PHASE 1: SIMULATED ───────────────────────────────────────────────────────
if run_phase_p 1; then
  log ""; log "── PHASE 1/6: Simulated benchmark (BA N=20/50/100 × 5 seeds) ──"
  if should_run "$SIM_OUT"; then
    T=$SECONDS
    run_cmd 01_simulated python -m rca_pipeline.benchmark \
      --sizes 20,50,100 \
      --seeds "$SEEDS" \
      --num-samples 3000 \
      --supervised-epochs 150 \
      --late-contrastive-epochs 80 \
      --sta-warmup-epochs 20 \
      --sta-k-anneal-epochs 60 \
      --baseline-epochs 150 \
      --deterministic \
      --output-json "$SIM_OUT"
    ok "Phase 1 done in $((SECONDS-T))s → $SIM_OUT"
  else
    ok "Skipped (output exists): $SIM_OUT"
  fi
fi

# ── PHASE 2: RCAEval METRICS-ONLY ────────────────────────────────────────────
if run_phase_p 2; then
  log ""; log "── PHASE 2/6: RCAEval real (metrics-only, 5 seeds) ──"
  if should_run "$REAL_OUT"; then
    T=$SECONDS
    run_cmd 02_real_metrics python -m rca_pipeline.benchmark_real \
      --dataset all \
      --seeds "$SEEDS" \
      --epochs 100 \
      --sta-contrastive-warmup 0 \
      --enable-drift \
      --deterministic \
      --output-json "$REAL_OUT"
    ok "Phase 2 done in $((SECONDS-T))s → $REAL_OUT"
  else
    ok "Skipped (output exists): $REAL_OUT"
  fi
fi

# ── PHASE 3: RCAEval MULTIMODAL STRICT ───────────────────────────────────────
if run_phase_p 3; then
  log ""; log "── PHASE 3/6: RCAEval multimodal strict (5 seeds) ──"
  if should_run "$MM_OUT"; then
    T=$SECONDS
    run_cmd 03_real_mm python -m rca_pipeline.benchmark_real \
      --dataset all \
      --seeds "$SEEDS" \
      --epochs 100 \
      --sta-contrastive-warmup 0 \
      --enable-drift \
      --deterministic \
      --use-mm --mm-strict \
      --output-json "$MM_OUT"
    ok "Phase 3 done in $((SECONDS-T))s → $MM_OUT"
  else
    ok "Skipped (output exists): $MM_OUT"
  fi
fi

# ── PHASE 4 & 5: OBSERVIT (live cluster) ─────────────────────────────────────
LOKI_PF_PID=""
cleanup_loki_pf() {
  if [[ -n "$LOKI_PF_PID" ]] && kill -0 "$LOKI_PF_PID" 2>/dev/null; then
    log "Stopping Loki port-forward (PID=$LOKI_PF_PID)"
    kill "$LOKI_PF_PID" 2>/dev/null || true
  fi
}
trap 'cleanup_loki_pf; err "Interrupted"; print_summary; exit 130' INT TERM

observit_preflight() {
  if [[ $SKIP_OBSERVIT -eq 1 ]]; then
    warn "ObservIT phases skipped (--skip-observit)"; return 1
  fi
  if ! command -v kubectl >/dev/null 2>&1; then
    warn "kubectl not found — skipping ObservIT phases"; return 1
  fi
  if ! kubectl get ns sock-shop >/dev/null 2>&1; then
    warn "namespace sock-shop unreachable — skipping ObservIT phases"; return 1
  fi
  if [[ ! -d "$OBSERVIT_DATA" ]] || ! ls "$OBSERVIT_DATA"/*.csv >/dev/null 2>&1; then
    warn "$OBSERVIT_DATA contains no CSVs — skipping ObservIT phases"
    warn "  Generate first: bash scripts/run_sockshop_campaign.sh"
    return 1
  fi
  ok "ObservIT pre-flight: cluster reachable, $(ls "$OBSERVIT_DATA"/*.csv | wc -l) CSVs in $OBSERVIT_DATA"
  return 0
}

if run_phase_p 4 && observit_preflight; then
  log ""; log "── PHASE 4/6: ObservIT sock-shop (metrics-only, 5 seeds) ──"
  if should_run "$OBS_OUT"; then
    T=$SECONDS
    run_cmd 04_observit_metrics python -m rca_pipeline.benchmark_real \
      --observit-csv      "$OBSERVIT_DATA" \
      --observit-window   12 \
      --observit-topology sock_shop \
      --seeds             "$SEEDS" \
      --epochs            100 \
      --sta-contrastive-warmup 0 \
      --enable-drift \
      --deterministic \
      --include-late-gib \
      --output-json "$OBS_OUT"
    ok "Phase 4 done in $((SECONDS-T))s → $OBS_OUT"
  else
    ok "Skipped (output exists): $OBS_OUT"
  fi
fi

if run_phase_p 5 && observit_preflight; then
  log ""; log "── PHASE 5/6: ObservIT sock-shop multimodal (Loki logs) ──"
  if should_run "$OBS_MM_OUT"; then
    # Set up Loki port-forward if not already reachable
    if ! curl -sf "$LOKI_URL/ready" >/dev/null 2>&1; then
      log "Starting Loki port-forward (3100:3100)…"
      kubectl port-forward -n loki svc/loki 3100:3100 --address=127.0.0.1 \
        >"$LOG_DIR/05_loki_pf.log" 2>&1 &
      LOKI_PF_PID=$!
      sleep 4
      if ! curl -sf "$LOKI_URL/ready" >/dev/null 2>&1; then
        warn "Loki not reachable at $LOKI_URL — skipping Phase 5"
        cleanup_loki_pf
        LOKI_PF_PID=""
      else
        ok "Loki reachable at $LOKI_URL (PID=$LOKI_PF_PID)"
      fi
    else
      ok "Loki already reachable at $LOKI_URL"
    fi

    if curl -sf "$LOKI_URL/ready" >/dev/null 2>&1; then
      T=$SECONDS
      run_cmd 05_observit_mm python -m rca_pipeline.benchmark_real \
        --observit-csv      "$OBSERVIT_DATA" \
        --observit-window   12 \
        --observit-topology sock_shop \
        --seeds             "$SEEDS" \
        --epochs            100 \
        --sta-contrastive-warmup 0 \
        --enable-drift \
        --deterministic \
        --include-late-gib \
        --use-mm --mm-logs-only \
        --loki-url       "$LOKI_URL" \
        --loki-namespace "$LOKI_NS" \
        --output-json "$OBS_MM_OUT"
      ok "Phase 5 done in $((SECONDS-T))s → $OBS_MM_OUT"
      cleanup_loki_pf
      LOKI_PF_PID=""
    fi
  else
    ok "Skipped (output exists): $OBS_MM_OUT"
  fi
fi

# ── PHASE 6: POST-PROCESSING ─────────────────────────────────────────────────
if run_phase_p 6; then
  log ""; log "── PHASE 6/6: SHA256 + canonical manifest + summary report ──"

  if [[ $DRY_RUN -eq 0 ]]; then
    pushd "$OUT_DIR" >/dev/null
    : >"$(basename "$SHA_OUT")"
    for f in paper_benchmark_sim.json paper_benchmark_real.json \
             paper_benchmark_real_mm.json paper_benchmark_observit.json \
             paper_benchmark_observit_mm.json; do
      [[ -s "$f" ]] && sha256sum "$f" >>"$(basename "$SHA_OUT")"
    done
    popd >/dev/null
    ok "SHA256 written to $SHA_OUT"

    # Regenerate canonical manifest (best-effort; tool may not see new files)
    if python -m rca_pipeline.results_manifest --repo-root . \
         --output "$MANIFEST_OUT" >>"$LOG_DIR/06_manifest.log" 2>&1; then
      ok "Canonical manifest regenerated → $MANIFEST_OUT"
      sha256sum "$MANIFEST_OUT" > "${MANIFEST_OUT}.sha256"
    else
      warn "results_manifest tool failed; manifest not updated (see 06_manifest.log)"
    fi

    # Reliability summary across all produced JSONs
    python - <<PYEOF | tee "$LOG_DIR/06_reliability.txt" "$MAIN_LOG" >/dev/null
import json, os
files = [
    "results_transmission/paper_benchmark_sim.json",
    "results_transmission/paper_benchmark_real.json",
    "results_transmission/paper_benchmark_real_mm.json",
    "results_transmission/paper_benchmark_observit.json",
    "results_transmission/paper_benchmark_observit_mm.json",
]
print()
print("=" * 64)
print("RELIABILITY SUMMARY (5 seeds → bootstrap CIs should be reliable)")
print("=" * 64)
for f in files:
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        print(f"\n[MISSING] {f}")
        continue
    print(f"\n── {f}")
    try: d = json.load(open(f))
    except Exception as e:
        print(f"  parse error: {e}"); continue
    if "runs" in d:
        for run in d["runs"]:
            N = run.get("N", "?")
            n_seeds = len(run.get("seeds", []))
            print(f"  N={N:3} seeds={n_seeds}")
            best, bv = "", 0.0
            for k, v in run.get("metrics", {}).items():
                if isinstance(v, dict) and v.get("top1", 0) > bv:
                    bv, best = v["top1"], k
            if best: print(f"     best Top-1: {best[:40]:<40} {bv:.4f}")
    else:
        for ds, dv in d.items():
            if not isinstance(dv, dict): continue
            for m in ("STA", "GATv2", "Late-GIB", "STA-MM", "GATv2-MM"):
                if m in dv and isinstance(dv[m], dict) and "ci95" in dv[m]:
                    n = dv[m]["ci95"]["avg1"]["n"]
                    rel = dv[m]["ci95"]["avg1"]["seed_level_bootstrap_reliable"]
                    a1 = dv[m]["avg1"]; lo = dv[m]["ci95"]["avg1"]["ci_low"]
                    hi = dv[m]["ci95"]["avg1"]["ci_high"]
                    flag = "✓" if rel else "✗"
                    print(f"  {ds:20s} {m:10s} Avg@1={a1:.4f} [{lo:.4f},{hi:.4f}] n={n} reliable={flag}")
PYEOF
  fi
fi

cleanup_loki_pf
print_summary
log ""
log "${GRN}DONE.${RST} Cite this run in the paper as commit:"
log "  $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
log "  timestamp: $TS"
log ""
log "Next steps:"
log "  git add results_transmission/paper_benchmark_*.json $SHA_OUT \\"
log "          $MANIFEST_OUT ${MANIFEST_OUT}.sha256"
log "  git commit -m 'Add publication-grade benchmark results (5 seeds, deterministic)'"
