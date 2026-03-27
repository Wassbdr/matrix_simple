#!/usr/bin/env bash
# SOTA-oriented real-data campaign:
# - multi-root-cause coverage
# - multi-failure-family coverage
# - ready for benchmark_real.py --observit-csv <dir>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
PIPE="${ROOT}/rca_pipeline"
OUT_DIR="${ROOT}/datasets/observit_runs"

PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:8080}"
USERS="${USERS:-30}"
BASELINE="${BASELINE:-300}"
ANOMALY="${ANOMALY:-300}"

mkdir -p "$OUT_DIR"
cd "$PIPE"

# scenario|chaos_file|root_cause|failure_type
SCENARIOS=(
  "stress-cpu-db|../k8s/chaos/stress-cpu-db.yaml|database|hard"
  "stress-mem-cache|../k8s/chaos/stress-mem-cache.yaml|cache|hard"
  "network-delay-frontend-auth|../k8s/chaos/network-delay-frontend-auth.yaml|auth|gray"
  "network-delay-frontend-backend|../k8s/chaos/network-delay.yaml|backend|gray"
  "stress-db-cascade|../k8s/chaos/stress-db.yaml|database|cascade"
)

for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r scenario chaos_file root_cause failure_type <<< "$entry"
  echo
  echo "=== Running scenario: ${scenario} (rc=${root_cause}, type=${failure_type}) ==="
  "$PY" dataset_builder.py \
    --prometheus-url "$PROM_URL" \
    --chaos-file "$chaos_file" \
    --locust-file locustfile.py \
    --frontend-url "$FRONTEND_URL" \
    --users "$USERS" \
    --baseline-duration "$BASELINE" \
    --anomaly-duration "$ANOMALY" \
    --root-cause "$root_cause" \
    --failure-type "$failure_type" \
    --scenario "$scenario" \
    --output-dir "$OUT_DIR"
done

echo
echo "Campaign complete. Validating coverage..."
"$PY" validate_observit_campaign.py --input-dir "$OUT_DIR"
echo
echo "Run benchmark:"
echo "  cd \"$PIPE\" && ${PY} benchmark_real.py --observit-csv \"$OUT_DIR\" --observit-window 12 --seeds 42,123,7 --epochs 100 --output-json \"$OUT_DIR/benchmark_full_real.json\""
