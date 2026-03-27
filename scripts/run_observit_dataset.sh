#!/usr/bin/env bash
# Génère un CSV via Chaos + Locust + Prometheus (cluster observ-it).
# Exemple :
#   kubectl port-forward -n rca-sandbox svc/rca-prometheus-server 9090:80 &
#   kubectl port-forward -n rca-sandbox svc/frontend 8080:80 &
#   ./scripts/run_observit_dataset.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/rca_pipeline"
exec "${ROOT}/.venv/bin/python" dataset_builder.py \
  --prometheus-url http://127.0.0.1:9090 \
  --chaos-file ../k8s/chaos/stress-cpu-db.yaml \
  --locust-file locustfile.py \
  --frontend-url http://127.0.0.1:8080 \
  --users 30 \
  --baseline-duration 300 \
  --anomaly-duration 300 \
  --root-cause database \
  --output-dir ../datasets/observit_runs/ \
  "$@"
