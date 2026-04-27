#!/usr/bin/env bash
##############################################
# Génère un CSV RCA via Chaos-Mesh + Locust + Prometheus
# Cible: sock-shop sur observit-cluster1.
#
# Prometheus: monitoring-metrics/prometheus-server (svc port 80 -> pod 9090)
#   (le svc cattle-monitoring-system/rancher-monitoring-prometheus-nodeport n'a pas d'endpoints)
# Front-end:  sock-shop/front-end (NodePort 30001; on port-forward 8080->80)
#
# Usage:
#   ./scripts/run_sockshop_dataset.sh --chaos-file k8s/chaos/sockshop/stress-cpu-catalogue.yaml \
#       --root-cause catalogue --scenario stress-cpu-catalogue
##############################################
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Port-forward front-end (kill propre à la fin).
PF_LOG="$(mktemp)"
kubectl port-forward -n sock-shop svc/front-end 8080:80 --address=127.0.0.1 \
  >"${PF_LOG}" 2>&1 &
PF_FRONT_PID=$!
trap 'kill ${PF_FRONT_PID} 2>/dev/null || true; rm -f "${PF_LOG}"' EXIT
sleep 3

mkdir -p datasets/sockshop_runs

# Le builder gère le port-forward Prometheus tout seul via --prom-pf-*
exec python3 -m rca_pipeline.dataset_builder \
  --prometheus-url http://127.0.0.1:9090 \
  --namespace sock-shop \
  --pod-regex "(front-end|catalogue|carts|orders|payment|user|shipping|queue-master|rabbitmq|carts-db|catalogue-db|orders-db|user-db|session-db)(-.*)?" \
  --prom-pf-ns monitoring-metrics \
  --prom-pf-svc svc/prometheus-server \
  --prom-pf-ports 9090:80 \
  --locust-file rca_pipeline/locustfile_sockshop.py \
  --frontend-url http://127.0.0.1:8080 \
  --users 20 \
  --baseline-duration 300 \
  --anomaly-duration 300 \
  --output-dir datasets/sockshop_runs/ \
  "$@"
