#!/usr/bin/env bash
##############################################
# Campagne metrics-only sock-shop: ~10 runs sur 5 scénarios chaos.
# 2 runs par scénario => 10 runs.
# Durée totale: ~100 min (5min baseline + 5min anomalie par run).
#
# Pré-requis:
#   1. Chaos-Mesh cluster-scoped:
#        ./scripts/05-enable-chaos-mesh-cluster-scope.sh
#   2. sock-shop accessible (déjà deployé sur observit-cluster1)
#   3. .venv activé, deps installées
##############################################
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPEATS="${REPEATS:-2}"
BASELINE="${BASELINE:-300}"
ANOMALY="${ANOMALY:-300}"

# (root_cause | failure_type | scenario | chaos_file)
CASES=(
  "catalogue|hard|stress-cpu-catalogue|k8s/chaos/sockshop/stress-cpu-catalogue.yaml"
  "carts|hard|stress-mem-carts|k8s/chaos/sockshop/stress-mem-carts.yaml"
  "orders-db|cascade|network-delay-orders-to-orders-db|k8s/chaos/sockshop/network-delay-orders-to-orders-db.yaml"
  "catalogue-db|gray|network-loss-catalogue-to-catalogue-db|k8s/chaos/sockshop/network-loss-catalogue-to-catalogue-db.yaml"
  "user|hard|pod-kill-user|k8s/chaos/sockshop/pod-kill-user.yaml"
)

mkdir -p datasets/sockshop_runs
i=0
for case in "${CASES[@]}"; do
  IFS='|' read -r RC FT SC FILE <<<"${case}"
  for r in $(seq 1 "${REPEATS}"); do
    i=$((i+1))
    echo "=== Run ${i} : scenario=${SC} rc=${RC} repeat=${r}/${REPEATS} ==="
    ./scripts/run_sockshop_dataset.sh \
      --chaos-file "${FILE}" \
      --root-cause "${RC}" \
      --failure-type "${FT}" \
      --scenario "${SC}" \
      --baseline-duration "${BASELINE}" \
      --anomaly-duration "${ANOMALY}"
    sleep 20
  done
done

echo
echo "=== Campagne terminée: ${i} runs -> datasets/sockshop_runs/ ==="
ls -1 datasets/sockshop_runs/*.csv 2>/dev/null | wc -l
