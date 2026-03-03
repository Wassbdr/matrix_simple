#!/bin/bash
##############################################
# Installation de Chaos Mesh
# Scope limité au namespace rca-sandbox
##############################################
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Vérification du contexte
source "${SCRIPT_DIR}/00-check-context.sh"

NAMESPACE="rca-sandbox"

echo "=== Installation de Chaos Mesh ==="

helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
helm repo update

# Installation de Chaos Mesh dans le namespace rca-sandbox
# targetNamespace limite les expériences au seul namespace rca-sandbox
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace "${NAMESPACE}" \
  --set controllerManager.targetNamespace="${NAMESPACE}" \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/k3s/containerd/containerd.sock \
  --set dashboard.enabled=false \
  --set controllerManager.resources.requests.cpu=50m \
  --set controllerManager.resources.requests.memory=64Mi \
  --set controllerManager.resources.limits.cpu=200m \
  --set controllerManager.resources.limits.memory=256Mi

echo ""
echo "Attente de l'installation de Chaos Mesh..."
sleep 10
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/instance=chaos-mesh \
  -n "${NAMESPACE}" --timeout=300s || {
    echo "⚠️  Certains pods Chaos Mesh ne sont pas encore prêts"
    kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/instance=chaos-mesh
}

echo ""
echo "=== Chaos Mesh installé avec succès ! ==="
echo ""
echo "🧪 Expériences disponibles :"
echo "  Latence réseau    : kubectl apply -f k8s/chaos/network-delay.yaml"
echo "  Stress CPU (DB)   : kubectl apply -f k8s/chaos/stress-db.yaml"
echo ""
echo "  Annuler tout      : kubectl delete networkchaos,stresschaos,podchaos --all -n ${NAMESPACE}"
echo ""
