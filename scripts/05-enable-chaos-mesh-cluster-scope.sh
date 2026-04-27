#!/usr/bin/env bash
##############################################
# Reconfigure le Chaos-Mesh existant en cluster-scoped
# pour permettre l'injection dans sock-shop (et autres NS).
#
# Release Helm inchangée (chaos-mesh/rca-sandbox), seules les
# RBAC s'élargissent. Les expériences rca-sandbox existantes
# continuent de fonctionner.
#
# Avant: controllerManager.targetNamespace=rca-sandbox
# Après: clusterScoped=true (targetNamespace ignoré)
##############################################
set -euo pipefail

RELEASE="chaos-mesh"
NS="rca-sandbox"

echo "=== Helm upgrade chaos-mesh -> clusterScoped=true ==="
helm repo update chaos-mesh 2>/dev/null || helm repo add chaos-mesh https://charts.chaos-mesh.org
helm upgrade --install "${RELEASE}" chaos-mesh/chaos-mesh \
  --namespace "${NS}" \
  --reuse-values \
  --set clusterScoped=true \
  --set controllerManager.targetNamespace=null \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/k3s/containerd/containerd.sock \
  --set dashboard.enabled=false

echo
echo "Attente du rollout du controller..."
kubectl rollout status -n "${NS}" deploy/chaos-controller-manager --timeout=180s || true

echo
echo "=== Vérification : création d'un PodChaos dry-run dans sock-shop ==="
cat <<'EOF' | kubectl apply --dry-run=server -f - && echo OK || { echo FAIL; exit 1; }
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: dryrun-sockshop
  namespace: sock-shop
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces: [sock-shop]
    labelSelectors:
      name: user
EOF

echo
echo "Chaos-Mesh est maintenant cluster-scoped et peut cibler sock-shop."
