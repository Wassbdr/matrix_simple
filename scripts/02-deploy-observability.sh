#!/bin/bash
##############################################
# Déploiement de la stack d'Observabilité
# Tout est déployé dans le namespace rca-sandbox
#
# NOTE: On utilise le préfixe "rca-" sur les noms
# de releases Helm pour éviter les conflits avec
# les ressources cluster-wide existantes.
# On désactive aussi la création de ClusterRole/
# ClusterRoleBinding (rbac.create=false) car le
# cluster a déjà des rôles Prometheus globaux.
##############################################
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Vérification du contexte
source "${SCRIPT_DIR}/00-check-context.sh"

NAMESPACE="rca-sandbox"

echo "=== Création du namespace ==="
kubectl apply -f "${SCRIPT_DIR}/../k8s/namespace.yaml"

echo ""
echo "=== Ajout des repos Helm ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts 2>/dev/null || true
helm repo update

# ──────────────────────────────────────────
# 1. Prometheus (version légère)
#    - rbac.create=false pour éviter le conflit
#      ClusterRole "prometheus-server" existant
#    - Nom de release "rca-prometheus" unique
# ──────────────────────────────────────────
echo ""
echo "=== Installation de Prometheus (léger) ==="
helm upgrade --install rca-prometheus prometheus-community/prometheus \
  --namespace "${NAMESPACE}" \
  --set rbac.create=false \
  --set server.persistentVolume.enabled=false \
  --set server.retention="6h" \
  --set alertmanager.enabled=false \
  --set kube-state-metrics.enabled=false \
  --set prometheus-node-exporter.enabled=false \
  --set prometheus-pushgateway.enabled=false \
  --set server.global.scrape_interval="15s" \
  --set server.resources.requests.cpu=100m \
  --set server.resources.requests.memory=256Mi \
  --set server.resources.limits.cpu=500m \
  --set server.resources.limits.memory=512Mi \
  --set configmapReload.prometheus.enabled=true \
  --set server.fullnameOverride="rca-prometheus-server"

# ──────────────────────────────────────────
# 2. Jaeger (All-in-one avec récepteur OTLP)
# ──────────────────────────────────────────
echo ""
echo "=== Installation de Jaeger (all-in-one) ==="
helm upgrade --install rca-jaeger jaegertracing/jaeger \
  --namespace "${NAMESPACE}" \
  --set provisionDataStore.cassandra=false \
  --set allInOne.enabled=true \
  --set storage.type=memory \
  --set agent.enabled=false \
  --set collector.enabled=false \
  --set query.enabled=false \
  --set allInOne.extraEnv[0].name=COLLECTOR_OTLP_ENABLED \
  --set allInOne.extraEnv[0].value="true"

# ──────────────────────────────────────────
# 3. Loki + Promtail
# ──────────────────────────────────────────
echo ""
echo "=== Installation de Loki-Stack ==="
helm upgrade --install rca-loki grafana/loki-stack \
  --namespace "${NAMESPACE}" \
  --set grafana.enabled=false \
  --set prometheus.enabled=false \
  --set promtail.enabled=true \
  --set loki.persistence.enabled=false \
  --set loki.resources.requests.cpu=50m \
  --set loki.resources.requests.memory=128Mi \
  --set loki.resources.limits.cpu=200m \
  --set loki.resources.limits.memory=256Mi

# ──────────────────────────────────────────
# 4. Grafana (avec datasources préconfigurées)
# ──────────────────────────────────────────
echo ""
echo "=== Installation de Grafana ==="
helm upgrade --install rca-grafana grafana/grafana \
  --namespace "${NAMESPACE}" \
  --set adminPassword=admin \
  --set persistence.enabled=false \
  --set rbac.create=false \
  --set 'datasources.datasources\.yaml.apiVersion=1' \
  --set 'datasources.datasources\.yaml.datasources[0].name=Prometheus' \
  --set 'datasources.datasources\.yaml.datasources[0].type=prometheus' \
  --set "datasources.datasources\.yaml.datasources[0].url=http://rca-prometheus-server.${NAMESPACE}.svc.cluster.local:80" \
  --set 'datasources.datasources\.yaml.datasources[0].access=proxy' \
  --set 'datasources.datasources\.yaml.datasources[0].isDefault=true' \
  --set 'datasources.datasources\.yaml.datasources[1].name=Loki' \
  --set 'datasources.datasources\.yaml.datasources[1].type=loki' \
  --set "datasources.datasources\.yaml.datasources[1].url=http://rca-loki.${NAMESPACE}.svc.cluster.local:3100" \
  --set 'datasources.datasources\.yaml.datasources[1].access=proxy' \
  --set 'datasources.datasources\.yaml.datasources[2].name=Jaeger' \
  --set 'datasources.datasources\.yaml.datasources[2].type=jaeger' \
  --set "datasources.datasources\.yaml.datasources[2].url=http://rca-jaeger-query.${NAMESPACE}.svc.cluster.local:16686" \
  --set 'datasources.datasources\.yaml.datasources[2].access=proxy' \
  --set resources.requests.cpu=50m \
  --set resources.requests.memory=128Mi \
  --set resources.limits.cpu=200m \
  --set resources.limits.memory=256Mi

echo ""
echo "=== Observabilité déployée avec succès dans ${NAMESPACE} ! ==="
echo ""
echo " Accès aux dashboards :"
echo "  Grafana    : kubectl port-forward svc/rca-grafana -n ${NAMESPACE} 3000:80     (admin/admin)"
echo "  Prometheus : kubectl port-forward svc/rca-prometheus-server -n ${NAMESPACE} 9090:80"
echo "  Jaeger     : kubectl port-forward svc/rca-jaeger-query -n ${NAMESPACE} 16686:16686"
echo ""
