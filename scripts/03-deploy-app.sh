#!/bin/bash
##############################################
# Déploiement de l'application 3-Tiers
# Namespace : rca-sandbox
##############################################
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Vérification du contexte
source "${SCRIPT_DIR}/00-check-context.sh"

NAMESPACE="rca-sandbox"

echo "=== Création du namespace ==="
kubectl apply -f "${SCRIPT_DIR}/../k8s/namespace.yaml"

echo ""
echo "=== Déploiement de la Base de Données (PostgreSQL) ==="
kubectl apply -f "${SCRIPT_DIR}/../k8s/app/db.yaml"

echo "Attente du pod de la base de données..."
sleep 5
kubectl wait --for=condition=Ready pod -l app=database -n "${NAMESPACE}" --timeout=120s || {
    echo "⚠️  Le pod database n'est pas prêt après 120s — continuons quand même"
}

echo ""
echo "=== Déploiement du Backend (FastAPI) ==="
kubectl apply -f "${SCRIPT_DIR}/../k8s/app/backend.yaml"

echo ""
echo "=== Déploiement du Frontend (Nginx) ==="
kubectl apply -f "${SCRIPT_DIR}/../k8s/app/frontend.yaml"

echo ""
echo "Attente de tous les pods..."
sleep 5
kubectl wait --for=condition=Ready pod --all -n "${NAMESPACE}" --timeout=300s || {
    echo "⚠️  Certains pods ne sont pas encore prêts"
}

echo ""
echo "=== Application 3-Tiers déployée avec succès ! ==="
echo ""
kubectl get pods -n "${NAMESPACE}" -l 'app in (database,backend,frontend)'
echo ""
echo "🌐 Accès au Frontend :"
echo "  kubectl port-forward svc/frontend -n ${NAMESPACE} 8000:80"
echo "  Puis ouvrir http://localhost:8000"
echo ""
