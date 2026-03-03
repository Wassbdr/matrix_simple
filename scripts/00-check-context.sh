#!/bin/bash
##############################################
# Vérification du contexte Kubernetes
# S'assure qu'on cible bien observit-cluster1
##############################################
set -e

EXPECTED_CONTEXT="observit-cluster1"
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "NONE")

echo "=== Vérification du contexte Kubernetes ==="
echo "  Contexte attendu  : ${EXPECTED_CONTEXT}"
echo "  Contexte actuel   : ${CURRENT_CONTEXT}"

if [ "${CURRENT_CONTEXT}" != "${EXPECTED_CONTEXT}" ]; then
    echo ""
    echo "⚠️  ERREUR : Le contexte kubectl actuel n'est pas '${EXPECTED_CONTEXT}'."
    echo "   Changez de contexte avec :"
    echo "   kubectl config use-context ${EXPECTED_CONTEXT}"
    echo ""
    exit 1
fi

echo "✅ Contexte OK — ciblant ${EXPECTED_CONTEXT}"
echo ""

# Vérification de la connectivité
echo "Vérification de la connectivité au cluster..."
if kubectl cluster-info &>/dev/null; then
    echo "✅ Cluster accessible"
    kubectl get nodes --no-headers 2>/dev/null | head -3
else
    echo "❌ Impossible de contacter le cluster"
    exit 1
fi
echo ""
