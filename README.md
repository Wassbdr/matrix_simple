# RCA Sandbox -- Environnement de Laboratoire

Environnement Kubernetes isole pour la recherche sur le **Root Cause Analysis (RCA)** base sur l'**Information Bottleneck** et l'**apprentissage contrastif spatio-temporel**. Deploye sur le cluster distant `observit-cluster1` (Rancher), entierement dans le namespace `rca-sandbox`.

---

## Architecture (5 noeuds)

```
+-------------------------------------------------------------------+
|                    Namespace: rca-sandbox                          |
|                                                                   |
|  +----------+     +----------+     +----------+                   |
|  | Frontend |---->|   Auth   |     |  Cache   |                   |
|  |  (Nginx) |     | (FastAPI)|     |  (Redis) |                   |
|  +----+-----+     +----------+     +----+-----+                   |
|       |                                 ^                         |
|       +----------->+----------+         |                         |
|                    | Backend  |---------+                         |
|                    | (FastAPI)|                                    |
|                    +----+-----+                                   |
|                         |                                         |
|                         v                                         |
|                    +----------+                                   |
|                    |PostgreSQL|                                    |
|                    |(Database)|                                    |
|                    +----------+                                   |
|                                                                   |
|  Observabilite : Prometheus | Jaeger (OTLP) | Loki | Grafana     |
|  Chaos Mesh    : NetworkChaos | StressChaos | PodChaos            |
+-------------------------------------------------------------------+
```

**Graphe de dependances** :
`Frontend -> Auth`, `Frontend -> Backend`, `Backend -> Cache`, `Backend -> Database`

---

## Prerequis

- [kubectl](https://kubernetes.io/docs/tasks/tools/) configure sur le contexte `observit-cluster1`
- [Helm](https://helm.sh/docs/intro/install/) v3+
- Python 3.9+ avec les paquets suivants : `locust`, `requests`, `pandas`, `torch`, `torch_geometric`, `scikit-learn`
- Acces RBAC suffisant pour creer des ressources dans un namespace dedie

---

## Structure du projet

```
matrix_simple/
├── k8s/
│   ├── namespace.yaml                  # Namespace rca-sandbox
│   ├── app/
│   │   ├── db.yaml                     # PostgreSQL (Deployment + Service + PVC)
│   │   ├── cache.yaml                  # Redis (Deployment + Service)
│   │   ├── auth.yaml                   # FastAPI Auth (ConfigMap + Deployment + Service)
│   │   ├── backend.yaml                # FastAPI Backend (ConfigMap + Deployment + Service)
│   │   └── frontend.yaml               # Nginx (ConfigMap + Deployment + Service)
│   └── chaos/
│       ├── network-delay.yaml          # NetworkChaos : latence Frontend <-> Backend
│       └── stress-db.yaml              # StressChaos (CPU) + PodChaos (kill)
├── rca_pipeline/
│   ├── locustfile.py                   # Generateur de trafic (80% lecture, 20% ecriture)
│   ├── dataset_builder.py              # Orchestrateur Chaos Mesh + Extracteur Prometheus
│   ├── graph_builder.py                # Convertisseur CSV -> PyTorch Geometric Data
│   └── verify_tensors.py               # Verificateur de tenseurs (dimensions, NaN, scaling)
├── scripts/
│   ├── 00-check-context.sh             # Verifie le contexte kubectl
│   ├── 02-deploy-observability.sh      # Helm : Prometheus, Jaeger, Loki, Grafana
│   ├── 03-deploy-app.sh                # Deploie l'application 5-tiers
│   └── 04-install-chaos-mesh.sh        # Installe Chaos Mesh (scope rca-sandbox)
└── README.md
```

---

## Deploiement etape par etape

### 0. Verifier le contexte

```bash
kubectl config use-context observit-cluster1
kubectl config current-context  # Doit afficher: observit-cluster1
```

### 1. Rendre les scripts executables

```bash
chmod +x scripts/*.sh
```

### 2. Deployer l'Observabilite

```bash
./scripts/02-deploy-observability.sh
```

| Composant | Chart Helm | Port |
|-----------|-----------|------|
| Prometheus | `prometheus-community/prometheus` | 9090 |
| Jaeger | `jaegertracing/jaeger` (all-in-one) | 16686 |
| Loki+Promtail | `grafana/loki-stack` | 3100 |
| Grafana | `grafana/grafana` | 3000 |

### 3. Deployer l'Application

```bash
./scripts/03-deploy-app.sh
# Puis deployer les nouveaux services :
kubectl apply -f k8s/app/auth.yaml
kubectl apply -f k8s/app/cache.yaml
# Redemarrer frontend et backend pour charger les ConfigMaps mis a jour :
kubectl rollout restart deployment frontend backend -n rca-sandbox
```

### 4. Installer Chaos Mesh

```bash
./scripts/04-install-chaos-mesh.sh
```

### 5. Verifier l'etat

```bash
kubectl get pods -n rca-sandbox
# Les 5 pods applicatifs doivent etre Running :
# auth, backend, cache, database, frontend
```

---

## Acces aux services

```bash
# Frontend
kubectl port-forward svc/frontend -n rca-sandbox 8000:80

# Grafana (admin/admin)
kubectl port-forward svc/grafana -n rca-sandbox 3000:80

# Prometheus
kubectl port-forward svc/rca-prometheus-server -n rca-sandbox 9090:80

# Jaeger
kubectl port-forward svc/rca-jaeger -n rca-sandbox 16686:16686
```

---

## Pipeline RCA (rca_pipeline/)

Le pipeline automatise la boucle complete : generation de trafic, injection de fautes, extraction de metriques et transformation en graphes pour l'entrainement d'un modele spatio-temporel.

### Etape 1 : Generation de trafic (Locust)

Le script `locustfile.py` simule des utilisateurs visitant le Frontend avec une distribution asymetrique :
- **80%** de requetes de lecture (`GET /api/data` precede de `GET /auth/verify`)
- **20%** de requetes d'ecriture (`POST /api/data`)

```bash
# Lancement manuel (optionnel, le dataset_builder le fait automatiquement)
kubectl port-forward svc/frontend -n rca-sandbox 8080:80 &
locust -f rca_pipeline/locustfile.py --headless -u 20 -r 5 -H http://localhost:8080
```

### Etape 2 : Orchestration et extraction (dataset_builder.py)

Le script `dataset_builder.py` orchestre la boucle d'experimentation :
1. Demarre Locust en tache de fond
2. Attend 5 minutes (baseline saine)
3. Applique un fichier Chaos Mesh via `kubectl apply -f`
4. Attend 5 minutes (periode d'anomalie)
5. Supprime le chaos via `kubectl delete`
6. Interroge l'API Prometheus pour extraire les metriques (step=10s)
7. Exporte le resultat en `dataset_<run_id>.csv`

**Metriques PromQL extraites** : CPU usage, RAM usage, Network TX/RX, Error rate.

```bash
python3 rca_pipeline/dataset_builder.py
```

### Etape 3 : Construction des graphes (graph_builder.py)

Le script `graph_builder.py` convertit les CSV en objets `torch_geometric.data.Data` :
- **Noeuds (V=5)** : Frontend, Auth, Backend, Cache, Database
- **Aretes (E=4)** : Frontend->Auth, Frontend->Backend, Backend->Cache, Backend->Database
- **Features (F=5)** : cpu_usage, ram_usage, network_tx, network_rx, error_rate
- **Normalisation** : `StandardScaler` applique sur l'ensemble de la serie
- **Fenetres glissantes** : Tenseur final `X` de dimension `[5, 5, W]` (noeuds, features, window_size)

```bash
python3 -c "
from rca_pipeline.graph_builder import GraphBuilder
builder = GraphBuilder('dataset_XXXXXXXX.csv')
snapshots = builder.build_snapshots()
windows = builder.create_sliding_windows(snapshots, window_size=6)
builder.save_windows(windows, 'dataset_windows.pt')
"
```

### Etape 4 : Verification des tenseurs (verify_tensors.py)

Le script `verify_tensors.py` charge le fichier `.pt` et verifie :
- Dimension de `X` : `[nombre_de_noeuds, nombre_de_features, taille_fenetre]`
- Coherence de `edge_index` : `[2, nombre_d_aretes]`
- Absence de NaN / Inf
- Statistiques post-normalisation (moyenne ~ 0, variance ~ 1)

```bash
python3 -c "
from rca_pipeline.verify_tensors import verify_tensors
verify_tensors('dataset_windows.pt')
"
```

---

## Scenarios de Chaos

### A. Latence Reseau (Frontend <-> Auth)

```bash
kubectl apply -f k8s/chaos/network-delay-frontend-auth.yaml
# Annuler :
kubectl delete -f k8s/chaos/network-delay-frontend-auth.yaml
```

### B. Stress CPU sur Database

```bash
kubectl apply -f k8s/chaos/stress-cpu-db.yaml
# Annuler :
kubectl delete -f k8s/chaos/stress-cpu-db.yaml
```

### C. PodKill sur Cache (Redis)

```bash
kubectl apply -f k8s/chaos/pod-kill-cache.yaml
# Annuler :
kubectl delete -f k8s/chaos/pod-kill-cache.yaml
```

### D. Scenarios historiques

```bash
# Latence reseau Frontend <-> Backend (500ms + 5% perte)
kubectl apply -f k8s/chaos/network-delay.yaml

# Stress CPU + Kill sur Database
kubectl apply -f k8s/chaos/stress-db.yaml
```

---

## Alignement Mathematique

Le modele spatio-temporel suit la dynamique :

```
X(t+1) = f(A, X(t)) + B * U(t)
```

| Symbole | Correspondance PyTorch | Description |
|---------|----------------------|-------------|
| `X(t)` | `data.x[:, :, idx]` | Etat des 5 noeuds a l'instant t |
| `A` | `data.edge_index` | Matrice d'adjacence [2, 4] |
| `U(t)` | Trafic Locust entrant | Signal exogene |
| `f()` | GCN / GAT layer | Propagation spatiale (Message Passing) |

La perte **InfoNCE** consomme les fenetres glissantes :
- **Ancre (Z_anchor)** : fenetre `[t, t+W]` encode
- **Positif (Z_positive)** : fenetre augmentee ou `[t+1, t+W+1]` en regime sain
- **Negatif (Z_negative)** : fenetre echantillonnee pendant l'injection de chaos

---

## Nettoyage complet

```bash
# Supprimer toutes les experiences de chaos
kubectl delete networkchaos,stresschaos,podchaos --all -n rca-sandbox

# Desinstaller les charts Helm
helm uninstall chaos-mesh grafana loki prometheus jaeger -n rca-sandbox

# Supprimer l'application
kubectl delete -f k8s/app/ -n rca-sandbox

# Supprimer le namespace (supprime TOUT)
kubectl delete namespace rca-sandbox
```
