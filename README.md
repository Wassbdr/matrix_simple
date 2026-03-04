# RCA Sandbox — Environnement de Laboratoire

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
│   ├── namespace.yaml                      # Namespace rca-sandbox
│   ├── app/
│   │   ├── db.yaml                         # PostgreSQL (Deployment + Service + PVC)
│   │   ├── cache.yaml                      # Redis (Deployment + Service)
│   │   ├── auth.yaml                       # FastAPI Auth (ConfigMap + Deployment + Service)
│   │   ├── backend.yaml                    # FastAPI Backend (ConfigMap + Deployment + Service)
│   │   └── frontend.yaml                   # Nginx (ConfigMap + Deployment + Service)
│   └── chaos/
│       ├── network-delay.yaml              # NetworkChaos : latence Frontend <-> Backend
│       ├── network-delay-frontend-auth.yaml # NetworkChaos : latence Frontend <-> Auth
│       ├── stress-db.yaml                  # StressChaos (CPU) + PodChaos (kill) Database
│       ├── stress-cpu-db.yaml              # StressChaos isole (CPU) Database
│       └── pod-kill-cache.yaml             # PodChaos : kill Cache (Redis)
├── rca_pipeline/
│   ├── locustfile.py                       # Generateur de trafic (80% lecture, 20% ecriture)
│   ├── dataset_builder.py                  # Orchestrateur Chaos Mesh + Extracteur Prometheus
│   ├── graph_builder.py                    # Convertisseur CSV -> PyTorch Geometric Data
│   ├── verify_tensors.py                   # Verificateur de tenseurs (dimensions, NaN, scaling)
│   ├── model.py                            # STGCN + GIB + RootCauseClassifier
│   ├── losses.py                           # InfoNCE + GIB sparsity + root cause CE loss
│   ├── train.py                            # Entrainement 2 phases (contrastif + supervise)
│   ├── verify_training.py                  # Sanity check (overfit + gradients + mask collapse)
│   ├── benchmark.py                        # Benchmark multi-modeles (STGCN, GCN, RF)
│   ├── baselines.py                        # Modeles baseline (PureGCN, RandomForest)
│   ├── infer.py                            # Inference RCA (classificateur + vote majoritaire)
│   ├── evaluate.py                         # Evaluation (Precision/Recall/F1 par classe)
│   ├── visualize.py                        # Double visualisation K8S + prediction RCA
│   └── augment.py                          # Augmentation contrastive (bruit, jitter, masking)
├── scripts/
│   ├── 00-check-context.sh                 # Verifie le contexte kubectl
│   ├── 02-deploy-observability.sh          # Helm : Prometheus, Jaeger, Loki, Grafana
│   ├── 03-deploy-app.sh                    # Deploie l'application 5-tiers
│   └── 04-install-chaos-mesh.sh            # Installe Chaos Mesh (scope rca-sandbox)
├── requirements.txt                        # Dependances Python
├── .gitignore                              # Exclusions Git
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
2. Attend N secondes (baseline saine)
3. Applique un fichier Chaos Mesh via `kubectl apply -f`
4. Attend N secondes (periode d'anomalie)
5. Supprime le chaos via `kubectl delete`
6. Interroge l'API Prometheus pour extraire les metriques (step=2s)
7. Exporte en CSV avec colonnes `label` (0/1) et `root_cause`

**Metriques PromQL extraites (5)** : CPU usage, RAM usage, Network TX/RX, Error rate (pod restarts).

```bash
# Exemple : scenario PodKill Cache avec labeling
python3 rca_pipeline/dataset_builder.py \
    --chaos-file k8s/chaos/pod-kill-cache.yaml \
    --locust-file rca_pipeline/locustfile.py \
    --frontend-url http://localhost:8080 \
    --users 30 \
    --baseline-duration 300 \
    --anomaly-duration 300 \
    --root-cause cache \
    --output-dir datasets/
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
builder = GraphBuilder('datasets/dataset_merged.csv')
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

### Etape 5 : Entrainement du modele (train.py)

Le modele `SpatioTemporalRCA` (defini dans `model.py`) enchaine :
1. **Convolution temporelle** (Conv1d x2) : reduit la dimension W en vecteur latent
2. **Convolution spatiale** (GCNConv x2) : message passing sur le graphe
3. **Graph Information Bottleneck** : MLP + Bernoulli relaxe -> masque sparse M
4. **Sortie** : Z = M * H (representation compressee)
5. **Classificateur** : `RootCauseClassifier` predit quel noeud est la cause racine

**Entrainement en 2 phases :**

| Phase | Objectif | Perte | Epochs |
|-------|---------|-------|--------|
| **Phase 1** — Pre-entrainement contrastif | Separer embeddings sains vs anomaux | `L_InfoNCE + β * L_GIB` | 100 |
| **Phase 2** — Fine-tuning supervise | Identifier le noeud cause racine | `L_CE + 0.1 * L_sparsity` | 150 |

Phase 1 utilise un annealing lineaire de β de 0 a `beta_max` sur `warmup_epochs`.
Phase 2 entraine le `RootCauseClassifier` sur les fenetres anomales etiquetees.

```bash
cd rca_pipeline && python3 train.py \
    --dataset ../dataset_windows.pt \
    --epochs 100 \
    --supervised-epochs 150 \
    --hidden-dim 32 \
    --beta-max 1.0 \
    --warmup-epochs 20
```

### Etape 6 : Benchmark (benchmark.py)

Compare 3 modeles sur un split stratifie (toutes les causes racines dans train et test) :

```bash
cd rca_pipeline && python3 benchmark.py \
    --dataset ../dataset_windows.pt \
    --checkpoint checkpoints/stgcn_rca_final.pt
```

**Resultat type :**

```
================================================================================
MODEL                | PRECISION       | RECALL          | F1-SCORE       
--------------------------------------------------------------------------------
STGCN + GIB          | 1.0000          | 1.0000          | 1.0000         
Pure GCN (Supervised) | 0.5000          | 0.6667          | 0.5556         
Random Forest        | 1.0000          | 1.0000          | 1.0000         
================================================================================
```

### Etape 7 : Evaluation detaillee (evaluate.py)

Rapport par classe (precision, recall, F1 pour chaque cause racine) :

```bash
cd rca_pipeline && python3 evaluate.py \
    --checkpoint checkpoints/stgcn_rca_final.pt \
    --dataset ../dataset_windows.pt
```

### Etape 8 : Inference RCA (infer.py)

Charge un checkpoint et identifie la cause racine via le classificateur :
- Vote majoritaire sur N fenetres anomales pour le verdict final

```bash
cd rca_pipeline && python3 infer.py \
    --checkpoint checkpoints/stgcn_rca_final.pt \
    --dataset ../dataset_windows.pt \
    --num-windows 10
```

### Etape 9 : Visualisation (visualize.py)

Double visualisation cote-a-cote :
- **Panneau gauche** : topologie Kubernetes (services, technologies, connexions)
- **Panneau droit** : prediction du modele (heatmap des probabilites de cause racine)
- **Barre inferieure** : comparaison cause reelle vs prediction

```bash
cd rca_pipeline && python3 visualize.py \
    --checkpoint checkpoints/stgcn_rca_final.pt \
    --dataset ../dataset_windows.pt \
    --output graph_heatmap.png
```

---

## Data Augmentation Contrastive (augment.py)

4 strategies composables pour generer des positifs diversifies :

| Strategie | Fonction | Description |
|-----------|----------|-------------|
| Bruit gaussien | `add_gaussian_noise(x, sigma)` | Perturbation isotrope des features |
| Jitter temporel | `temporal_jitter(x, max_shift)` | Decalage circulaire sur l'axe temps |
| Masquage features | `feature_masking(x, mask_ratio)` | Mise a zero aleatoire de canaux |
| Edge dropout | `edge_dropout(edge_index, drop_ratio)` | Suppression d'aretes du graphe |

```python
from augment import augment_positive
x_positive_aug = augment_positive(x, noise_sigma=0.1, jitter_shift=1, feature_mask_ratio=0.2)
```

---

## Scenarios de Chaos

| Scenario | Fichier YAML | Cible | Effet |
|----------|--------------|-------|-------|
| Latence Auth | `network-delay-frontend-auth.yaml` | Frontend -> Auth | 500ms + 5% perte |
| Stress CPU DB | `stress-cpu-db.yaml` | Database | 4 workers CPU 100% |
| PodKill Cache | `pod-kill-cache.yaml` | Cache (Redis) | Kill immediat |
| Latence Backend | `network-delay.yaml` | Frontend -> Backend | 500ms + 5% perte |
| Stress + Kill DB | `stress-db.yaml` | Database | CPU 100% + PodKill |

```bash
# Appliquer un scenario
kubectl apply -f k8s/chaos/pod-kill-cache.yaml
# Annuler
kubectl delete -f k8s/chaos/pod-kill-cache.yaml
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
| `M` | GIB mask probabilities | Isolation du sous-graphe causal |
| `C()` | `RootCauseClassifier` | Prediction supervisee de la cause racine |

**Entrainement en 2 phases :**

Phase 1 — La perte **InfoNCE** consomme les fenetres glissantes :
- **Ancre (Z_anchor)** : fenetre `[t, t+W]` encode
- **Positif (Z_positive)** : fenetre augmentee ou `[t+1, t+W+1]` en regime sain
- **Negatif (Z_negative)** : fenetre echantillonnee pendant l'injection de chaos

Phase 2 — La perte **Cross-Entropy** entraine le classificateur `C()` sur les fenetres anomales pour predire quel noeud est responsable de l'anomalie.

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
