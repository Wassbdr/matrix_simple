# RCA Pipeline — Root Cause Analysis par Graphes Spatio-Temporels

Plateforme de recherche pour le **Root Cause Analysis (RCA)** dans les architectures microservices, combinant un **laboratoire Kubernetes instrumenté** (Chaos Engineering + Prometheus) et deux architectures de réseaux de neurones sur graphes complémentaires :

- **Late Bottleneck** (STGCN + GIB) — traite le graphe complet puis filtre via un masque Information Bottleneck, complexité O(N²)
- **STA** (Select-then-Analyze) — sélectionne d'abord k nœuds pertinents via un Straight-Through Estimator, puis analyse le sous-graphe, complexité O(N + k²)

Le pipeline supporte à la fois les **données réelles** (Prometheus + Chaos Mesh sur le cluster `observit-cluster1`) et les **données simulées** (topologies Barabási-Albert à N nœuds arbitraires, de 5 à 200+).

---

## Architecture du Laboratoire K8s (5 nœuds)

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
|  Observabilité : Prometheus | Jaeger (OTLP) | Loki | Grafana     |
|  Chaos Mesh    : NetworkChaos | StressChaos | PodChaos            |
+-------------------------------------------------------------------+
```

**Graphe de dépendances** :
`Frontend -> Auth`, `Frontend -> Backend`, `Backend -> Cache`, `Backend -> Database`

---

## Prérequis

- [kubectl](https://kubernetes.io/docs/tasks/tools/) configuré sur le contexte `observit-cluster1`
- [Helm](https://helm.sh/docs/intro/install/) v3+
- Python 3.9+ avec les dépendances listées dans [`requirements.txt`](requirements.txt) (notamment `torch>=2.0`, `torch_geometric>=2.4`, `numpy`, `pandas`, `scikit-learn`, `networkx`, `locust`, `matplotlib`)
- Accès RBAC suffisant pour créer des ressources dans un namespace dédié

---

## Structure du projet

```
matrix_simple/
├── k8s/
│   ├── namespace.yaml                       # Namespace rca-sandbox
│   ├── app/                                 # Manifestes K8s (5 services)
│   │   ├── db.yaml, cache.yaml, auth.yaml, backend.yaml, frontend.yaml
│   └── chaos/                               # Scénarios Chaos Mesh
│       ├── network-delay.yaml, network-delay-frontend-auth.yaml
│       ├── stress-db.yaml, stress-cpu-db.yaml, stress-mem-cache.yaml
├── rca_pipeline/
│   │── # ── Données ──
│   ├── dataset_builder.py                   # Orchestrateur Chaos Mesh + Extracteur Prometheus
│   ├── graph_builder.py                     # CSV Prometheus -> PyG Data (5 nœuds K8s)
│   ├── simulated_data_scalable.py           # Générateur BA(N,m) pour N quelconque
│   ├── simulated_data.py                    # Générateur simulé 20 nœuds (legacy)
│   ├── locustfile.py                        # Générateur de trafic (80% lecture, 20% écriture)
│   ├── loaders/                             # Loaders RCAEval + ObservIT (CSV ou dossier dataset_*.csv)
│   ├── tests/                               # Tests unitaires (pytest)
│   │── # ── Modèles ──
│   ├── model.py                             # Late Bottleneck : STGCN + GIB (O(N²))
│   ├── model_sta.py                         # STA : Select-then-Analyze (O(N+k²))
│   ├── model_late_gat.py                    # Variante Late-BN en GATv2
│   ├── model_sta_mm.py                      # Variante STA multimodale (metrics + logs/traces)
│   ├── model_late_gat_mm.py                 # Variante Late-BN GATv2 multimodale
│   ├── losses.py                            # InfoNCE + GIB sparsity asymétrique
│   ├── augment.py                           # Augmentation contrastive (bruit, jitter, masking)
│   ├── common.py                            # Constantes partagées, stratified_split()
│   │── # ── Entraînement ──
│   ├── train.py                             # Entraînement Late Bottleneck (2 phases)
│   ├── train_sta.py                         # Entraînement STA (2 phases + k-annealing)
│   │── # ── Évaluation ──
│   ├── benchmark.py                         # Benchmark unifié multi-échelle multi-seed
│   ├── benchmark_ablation.py                # Étude d'ablation (composants, k, protocole)
│   ├── benchmark_real.py                    # Benchmark sur RCAEval + ObservIT (STA/GATv2, options MM/drift)
│   │── # ── Outils ──
│   ├── infer.py                             # Inférence RCA (Late BN ou STA, tout N)
│   ├── evaluate.py                          # Évaluation détaillée (P/R/F1, Top-K, MRR)
│   ├── visualize.py                         # Visualisation dynamique (tout N)
│   ├── verify_tensors.py                    # Vérificateur de tenseurs
│   └── verify_training.py                   # Sanity check (overfit, gradients, mask)
├── scripts/
│   ├── 00-check-context.sh                  # Vérifie le contexte kubectl
│   ├── 02-deploy-observability.sh           # Helm : Prometheus, Jaeger, Loki, Grafana
│   ├── 03-deploy-app.sh                     # Déploie l'application 5-tiers
│   └── 04-install-chaos-mesh.sh             # Installe Chaos Mesh (scope rca-sandbox)
├── requirements.txt
├── docs/                                    # Rapports & audits (Week3, ObservIT, validations, etc.)
└── README.md
```

---

## Déploiement étape par étape

### 0. Vérifier le contexte

```bash
kubectl config use-context observit-cluster1
kubectl config current-context  # Doit afficher: observit-cluster1
```

### 1. Rendre les scripts exécutables

```bash
chmod +x scripts/*.sh
```

### 2. Déployer l'Observabilité

```bash
./scripts/02-deploy-observability.sh
```

| Composant | Chart Helm | Port |
|-----------|-----------|------|
| Prometheus | `prometheus-community/prometheus` | 9090 |
| Jaeger | `jaegertracing/jaeger` (all-in-one) | 16686 |
| Loki+Promtail | `grafana/loki-stack` | 3100 |
| Grafana | `grafana/grafana` | 3000 |

### 3. Déployer l'Application

```bash
./scripts/03-deploy-app.sh
kubectl apply -f k8s/app/auth.yaml
kubectl apply -f k8s/app/cache.yaml
kubectl rollout restart deployment frontend backend -n rca-sandbox
```

### 4. Installer Chaos Mesh

```bash
./scripts/04-install-chaos-mesh.sh
```

### 5. Vérifier l'état

```bash
kubectl get pods -n rca-sandbox
# Les 5 pods applicatifs doivent être Running :
# auth, backend, cache, database, frontend
```

---

## Accès aux services

```bash
kubectl port-forward svc/frontend -n rca-sandbox 8080:80       # Frontend
kubectl port-forward svc/grafana -n rca-sandbox 3000:80         # Grafana (admin/admin)
kubectl port-forward svc/rca-prometheus-server -n rca-sandbox 9090:80  # Prometheus
kubectl port-forward svc/rca-jaeger -n rca-sandbox 16686:16686 # Jaeger
```

---

## Pipeline RCA

Le pipeline supporte deux sources de données et deux architectures de modèles.

## Quickstart (local)

### 1) Setup Python

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2) Benchmark simulé (Barabási–Albert)

```bash
cd rca_pipeline
python3 benchmark.py --sizes 20 --seeds 42 --num-samples 400 --supervised-epochs 40 --late-contrastive-epochs 10 --baseline-epochs 40
```

### 3) Benchmark RCAEval (réel, si `rcaeval/` est présent)

```bash
.venv/bin/python rca_pipeline/benchmark_real.py --dataset trainticket --seeds 42 --epochs 30 --output-json rca_pipeline/benchmark_real_smoke.json
```

### Source A : Données réelles (Prometheus + Chaos Mesh)

```bash
# 1. Générer du trafic + injecter du chaos + extraire les métriques
python3 rca_pipeline/dataset_builder.py \
    --chaos-file k8s/chaos/stress-cpu-db.yaml \
    --locust-file rca_pipeline/locustfile.py \
    --frontend-url http://localhost:8080 \
    --users 30 --baseline-duration 300 --anomaly-duration 300 \
    --root-cause database --output-dir datasets/

# 2. Convertir CSV -> fenêtres PyG (topologie fixe 5 nœuds)
# - si tu as un seul run: passe le fichier `datasets/dataset_<run_id>.csv`
# - si tu as plusieurs runs: passe directement le dossier `datasets/` (fusion interne)
python3 -c "
from rca_pipeline.loaders.observit_loader import load_observit
windows, node_names, edge_index = load_observit('datasets/', window_size=12)
import torch
torch.save(windows, 'dataset_windows.pt')
print('saved:', len(windows), 'windows -> dataset_windows.pt')
"
```

### Source B : Données simulées (Barabási-Albert, N quelconque)

```bash
# Génère 3000 fenêtres sur un graphe BA(100, m=3)
python3 rca_pipeline/simulated_data_scalable.py --num-nodes 100 --num-samples 3000
```

Le générateur produit des topologies *scale-free* réalistes avec :
- Des nœuds nommés par rôle (db, cache, mq, svc, gw, api) selon la centralité
- Une propagation multi-hop via une matrice P construite sur le graphe dirigé
- Des anomalies : signal transitoire au nœud racine + cascade stochastique + distracteurs

### Entraînement

**Late Bottleneck** (STGCN + GIB) :

```bash
cd rca_pipeline && python3 train.py --dataset ../dataset_windows.pt \
    --epochs 100 --supervised-epochs 150 --hidden-dim 32 --beta-max 1.0
```

| Phase | Objectif | Perte |
|-------|---------|-------|
| Phase 1 — Contrastif | Séparer embeddings sains vs anomaux | `L_InfoNCE + β * L_GIB` |
| Phase 2 — Supervisé | Identifier le nœud cause racine | `L_CE` sur les logits GIB bruts |

**STA** (Select-then-Analyze) :

```bash
cd rca_pipeline && python3 train_sta.py --num-nodes 100 --k 10 \
    --supervised-epochs 150 --hidden-dim 32
```

---

## Évaluation RCAEval (données réelles) — `benchmark_real.py`

Le script [`rca_pipeline/benchmark_real.py`](rca_pipeline/benchmark_real.py) évalue **STA** et **GATv2** sur les datasets RCAEval (TrainTicket, SockShop, OnlineBoutique…) avec les métriques **Avg@k, MRR, NDCG**.

### Correctif important (STA réel)

Sur RCAEval réel, le warmup contrastif STA peut dégrader fortement les scores. Le benchmark expose donc :
- `--sta-contrastive-warmup` (**défaut: 0** pour le réel)

Exemple (campagne longue comparable) :

```bash
.venv/bin/python rca_pipeline/benchmark_real.py \
  --dataset all \
  --seeds 42,123,7 \
  --epochs 100 \
  --max-cases 90 \
  --sta-contrastive-warmup 0 \
  --output-json rca_pipeline/benchmark_real_run.json
```

### Validation protocolaire (presets `smoke` / `long`)

Le script [`rca_pipeline/validation_protocol.py`](rca_pipeline/validation_protocol.py) lance une petite matrice de configs et écrit les JSON dans `rca_pipeline/validation_runs/`.

Variables d’environnement :
- `RCA_VAL_PRESET=smoke|long` (défaut: `smoke`)
- `RCA_VAL_STA_WARMUP=0` (défaut: `0`)
- `RCA_VAL_DRIFT=0|1` (défaut: `1`)

Exemple :

```bash
RCA_VAL_PRESET=long RCA_VAL_STA_WARMUP=0 RCA_VAL_DRIFT=0 \
  .venv/bin/python rca_pipeline/validation_protocol.py
```

---

## ObservIT (cluster K8s) — collecte + dataset réel + benchmark

Le flux ObservIT complète RCAEval avec des runs contrôlés sur cluster (Chaos Mesh + Locust + Prometheus).

- **Collecte** : `rca_pipeline/dataset_builder.py` produit `dataset_<run_id>.csv`
- **Conversion + benchmark** : `rca_pipeline/benchmark_real.py --observit-csv <csv_ou_dossier>`

Doc dédiée (schéma colonnes, commandes, dépannage) :
- [`docs/observit_dataset.md`](docs/observit_dataset.md)

Scripts pratiques :
- `scripts/run_observit_dataset.sh`
- `scripts/run_observit_benchmark.sh`
- `scripts/run_observit_sota_campaign.sh`

---

## Résultats & rapports

- Compte rendu semaine 3 : [`docs/Week3.md`](docs/Week3.md)
- Validation (v1 vs v2, fix STA warmup=0, décisions) : [`docs/validation_comparison_v1_v2.md`](docs/validation_comparison_v1_v2.md)
- Audit stage + roadmap (STA vs Late-Bottleneck) : [`docs/stage_audit_sta_latebn.md`](docs/stage_audit_sta_latebn.md)

| Phase | Objectif | Perte |
|-------|---------|-------|
| Phase 1 — Warmup contrastif | Initialiser le LightEncoder | `L_InfoNCE` (20 epochs) |
| Phase 2 — Supervisé + k-annealing | CE sur selector + scorer, k décroît de N/2 → k_target | `L_CE_selector + L_CE_scorer` |

### Benchmark multi-échelle

```bash
cd rca_pipeline && python3 benchmark.py
```

### Paramétrage CLI du benchmark

Le script `benchmark.py` est maintenant paramétrable en ligne de commande.

Commande par défaut (équivalente à la config historique):

```bash
cd rca_pipeline && python3 benchmark.py
```

Valeurs par défaut:
- `--sizes 20,50,100`
- `--ba-m-map 20:2,50:3,100:3`
- `--k-map 20:5,50:8,100:10`
- `--num-samples 3000`
- `--window-size 12`
- `--num-features 5`
- `--anomaly-ratio 0.4`
- `--dataset-seed 42`
- `--seeds 42,123,7`
- `--hidden-dim 32`
- `--batch-size 256`
- `--supervised-epochs 150`
- `--late-contrastive-epochs 80`
- `--sta-warmup-epochs 20`
- `--sta-k-anneal-epochs 60`
- `--baseline-epochs 150`

Exemple personnalisé:

```bash
cd rca_pipeline && python3 benchmark.py \
    --sizes 20,100 \
    --ba-m-map 20:2,100:3 \
    --k-map 20:5,100:15 \
    --num-samples 2000 \
    --seeds 42,123 \
    --supervised-epochs 120 \
    --late-contrastive-epochs 60 \
    --sta-warmup-epochs 10 \
    --sta-k-anneal-epochs 40
```

Compare 5 modèles (STA, Late Bottleneck, Full-Graph GCN, MLP, Random Forest) sur 3 échelles (N=20/50/100) avec 3 graines aléatoires. Résultat type :

| N | STA Top-1 | Late BN Top-1 | GCN Top-1 | RF Top-1 | MLP Top-1 |
|---|-----------|---------------|-----------|----------|-----------|
| 20 | 97.9% ± 0.7% | 96.6% ± 1.4% | 97.3% ± 0.5% | 59.3% | 6.7% |
| 50 | 93.7% ± 1.8% | 95.4% ± 0.8% | 94.4% ± 3.2% | 17.7% | 1.9% |
| 100 | 90.2% ± 2.1% | 90.2% ± 9.0% | 94.1% ± 1.1% | 7.0% | 1.7% |

### Étude d'ablation

```bash
cd rca_pipeline && python3 benchmark_ablation.py
```

Quatre axes d'ablation :
- **A — Décomposition par composant** : Late BN vs STGCN-seul vs Spatial-only vs Temporal-only
- **B — Protocole Late BN** : contrastif+supervisé vs supervisé seul
- **C — Sensibilité au k de STA** : k ∈ {10, 15, 20, 30} pour N=100
- **D — Protocole STA** : warmup+k-annealing vs sans warmup vs sans k-annealing

**Résultats clés de l'ablation :**

| Axe | Résultat principal |
|-----|-------------------|
| A (composants) | Temporal-only = 98.4% à N=20 mais chute à 64.6% à N=100. Spatial-only s'effondre partout (<20%). GIB dégrade : Late BN = 61.8% ± 43% vs GCN = 94.1% ± 1.1% à N=100. |
| B (protocole Late BN) | Contrastif marginal à N=20 (+0.4%) mais **critique à N=100** (+29.9%). |
| C (k-sensibilité STA) | k=10 trop agressif. k=15 → +2.7%, k=30 → 93.8% ± 0.3% (proche du GCN pur). Params constants (16 549). |
| D (protocole STA) | Warmup et k-annealing **indispensables** : sans eux, effondrement à ~62% ± 43%. |

### Inférence et évaluation

```bash
# Inférence (supporte --arch late ou --arch sta, tout N)
cd rca_pipeline && python3 infer.py --checkpoint checkpoints/sta_semi_sup.pt \
    --dataset ../dataset_windows.pt --arch sta --num-windows 10

# Évaluation détaillée (P/R/F1 par classe, Top-K, MRR)
cd rca_pipeline && python3 evaluate.py --checkpoint checkpoints/sta_semi_sup.pt \
    --dataset ../dataset_windows.pt --arch sta

# Visualisation (layout dynamique pour tout N)
cd rca_pipeline && python3 visualize.py --checkpoint checkpoints/sta_semi_sup.pt \
    --dataset ../dataset_windows.pt --arch sta --output graph_heatmap.png
```

---

## Les deux architectures

### Late Bottleneck (model.py) — O(N²)

```
x [B, N, F, W]
  → TemporalConvBlock (Conv1d × 2 + attention temporelle)  [B*N, H]
  → SpatialConvBlock (GCNConv × 2 + résiduel)              [B*N, H]
  → GraphInformationBottleneck (MLP → Bernoulli relaxé → masque M)
  → z_graph = sum(M * H, dim=nodes)                        [B, H]
```
Le masque GIB isole le sous-graphe causal *après* traitement complet. Simple et robuste sur petits graphes.

### STA (model_sta.py) — O(N + k²)

```
x [B, N, F, W]
  → LightEncoder (Conv1d + GCN léger)                      [B, N, H]   O(N)
  → STESelector (Gumbel-sigmoid + top-k + STE)             [B, N] mask  O(N)
  → extract_subgraph_vectorized (pyg_subgraph)             [B*k, F, W]  O(E)
  → DeepTemporalBlock + DeepSpatialBlock (sur k nœuds)     [B, k, H]   O(k²)
  → SubgraphScorer (attention + contexte)                   [B, k]      O(k)
```
Le nombre de paramètres reste **constant** (~16 500) quelle que soit N. Conçu pour le passage à très grande échelle.

---

## Scénarios de Chaos

| Scénario | Fichier YAML | Cible | Effet |
|----------|--------------|-------|-------|
| Latence Auth | `network-delay-frontend-auth.yaml` | Frontend → Auth | 500ms + 5% perte |
| Stress CPU DB | `stress-cpu-db.yaml` | Database | 4 workers CPU 100% |
| Latence Backend | `network-delay.yaml` | Frontend → Backend | 500ms + 5% perte |
| Stress + Kill DB | `stress-db.yaml` | Database | CPU 100% + PodKill |
| Stress mémoire Cache | `stress-mem-cache.yaml` | Cache (Redis) | Saturation mémoire |

---

## Alignement mathématique

Le modèle spatio-temporel suit la dynamique :

```
X(t+1) = f_θ(A, X(t)) + B · U(t)
```

| Symbole | Correspondance | Description |
|---------|---------------|-------------|
| X(t) | `data.x[:, :, t]` | État des N nœuds à l'instant t |
| A | `data.edge_index` | Matrice d'adjacence (topologie K8s ou BA) |
| U(t) | Anomalie injectée | Signal exogène (trafic normal + perturbation chaos) |
| f_θ | STGCN layers | Propagation spatio-temporelle apprise |
| M | Masque GIB / STE | Isolation du sous-graphe causal |

**Objectif RCA** : trouver le support minimal de U(t) expliquant la dérive observée :
```
min_U ||U||₀  sous contrainte  Y(t) ≈ Ŷ(t)
```

En pratique, le masque M (GIB ou STE) est un proxy différentiable de cette contrainte L₀.

---

## Nettoyage complet

```bash
kubectl delete networkchaos,stresschaos,podchaos --all -n rca-sandbox
helm uninstall chaos-mesh grafana loki prometheus jaeger -n rca-sandbox
kubectl delete -f k8s/app/ -n rca-sandbox
kubectl delete namespace rca-sandbox
```
