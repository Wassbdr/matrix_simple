# Notes sur le protocole historique

Pour installer le produit actuel et utiliser le modèle RE1, suivre le [README](README.md)
et le [tutoriel](docs/tutorial.md). Les sections ci-dessous décrivent le protocole de recherche
et le déploiement du modèle précédent.

Ce document est la porte d'entrée. Il répond à trois questions dans l'ordre :
**qu'est-ce que ce dépôt**, **comment le faire tourner**, **où sont les résultats**.

- Vue d'ensemble scientifique et tables de résultats → [`README.md`](README.md)
- Carte du code et index des documents → [`docs/README.md`](docs/README.md)
- Opérer le stack déployé → [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- Contribuer (conventions, garde-fous) → [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 1. Ce qu'est ce dépôt

Un pipeline de **Root Cause Analysis** pour microservices, qui compare deux
architectures GNN spatio-temporelles :

- **Late-Bottleneck STGCN + GIB** - analyse du graphe complet, coût O(N²)
- **STA (Select-Then-Analyze)** - sélection différentiable de k nœuds puis
  analyse locale, coût O(N + k²) dans le cas dense. La latence augmente avec N.

Évalué sur quatre terrains : simulé (Barabási-Albert, N = 20/50/100), RCAEval
(TrainTicket, SockShop, Online Boutique), un cluster Kubernetes réel
(Sock Shop + Chaos Mesh), et un déploiement synthétique à 50 services.

L'étude est **délibérément honnête** : aucune revendication SOTA, un résultat
négatif assumé sur la fusion multimodale, et une baseline BARO re-jouée
isopérimétriquement qui **bat le modèle sur Online Boutique**. Ne pas retirer ces
nuances du README, elles sont le cœur de la contribution.

Un second projet du même stage, `ewat`, traite le problème **complémentaire** :
l'early warning et le typage d'anomalies *avant* la panne, là où la RCA est
post-mortem. Les deux dépôts sont indépendants.

## 2. Prérequis

| | |
|---|---|
| Python | ≥ 3.9 (le poste d'origine utilise 3.12) |
| Disque | ~6 Go pour l'environnement virtuel, ~39 Go de plus si vous voulez les données RCAEval |
| PyTorch | ≥ 2.0 avec un `torch_geometric` ≥ 2.4 compatible |
| Optionnel | `kubectl`, `helm`, Chaos Mesh en mode cluster-scoped - uniquement pour recollecter des données |

Aucun accès cluster n'est nécessaire pour lire les résultats ni pour rejouer le
benchmark simulé.

## 3. Installation et vérification

```bash
git clone <url> matrix_simple && cd matrix_simple
python3 -m venv .venv && source .venv/bin/activate
make dev                      # pip install -e ".[dev]" + hooks pre-commit
make test                     # rca_pipeline/tests + serving/tests
make lint
```

`pyproject.toml` est la **source unique** des dépendances : il n'y a pas de
`requirements.txt` racine à synchroniser (`requirements/constraints-cpu.txt` sert
uniquement à épingler une install CPU-only, et `serving/requirements*.txt` aux
images Docker du service).

## 4. Trois parcours selon ce que vous voulez faire

### a. Lire les résultats sans rien exécuter

Tout est dans [`README.md`](README.md) (tables) et
[`results_transmission/`](results_transmission/) (JSON par graine, bootstrap,
hashes). Commencez par [`docs/COMPARABILITY_MATRIX.md`](docs/COMPARABILITY_MATRIX.md) :
il dit à quelles baselines les chiffres sont comparables et à quelles conditions.

Vérifier que rien n'a bougé :

```bash
# Attention : les deux manifestes n'ont pas la même convention de chemin.
# paper_benchmark*.sha256 liste des noms nus → se lance depuis results_transmission/
(cd results_transmission && sha256sum -c paper_benchmark.sha256)
(cd results_transmission && sha256sum -c paper_benchmark_10seed.sha256)
# canonical_results_manifest.json.sha256 liste un chemin depuis la racine
sha256sum -c results_transmission/canonical_results_manifest.json.sha256
```

Ne citez que les `paper_benchmark_*.json`. Les `rca_pipeline/*_prepublish.json`
sont des artefacts legacy à faible nombre d'epochs, dont les champs
`reference_top1` sont périmés.

### b. Reproduire ce qui ne demande aucune donnée externe

Le benchmark simulé génère ses propres graphes Barabási-Albert :

```bash
make benchmark-smoke                                   # ~2 min, 1 graine, N=20
python -m rca_pipeline.benchmark --output-json /tmp/bench_sim.json   # complet
python -m rca_pipeline.benchmark_ablation --output-json /tmp/abl.json
```

Le harnais de reproductibilité expose trois presets
(`rca_pipeline/validation_protocol.py`) :

| Preset | Graines | Epochs | Cas | Usage |
|---|---|---:|---:|---|
| `smoke` | 42, 123 | 20 | 20 | sanity CI (< 5 min) |
| `long` | 42,123,7,1337,2025 | 100 | tous | run de référence |
| `paper` | 10 graines | 100 | tous | puissance statistique du papier |

```bash
RCA_VAL_PRESET=smoke RCA_VAL_STA_WARMUP=0 python -m rca_pipeline.validation_protocol
```

Sorties dans `rca_pipeline/validation_runs/` (gitignoré).

### c. Reproduire les benchmarks réels

Nécessite les données RCAEval, **non versionnées** (~39 Go) : suivre les
instructions de [RCAEval](https://github.com/NetManAIOps/RCAEval) pour
télécharger les CSV bruts, puis les placer sous `rcaeval/data/`. Le parsing et la
construction du graphe sont pris en charge par
`rca_pipeline/loaders/rcaeval_loader.py`.

```bash
python -m rca_pipeline.benchmark_real \
    --config configs/benchmark_real.yaml \
    --output-json /tmp/bench_real.json
```

> **Garde-fou** : `--sta-contrastive-warmup 0` est obligatoire sur données
> réelles. Avec `warmup = 10`, TrainTicket Avg@1 passe de 73,3 % à **0,0 %**. La
> config le refuse au chargement ; ne contournez pas.

Pour recollecter des données depuis un cluster (Prometheus + Loki + Jaeger +
Chaos Mesh cluster-scoped), les scripts de bootstrap sont dans `scripts/`
(`02-deploy-observability.sh` → `05-enable-chaos-mesh-cluster-scope.sh`, puis
`run_sockshop_dataset.sh`). Manifests de chaos dans `k8s/chaos/sockshop/`.

## 5. État du stack MLOps

Le cycle de vie complet a été **validé de bout en bout sur un cluster réel** : un
champion 5 graines (Avg@1 0,786 / MRR 0,871) a été entraîné, enregistré, promu,
puis récupéré par alias par l'initContainer du service.

```
HPO → train_and_persist → MLflow registry (sta-sockshop@challenger)
                              → promote.py (porte manuelle) → @champion
                              → serve (FastAPI) + watcher + drift CronJob
```

Le runbook opérationnel - y compris les pièges rencontrés (basic-auth MLflow,
récupération par alias, MLflow OOMKilled à 512 Mi, StorageClass) - est dans
[`docs/OPERATIONS.md`](docs/OPERATIONS.md). Les manifests sont sous `deploy/`
(MLflow, chart Helm `rca-stack`, dashboard Grafana) et `serving/`.

Le dépôt ne contient **que** les manifests et le runbook, pas d'état vivant :
redéployer suppose un cluster fournissant Prometheus, Loki, Jaeger et Chaos Mesh
en mode cluster-scoped, plus un MLflow accessible. Vérifiez la disponibilité du
cluster de stage avant de partir du principe que le stack tourne encore.

## 6. Ce dont vous n'héritez pas

| Manquant | Pourquoi | Comment l'obtenir |
|---|---|---|
| `rcaeval/` (~39 Go) | Données externes, jamais versionnées | Télécharger depuis le dépôt RCAEval upstream |
| `datasets/`, `checkpoints/`, `*.pt` | Artefacts régénérables et volumineux | Rejouer l'entraînement, ou `train_and_persist` |
| `docs/internal/` | Notes personnelles de stage | Non transmis |
| Accès au cluster ObservIT | Fin de stage | - |
| PDF compilés du papier / rapport | Intermédiaires LaTeX gitignorés | `make paper`, `make report` |

## 7. Limites connues

À lire avant d'exploiter ou d'étendre les résultats - elles sont détaillées dans
le README (§ « Reproducibility caveats ») :

- Les tables principales sont à **5 graines**, où le bootstrap au niveau graine
  n'est pas jugé fiable (`seed_level_bootstrap_reliable: false`) ; les ±
  rapportés sont des **±1σ inter-graines**, pas des IC 95 %. Le jeu à 10 graines
  existe et est publié : l'avance de STA à N=100 y **retombe de +6,2 à +2,4 pp**.
- SockShop multimodal est **skippé, pas échoué** : la release publique RCAEval a
  0 % de couverture de traces, et `--mm-strict` refuse de substituer des zéros.
- Le déterminisme est **par graine sur une machine donnée**, pas bit-à-bit entre
  machines : ±0,5 pp possibles selon les versions CUDA/PyTorch.
- Aucune revendication SOTA uniforme face à BARO. Voir la matrice de
  comparabilité.

## 8. Suites possibles

Les expériences post-soutenance déjà menées (cascade prior, ensemble STA+GATv2,
saturation du GIB) sont documentées dans
[`docs/ANALYSIS_MODEL_IMPROVEMENTS.md`](docs/ANALYSIS_MODEL_IMPROVEMENTS.md),
avec ce qui a marché et ce qui n'a rien apporté. C'est le meilleur point de
départ pour savoir quelles pistes sont déjà épuisées.
