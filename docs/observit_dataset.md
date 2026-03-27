# Dataset réel cluster observ-it

Ce flux complète **RCAEval** : tu contrôles l’injection (Chaos Mesh), la charge (Locust) et le **label** `--root-cause` aligné sur les 5 services du graphe fixe (`frontend`, `auth`, `backend`, `cache`, `database`).

## 1. Collecte (cluster)

Prérequis : contexte `observit-cluster1`, namespace `rca-sandbox`, Prometheus joignable (souvent `kubectl port-forward` vers le service Prometheus du README).

```bash
cd rca_pipeline
python3 dataset_builder.py \
  --prometheus-url http://localhost:9090 \
  --chaos-file ../k8s/chaos/stress-cpu-db.yaml \
  --locust-file locustfile.py \
  --frontend-url http://localhost:8080 \
  --users 30 \
  --baseline-duration 300 \
  --anomaly-duration 300 \
  --root-cause database \
  --output-dir ../datasets/observit_runs/
```

Répète pour plusieurs scénarios (`stress-mem-cache.yaml`, etc.) en adaptant `--root-cause` au service cible du YAML Chaos.

Les CSV produits sont nommés `dataset_<run_id>.csv`.

## 2. Format attendu

Colonnes minimales : `timestamp`, `pod`, `cpu_usage`, `ram_usage`, `network_tx`, `network_rx`, `error_rate`, `label`, `root_cause` — identique à la sortie actuelle de `dataset_builder.py`.

Colonne optionnelle supportée : `latency_p99` (si le CSV vient d’un connecteur live type `prometheus_connector.py`).

## 3. Benchmark STA vs GATv2 sur ce jeu

Depuis `rca_pipeline`, en pointant vers **un fichier** ou **un répertoire** contenant plusieurs `dataset_*.csv` (fusion automatique vers `_merged_observit.csv` dans le dossier) :

```bash
python3 benchmark_real.py \
  --observit-csv ../datasets/observit_runs/ \
  --observit-window 12 \
  --seeds 42,123,7 \
  --epochs 100 \
  --output-json ../datasets/observit_benchmark.json
```

- La normalisation **StandardScaler** est **fit sur le train uniquement** puis appliquée au test (même logique que `graph_builder` / rapport Week 2).
- **Multimodal** (`--use-mm`) : non fourni par ce CSV Prometheus seul ; laisser désactivé pour l’instant.

## 4. Volume minimal

Le benchmark ignore les fenêtres sans `root_cause` (période saine). Il faut **assez de fenêtres anomales** par cause pour le split stratifié (plusieurs expériences ou durées longues). En pratique : viser **au moins quelques dizaines de fenêtres** étiquetées au total, réparties sur plusieurs causes.

## 5. Fichiers code

| Rôle | Fichier |
|------|---------|
| Orchestration chaos + export CSV | [`rca_pipeline/dataset_builder.py`](../rca_pipeline/dataset_builder.py) |
| CSV → fenêtres PyG `[N,F,W]` | [`rca_pipeline/graph_builder.py`](../rca_pipeline/graph_builder.py) |
| Chargement fusion + filtre labels | [`rca_pipeline/loaders/observit_loader.py`](../rca_pipeline/loaders/observit_loader.py) |
| Évaluation STA / GATv2 | [`rca_pipeline/benchmark_real.py`](../rca_pipeline/benchmark_real.py) (`--observit-csv`) |

## 6. Dépannage (collecte réelle)

- **`503` sur `/api/v1/query_range`** : le pod `rca-prometheus-server` peut être **Ready=false** (conteneur `prometheus-server` en restart / readiness). Tant que Prometheus répond 503, **`dataset_builder` ne pourra pas écrire de CSV**. À traiter côté cluster (ressources, stockage, config Helm).
- **Port-forward** : en fin d’expérience, le script relance un **`kubectl port-forward`** vers `svc/rca-prometheus-server` si l’URL est `http://127.0.0.1:9090` et que la sonde échoue.
- **Pipeline sans cluster** : générer un CSV compatible en respectant strictement le schéma ci-dessus, puis lancer `benchmark_real.py --observit-csv <csv_ou_dossier>`.

Scripts pratiques (venv `.venv` à la racine du repo) :

- [`scripts/run_observit_dataset.sh`](../scripts/run_observit_dataset.sh) — lance `dataset_builder.py` avec les flags par défaut du README.
- [`scripts/run_observit_benchmark.sh`](../scripts/run_observit_benchmark.sh) — lance `benchmark_real.py` (passer `--observit-csv`, etc.).
