# Audit datasets RCA -> reproduction ObservIT

## 1) Matrice datasets / loaders / schéma / split

| Dataset | Source | Loader / Builder | Schéma principal | Split | Run |
|---|---|---|---|---|---|
| Simulé BA(N,m) | Génération synthétique | `rca_pipeline/simulated_data_scalable.py` (`generate_scalable_dataset`) | PyG `Data.x [N,F,W]`, `root_cause`, `y`, optionnel `x_logs [N,8]`, `x_traces [N,6]` | Stratifié par root cause (train/test) dans benchmark | `python rca_pipeline/benchmark.py` |
| RCAEval réel | Datasets RCAEval | `rca_pipeline/loaders/rcaeval_loader.py` | Cases -> PyG `Data` depuis `metrics.csv/simple_metrics.csv`, root cause inférée du dossier | Stratifié par root cause dans `benchmark_real.py` | `python rca_pipeline/benchmark_real.py --dataset <name>` |
| ObservIT réel | CSV Prometheus + Chaos + Locust | `rca_pipeline/dataset_builder.py` + `rca_pipeline/graph_builder.py` + `rca_pipeline/loaders/observit_loader.py` | CSV: `timestamp,pod,cpu_usage,ram_usage,network_tx,network_rx,error_rate,label,root_cause` (+ `latency_p99` optionnel), puis PyG fenêtres `[N,F,W]` | Stratifié par root cause dans `benchmark_real.py` + scaler fit train-only | `python rca_pipeline/benchmark_real.py --observit-csv <csv|dir>` |

## 2) Points de contrôle avant benchmark réel ObservIT

- Vérifier le mapping pods->services (`frontend/auth/backend/cache/database`) dans les logs de `GraphBuilder`.
- Vérifier qu'il y a des fenêtres anomales labelisées sur plusieurs causes (pas seulement une cause).
- Vérifier la couverture classe train/test après split (warnings affichés par `benchmark_real.py`).
- Vérifier que la normalisation est activée sur train only (comportement par défaut côté ObservIT).

## 3) Procédure reproductible ObservIT (minimum)

1. Collecte:
   - `python rca_pipeline/dataset_builder.py --chaos-file ... --root-cause <service> --output-dir datasets/observit_runs/`
2. Répéter pour plusieurs scénarios/causes.
3. Benchmark:
   - `python rca_pipeline/benchmark_real.py --observit-csv datasets/observit_runs/ --observit-window 12 --seeds 42,123,7 --epochs 100 --output-json datasets/observit_benchmark.json`

## 4) Risques résiduels

- Classes rares: une cause avec un seul sample reste non évaluable en test.
- Mapping service: toute convention de nommage pod hors pattern actuel réduit la couverture.
- Domaine: les métriques en prod réelle (sampling/scrape interval) peuvent dégrader la stabilité sans recalibrage des fenêtres.
