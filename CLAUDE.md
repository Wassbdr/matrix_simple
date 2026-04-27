# matrix_simple — RCA Pipeline

Pipeline de Root Cause Analysis pour microservices. Compare STA (Select-then-Analyze) vs Late-Bottleneck STGCN+GIB, variantes GATv2, multimodal (metrics+logs+traces). Benchmarks: simulé (Barabasi-Albert), RCAEval réel, ObservIT K8s.

## Layout
- `rca_pipeline/` — modèles, training, benchmark, éval (source de vérité du code)
- `rca_pipeline/tests/` — pytest
- `scripts/` — flux K8s + campagnes ObservIT (`.sh`)
- `k8s/` — manifests
- `docs/` — analyses, rapport, benchmarks
- `results_transmission/` — manifest canonique (soutenance)
- `rcaeval/` — données RCAEval (non versionné, ignoré)

## Fichiers clés
- Modèles : `model_sta.py`, `model_late_gat.py`, `model_sta_mm.py`, `model_late_gat_mm.py`
- Entry points : `benchmark.py` (simulé), `benchmark_real.py` (RCAEval/ObservIT), `validation_protocol.py`, `benchmark_ablation.py`
- Data : `dataset_builder.py`, `simulated_data_scalable.py`, `loaders/`, `rcaeval/`
- Training : `train_sta.py`, `train.py`, `training_utils.py`, `losses.py`, `cascade_aware_loss.py`
- Features MM : `log_feature_extractor.py`, `trace_feature_extractor.py`, `sta_fusion_utils.py`
- Topologies : `topologies.py` (custom_app / sock_shop) consommée par `graph_builder.py`
- Connecteurs live : `prometheus_connector.py`, `loki_connector.py`

## Sock-shop (observit-cluster1)
- Chaos manifests : `k8s/chaos/sockshop/` (stress cpu/mem, network delay/loss, pod-kill)
- Locustfile : `rca_pipeline/locustfile_sockshop.py`
- Scripts : `scripts/05-enable-chaos-mesh-cluster-scope.sh`, `scripts/run_sockshop_dataset.sh`, `scripts/run_sockshop_campaign.sh`
- Flag benchmark : `--observit-topology sock_shop`
- Prerequisite : Chaos-Mesh doit être `clusterScoped=true` (helm upgrade via script 05)

## Commandes
- Install : `pip install -e ".[dev]"`
- Tests : `python -m pytest -q rca_pipeline/tests`
- Lint : `python -m ruff check rca_pipeline`
- Bench simulé : `python -m rca_pipeline.benchmark --output-json <path>`
- Bench réel : `python -m rca_pipeline.benchmark_real --dataset all --seeds 42,123,7 --epochs 100 --sta-contrastive-warmup 0 --output-json <path>`

## Conventions
- Python ≥3.9, ruff line-length=100, target py39, select=E,F,I,UP,B
- Torch + torch_geometric pour GNN ; éviter deps non listées dans `pyproject.toml`
- Seeds par défaut : 42,123,7,1337,2025 ; tailles simulées : 20,50,100
- **Guardrail** : sur benchmarks réels, toujours `--sta-contrastive-warmup 0` (warmup >0 casse STA sur réel, cf README §5.4)

## À NE PAS faire
- Ne pas modifier `results_transmission/canonical_results_manifest.json*` (artefact de soutenance figé)
- Ne pas committer `*.pt`, `checkpoints/`, `datasets/`, `rcaeval/data/`, `benchmark*.json` générés (déjà gitignore)
- Ne pas mocker la DB/loaders dans les tests d'intégration

## Documentation
Analyses détaillées : `docs/PAPER_BENCHMARKS.md`, `ANALYSIS_SIM_VS_REAL.md`, `ANALYSIS_MULTIMODAL.md`, `COMPARABILITY_MATRIX.md`, `docs/internal/rapport_stage.md` (interne).

## Notes pour Claude
- Préférer lire les fichiers ciblés plutôt que explorer (arbo large, beaucoup de JSON/logs lourds)
- Les `.pt`, `benchmark*.json`, `datasets/`, `.venv/`, `checkpoints/` sont volumineux — ne pas les lire sauf demande explicite
- Réponses courtes, code sans commentaires superflus (voir ruff config)
