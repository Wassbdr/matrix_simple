# Index de la documentation — matrix_simple / STA

Point d'entrée pour la reprise : [`../HANDOVER.md`](../HANDOVER.md).
Vue d'ensemble scientifique et tables de résultats : [`../README.md`](../README.md).

## Carte du dépôt

| Répertoire | Rôle |
|---|---|
| `rca_pipeline/` | **Source de vérité du code** : modèles, entraînement, benchmarks, évaluation |
| `rca_pipeline/tests/` | Suite pytest du cœur |
| `serving/` | Service FastAPI `/predict` + watcher + tests |
| `deploy/` | Manifests MLOps phase 2 (MLflow, Helm `rca-stack`, dashboard Grafana) |
| `docker/` | Image d'entraînement / retrain / initContainer |
| `configs/` | Configs YAML versionnées (validées par pydantic) |
| `scripts/` | Bootstrap cluster K8s + campagnes ObservIT (`.sh`) |
| `k8s/` | Manifests applicatifs et manifests de chaos |
| `results_transmission/` | Artefacts canoniques de soutenance, vérifiés SHA-256 |
| `rcaeval/` | Données RCAEval — **non versionné** (~39 Go), voir `HANDOVER.md` |

### Fichiers clés de `rca_pipeline/`

| Rôle | Fichiers |
|---|---|
| Modèles | `model_sta.py`, `model_late_gat.py`, `model_sta_mm.py`, `model_late_gat_mm.py` |
| Points d'entrée | `benchmark.py` (simulé), `benchmark_real.py` (RCAEval/ObservIT), `validation_protocol.py`, `benchmark_ablation.py` |
| Données | `dataset_builder.py`, `simulated_data_scalable.py`, `loaders/` |
| Entraînement | `train_sta.py`, `train.py`, `training_utils.py`, `losses.py`, `cascade_aware_loss.py` |
| Features multimodales | `log_feature_extractor.py`, `trace_feature_extractor.py`, `sta_fusion_utils.py` |
| Topologies | `topologies.py` (`custom_app` / `sock_shop`), consommée par `graph_builder.py` |
| Connecteurs live | `prometheus_connector.py`, `loki_connector.py` |
| MLOps | `tracking.py`, `promote.py`, `train_and_persist.py`, `model_bundle.py`, `monitoring/drift_job.py` |

## Documents

### Courants

| Document | Contenu |
|---|---|
| [`OPERATIONS.md`](OPERATIONS.md) | **Runbook MLOps phase 2** : HPO → bundle → registry → promotion → serving → drift. Le document à ouvrir pour toute opération sur le stack déployé. |
| [`PAPER.md`](PAPER.md) | Rédaction longue : méthode, protocole, résultats, discussion. |
| [`PAPER_BENCHMARKS.md`](PAPER_BENCHMARKS.md) | Commentaire détaillé des benchmarks, table par table. |
| [`COMPARABILITY_MATRIX.md`](COMPARABILITY_MATRIX.md) | **À lire avant toute comparaison aux baselines publiées.** Verdicts d'isopérimétrie (`partially_comparable`, `not_isoperimeter`) et re-run isopérimétrique de BARO. |
| [`ANALYSIS_SIM_VS_REAL.md`](ANALYSIS_SIM_VS_REAL.md) | Pourquoi l'écart simulé ↔ réel, et ce qu'il implique sur la portée des conclusions. |
| [`ANALYSIS_MULTIMODAL.md`](ANALYSIS_MULTIMODAL.md) | Résultat négatif sur la fusion multimodale, documenté. |
| [`ANALYSIS_MODEL_IMPROVEMENTS.md`](ANALYSIS_MODEL_IMPROVEMENTS.md) | Expériences post-soutenance (cascade prior, ensemble, saturation GIB). |
| [`PER_SCENARIO_BREAKDOWN.md`](PER_SCENARIO_BREAKDOWN.md) | Décomposition des résultats par scénario de chaos. |
| [`SYNTHESE_FINALE.md`](SYNTHESE_FINALE.md) | Synthèse de fin de stage. |

### Publications

| Chemin | Contenu |
|---|---|
| [`arxiv/`](arxiv/) | Préprint LaTeX. `make paper` depuis la racine, ou `make -C docs/arxiv pdf`. |
| [`rapport/`](rapport/) | Rapport de stage LaTeX. `make report` depuis la racine. |
| `sota_anomalies.pdf` | Note de veille sur l'état de l'art. |

Les figures des deux documents sont régénérées par `make figures`
(`rca_pipeline/make_figures.py`) depuis les JSON de `results_transmission/`.
Les intermédiaires LaTeX (`.aux`, `.bbl`, `.log`, PDF compilés) sont gitignorés :
un clone frais contient les `.tex` et les figures, pas les PDF.

### Absents d'un clone

- **`docs/internal/`** — notes hebdomadaires, rapport de stage en markdown,
  accompagnement maître de stage. Gitignoré volontairement (contenu personnel),
  présent uniquement sur le poste d'origine.

## Artefacts de résultats : lesquels citer

| Chemin | Statut |
|---|---|
| `results_transmission/paper_benchmark_*.json` | **Canonique.** Ce sont ces chiffres qui figurent dans le README et le papier. |
| `results_transmission/canonical_results_manifest.json` + `.sha256` | Instantané figé de soutenance. Ne jamais modifier. |
| `results_transmission/*_10seed.json` + `paper_benchmark_10seed.sha256` | Contrôle de robustesse à 10 graines. |
| `results_transmission/baselines_rcaeval_ourprotocol.json` | Re-run isopérimétrique de BARO. |
| `rca_pipeline/*_prepublish.json` | **Legacy, non canonique.** Issus d'un run à faible nombre d'epochs ; les champs `reference_top1` sont périmés et contredisent le run déterministe 5 graines. Ne pas citer. |

Vérification d'intégrité :

```bash
# Attention : les deux manifestes n'ont pas la même convention de chemin.
# paper_benchmark*.sha256 liste des noms nus → se lance depuis results_transmission/
(cd results_transmission && sha256sum -c paper_benchmark.sha256)
(cd results_transmission && sha256sum -c paper_benchmark_10seed.sha256)
# canonical_results_manifest.json.sha256 liste un chemin depuis la racine
sha256sum -c results_transmission/canonical_results_manifest.json.sha256
```
