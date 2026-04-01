# Week4 — Bilan General Depuis Week3

Date: 2026-04-01

## 1) Objectif de la semaine

Depuis Week3, l'objectif n'etait plus seulement d'ameliorer les modeles, mais de passer en mode pre-soumission:

- consolider la credibilite experimentale,
- verrouiller la reproductibilite,
- fiabiliser les scripts et artefacts,
- preparer un paquet soutenance defendable techniquement.

## 2) Methode d'analyse appliquee

Pour faire ce bilan general, l'analyse a ete faite sur l'ensemble des changements visibles dans la branche, en excluant les fichiers techniques de l'environnement (.venv, caches), avec trois vues:

- vue structurelle: volume et type de changements (modifies / ajoutes / remplaces),
- vue fonctionnelle: impact Software, ML, MLOps,
- vue livrable: docs, artefacts de transmission et preuves d'execution.

Synthese quantitative des modifications utiles:

- fichiers modifies: 24
- fichiers remplaces/supprimes puis refondus: 10
- nouveaux fichiers/dossiers ajoutes: 22
- total modifications pertinentes: 56

## 3) Analyse generale des modifications

## 3.1 Axe Software (architecture, robustesse, qualite)

Changements majeurs:

- Renforcement du noyau benchmark reel dans [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py):
	- garde-fou sur warmup reel,
	- statuts multimodaux dataset-level,
	- gestion explicite des seeds executes/skipped,
	- metadata split_mode exposees,
	- export per_root_cause detaille et invariant de coherence.
- Renforcement benchmark simule dans [rca_pipeline/benchmark.py](rca_pipeline/benchmark.py):
	- split stratifie melange,
	- traçabilite split fixe par dataset_seed,
	- protocole d'ablation plus explicite,
	- metadonnees runtime plus completes.
- Stabilisation utilitaires communs dans [rca_pipeline/common.py](rca_pipeline/common.py):
	- mode deterministe PyTorch,
	- runtime_env_info,
	- stratified_split_shuffle reutilisable.
- Correctif TTA dans [rca_pipeline/drift_detector.py](rca_pipeline/drift_detector.py):
	- should_adapt base sur cooldown detector.
- Harmonisation import/loaders:
	- [rca_pipeline/loaders/observit_loader.py](rca_pipeline/loaders/observit_loader.py)
	- [rca_pipeline/loaders/rcaeval_loader.py](rca_pipeline/loaders/rcaeval_loader.py)

Impact:

- Moins de comportements implicites,
- Plus de traces pour debugger/regarder les runs,
- Moins de risque de comparaison invalide sans signaler l'ambiguite.

## 3.2 Axe ML / experimentation

Changements majeurs:

- Clarification et extension du protocole d'evaluation reel dans [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py):
	- separation raw vs drift-filtered,
	- CI95 seed-level avec drapeau de fiabilite,
	- MM coverage et statut explicite,
	- split_mode explicite.
- Enrichissement des metriques dans [rca_pipeline/metrics_rcaeval.py](rca_pipeline/metrics_rcaeval.py):
	- top1_per_service_detailed,
	- weighted_top1,
	- bootstrap_ci avec seed_level_bootstrap_reliable.
- Campagne d'ablation plus large dans [rca_pipeline/benchmark_ablation.py](rca_pipeline/benchmark_ablation.py):
	- decomposition spatial-only / temporal-only,
	- protocoles warmup/no-warmup, k-annealing/no-k-annealing,
	- export JSON ablation.
- Ajout d'une ablation reel temporal-only dans [rca_pipeline/ablation_temporal_only_real.py](rca_pipeline/ablation_temporal_only_real.py).

Impact:

- Les conclusions sont moins basees sur un score global unique,
- La variance inter-seeds et les limites statistiques sont explicites,
- Le discours "simule vs reel" devient objectivable.

## 3.3 Axe MLOps / reproductibilite / comparabilite

Changements majeurs:

- Ajout d'un manifeste canonique avec checksum:
	- [rca_pipeline/results_manifest.py](rca_pipeline/results_manifest.py)
	- [results_transmission/canonical_results_manifest.json](results_transmission/canonical_results_manifest.json)
	- [results_transmission/canonical_results_manifest.json.sha256](results_transmission/canonical_results_manifest.json.sha256)
- Ajout d'une matrice de comparabilite isoperimetre:
	- [rca_pipeline/comparability_matrix.py](rca_pipeline/comparability_matrix.py)
	- [docs/COMPARABILITY_MATRIX.md](docs/COMPARABILITY_MATRIX.md)
	- [results_transmission/comparability_matrix.json](results_transmission/comparability_matrix.json)
- Protocole de validation formalise par presets smoke/long/paper dans [rca_pipeline/validation_protocol.py](rca_pipeline/validation_protocol.py).
- Clarification perimetre MM paper dans [docs/PAPER_BENCHMARKS.md](docs/PAPER_BENCHMARKS.md).
- Passage packaging via [pyproject.toml](pyproject.toml) et documentation de reproduction mise a jour dans [README.md](README.md).

Impact:

- Traçabilite nettement meilleure pour la soutenance,
- Claims baseline/benchmark mieux qualifies,
- Reproduction externe plus realiste (meme sans commit final).

## 3.4 Axe data/cluster et operation terrain

Changements majeurs:

- Script de campagne ObservIT orientee SOTA dans [scripts/run_observit_sota_campaign.sh](scripts/run_observit_sota_campaign.sh):
	- couverture multi-scenarios,
	- root-cause + failure_type explicites,
	- validation finale de campagne.
- Scripts raccourcis pour collecte/benchmark:
	- [scripts/run_observit_dataset.sh](scripts/run_observit_dataset.sh)
	- [scripts/run_observit_benchmark.sh](scripts/run_observit_benchmark.sh)
- Ajustement de deploiement app dans [scripts/03-deploy-app.sh](scripts/03-deploy-app.sh) (port-forward frontend 8080).

Impact:

- Meilleure capacite a rejouer une campagne reelle,
- Pipeline terrain plus simple a transmettre a une autre equipe.

## 3.5 Axe tests et validation

Nouveaux tests ajoutes (principaux):

- [rca_pipeline/tests/test_p0_audit_guards.py](rca_pipeline/tests/test_p0_audit_guards.py)
- [rca_pipeline/tests/test_results_manifest.py](rca_pipeline/tests/test_results_manifest.py)
- [rca_pipeline/tests/test_comparability_matrix.py](rca_pipeline/tests/test_comparability_matrix.py)
- [rca_pipeline/tests/test_validation_protocol_mm_scope.py](rca_pipeline/tests/test_validation_protocol_mm_scope.py)
- [rca_pipeline/tests/test_metrics_correctness.py](rca_pipeline/tests/test_metrics_correctness.py)
- [rca_pipeline/tests/test_model_forward.py](rca_pipeline/tests/test_model_forward.py)
- [rca_pipeline/tests/test_tta.py](rca_pipeline/tests/test_tta.py)
- [rca_pipeline/tests/test_audit_fixes.py](rca_pipeline/tests/test_audit_fixes.py)

Tests modifies:

- [rca_pipeline/tests/test_failure_type_utils.py](rca_pipeline/tests/test_failure_type_utils.py)
- [rca_pipeline/tests/test_k_selection.py](rca_pipeline/tests/test_k_selection.py)
- [rca_pipeline/tests/test_rcaeval_loader_mapping.py](rca_pipeline/tests/test_rcaeval_loader_mapping.py)
- [rca_pipeline/tests/test_sta_fusion_utils.py](rca_pipeline/tests/test_sta_fusion_utils.py)

Lecture:

- La couverture des points critiques audit/reproductibilite est montee,
- La logique de garde-fou est testee explicitement.

## 3.6 Axe documentation et narration scientifique

Nouveaux documents de cadrage/analyse:

- [docs/ANALYSIS_MULTIMODAL.md](docs/ANALYSIS_MULTIMODAL.md)
- [docs/ANALYSIS_SIM_VS_REAL.md](docs/ANALYSIS_SIM_VS_REAL.md)
- [docs/AUDIT_CODE_RESULTS.md](docs/AUDIT_CODE_RESULTS.md)
- [docs/BENCHMARK_ARTIFACTS.md](docs/BENCHMARK_ARTIFACTS.md)
- [docs/COMPARABILITY_MATRIX.md](docs/COMPARABILITY_MATRIX.md)
- [docs/PAPER_BENCHMARKS.md](docs/PAPER_BENCHMARKS.md)

Livrables de transmission:

- [results_transmission/AUDIT_FINAL_SOUTENANCE_2026-04-01.md](results_transmission/AUDIT_FINAL_SOUTENANCE_2026-04-01.md)
- [results_transmission/smoke_pre_submission_check.json](results_transmission/smoke_pre_submission_check.json)
- [results_transmission/warmup_real_trainticket_w0.json](results_transmission/warmup_real_trainticket_w0.json)
- [results_transmission/warmup_real_trainticket_w10.json](results_transmission/warmup_real_trainticket_w10.json)
- [results_transmission/README_resultats.txt](results_transmission/README_resultats.txt)
- [results_transmission/SOUTENANCE_PACKAGE_2026-04-01.tar.gz](results_transmission/SOUTENANCE_PACKAGE_2026-04-01.tar.gz)

Impact:

- Le passage code -> evidence -> message soutenance est beaucoup plus clair.

## 4) Trame Week4 (presentation generale conseillée)

Cette trame peut etre utilisee pour le rapport et l'oral.

## 4.1 Point de depart (fin Week3)

- Modeles et benchmarks operationnels,
- mais risques de comparabilite, d'ambiguite MM et de reproductibilite pre-soutenance.

## 4.2 Travaux de Week4

- Volet Software:
	- hardening des pipelines benchmark,
	- garde-fous explicites,
	- normalisation des metadata de runs.
- Volet ML:
	- tests A/B warmup reel,
	- metriques per-service,
	- split_mode explicite,
	- lecture simule vs reel plus rigoureuse.
- Volet MLOps:
	- manifeste canonique signe,
	- matrice de comparabilite,
	- package de transmission.
- Volet Documentation:
	- cadrage perimetre paper,
	- analyses transverses,
	- synthese audit-action.

## 4.3 Resultats obtenus

- Paquet soutenance techniquement defendable,
- claims mieux qualifies (notamment isoperimetre/MM),
- pipeline de reproduction clarifie.

## 4.4 Limites explicites

- real_failure_type_metrics encore incomplet sur RCAEval reel,
- dette lint encore elevee,
- validation statistique "paper-grade" encore a etendre (10 seeds).

## 4.5 Suite proposee

- court terme: fermer les limites ouvertes avant publication externe,
- moyen terme: automatiser un preflight unique (tests + generation artefacts + checks),
- long terme: renforcer la robustesse statistique et la taxonomie reelle des pannes.

## 5) Inventaire exhaustif des modifications de la branche

## 5.1 Fichiers modifies

- [.gitignore](.gitignore)
- [README.md](README.md)
- [rca_pipeline/benchmark.py](rca_pipeline/benchmark.py)
- [rca_pipeline/benchmark_ablation.py](rca_pipeline/benchmark_ablation.py)
- [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py)
- [rca_pipeline/common.py](rca_pipeline/common.py)
- [rca_pipeline/drift_detector.py](rca_pipeline/drift_detector.py)
- [rca_pipeline/evaluate.py](rca_pipeline/evaluate.py)
- [rca_pipeline/graph_builder.py](rca_pipeline/graph_builder.py)
- [rca_pipeline/loaders/observit_loader.py](rca_pipeline/loaders/observit_loader.py)
- [rca_pipeline/loaders/rcaeval_loader.py](rca_pipeline/loaders/rcaeval_loader.py)
- [rca_pipeline/metrics_rcaeval.py](rca_pipeline/metrics_rcaeval.py)
- [rca_pipeline/model_late_gat.py](rca_pipeline/model_late_gat.py)
- [rca_pipeline/model_late_gat_mm.py](rca_pipeline/model_late_gat_mm.py)
- [rca_pipeline/model_sta_mm.py](rca_pipeline/model_sta_mm.py)
- [rca_pipeline/tests/test_failure_type_utils.py](rca_pipeline/tests/test_failure_type_utils.py)
- [rca_pipeline/tests/test_k_selection.py](rca_pipeline/tests/test_k_selection.py)
- [rca_pipeline/tests/test_rcaeval_loader_mapping.py](rca_pipeline/tests/test_rcaeval_loader_mapping.py)
- [rca_pipeline/tests/test_sta_fusion_utils.py](rca_pipeline/tests/test_sta_fusion_utils.py)
- [rca_pipeline/validation_protocol.py](rca_pipeline/validation_protocol.py)
- [scripts/03-deploy-app.sh](scripts/03-deploy-app.sh)
- [scripts/run_observit_benchmark.sh](scripts/run_observit_benchmark.sh)
- [scripts/run_observit_dataset.sh](scripts/run_observit_dataset.sh)
- [scripts/run_observit_sota_campaign.sh](scripts/run_observit_sota_campaign.sh)

## 5.2 Fichiers ajoutes

- [docs/ANALYSIS_MULTIMODAL.md](docs/ANALYSIS_MULTIMODAL.md)
- [docs/ANALYSIS_SIM_VS_REAL.md](docs/ANALYSIS_SIM_VS_REAL.md)
- [docs/AUDIT_CODE_RESULTS.md](docs/AUDIT_CODE_RESULTS.md)
- [docs/BENCHMARK_ARTIFACTS.md](docs/BENCHMARK_ARTIFACTS.md)
- [docs/COMPARABILITY_MATRIX.md](docs/COMPARABILITY_MATRIX.md)
- [docs/PAPER_BENCHMARKS.md](docs/PAPER_BENCHMARKS.md)
- [docs/Week4.md](docs/Week4.md)
- [pyproject.toml](pyproject.toml)
- [rca_pipeline/ablation_temporal_only_real.py](rca_pipeline/ablation_temporal_only_real.py)
- [rca_pipeline/comparability_matrix.py](rca_pipeline/comparability_matrix.py)
- [rca_pipeline/results_manifest.py](rca_pipeline/results_manifest.py)
- [rca_pipeline/tests/test_audit_fixes.py](rca_pipeline/tests/test_audit_fixes.py)
- [rca_pipeline/tests/test_comparability_matrix.py](rca_pipeline/tests/test_comparability_matrix.py)
- [rca_pipeline/tests/test_metrics_correctness.py](rca_pipeline/tests/test_metrics_correctness.py)
- [rca_pipeline/tests/test_model_forward.py](rca_pipeline/tests/test_model_forward.py)
- [rca_pipeline/tests/test_p0_audit_guards.py](rca_pipeline/tests/test_p0_audit_guards.py)
- [rca_pipeline/tests/test_results_manifest.py](rca_pipeline/tests/test_results_manifest.py)
- [rca_pipeline/tests/test_tta.py](rca_pipeline/tests/test_tta.py)
- [rca_pipeline/tests/test_validation_protocol_mm_scope.py](rca_pipeline/tests/test_validation_protocol_mm_scope.py)
- [rca_pipeline/validation_runs/](rca_pipeline/validation_runs/)
- [rcaeval/](rcaeval/)
- [results_transmission/](results_transmission/)

## 5.3 Fichiers remplaces/supprimes dans l'historique de branche

- 10 artefacts historiques ont ete retires/remplaces dans la branche (anciens rapports d'audit, anciens snapshots baseline, ancien fichier de dependances plat).
- Ces retraits sont compenses par les nouveaux livrables de Week4.
- [docs/AUDIT_CODE_RESULTS.md](docs/AUDIT_CODE_RESULTS.md)
- [docs/COMPARABILITY_MATRIX.md](docs/COMPARABILITY_MATRIX.md)
- [docs/PAPER_BENCHMARKS.md](docs/PAPER_BENCHMARKS.md)
- [rca_pipeline/results_manifest.py](rca_pipeline/results_manifest.py)
- [results_transmission/canonical_results_manifest.json](results_transmission/canonical_results_manifest.json)

Note: l'objectif de ces remplacements est de basculer vers une chaine plus traçable et plus robuste pour la soutenance.

## 6) Conclusion Week4

Week4 marque le passage d'un projet "qui tourne" a un projet "presentable et defendable":

- stabilisation technique,
- explicitation des hypotheses et limites,
- traçabilite des resultats,
- packaging de transmission.

Le gain principal n'est pas un nouveau modele, mais une maturite Software/ML/MLOps nettement superieure pour la soutenance et le transfert.
