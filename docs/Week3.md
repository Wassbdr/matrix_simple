# Rapport de Progression — Semaine 3 : Unification des Données, Renforcement SOTA et Benchmark Réaliste

## 1. Résumé Exécutif

Cette semaine marque un pivot important du projet : passage d’un pipeline majoritairement simulé vers un pipeline de données unifié orienté production, enrichissement du modèle pour les cas difficiles (multi-root, récurrence, causalité temporelle), et audit méthodologique des benchmarks pour éliminer les biais d’évaluation.

Les apports majeurs depuis la Semaine 2 sont :
- Mise en place d’un pipeline unifié multi-sources (simulé, cluster réel, RCAEval, fichiers).
- Introduction d’une configuration centrale modulaire pour les expériences.
- Ajout de pertes et mécanismes SOTA (GIB adaptatif, multi-root, mémoire de récurrence).
- Refonte des baselines MLP/RF pour supprimer la mémorisation positionnelle.
- Élargissement des métriques d’évaluation (Top-k, MRR, SW-MRR, breakdown par pattern).
- Production d’analyses deep-dive et d’un plan d’amélioration aligné état de l’art 2026.

---

## 2. Travaux Réalisés

### 2.1 Unification du pipeline de données

Objectif : sortir d’un flux mono-source et rendre l’entraînement robuste à des distributions hétérogènes.

Livrables clés :
- `rca_pipeline/unified_data.py`
- `rca_pipeline/collect_cluster_data.py`
- `rca_pipeline/collect_log_data.py`
- `rca_pipeline/load_rcaeval.py`
- `rca_pipeline/config.py`

Apports techniques :
- `UnifiedDataLoader` capable de mixer plusieurs sources :
  - Données simulées V2 (`add_synthetic`)
  - Données cluster réelles Prometheus/Jaeger/Loki (`add_cluster`)
  - Datasets de benchmark RCAEval (`add_rcaeval`)
  - Jeux pré-sérialisés (`add_file`)
- Fusion multimodale explicite des features :
  - Métriques : 5 dimensions
  - Traces : +4 dimensions
  - Logs : +4 dimensions
  - Total possible : 13 dimensions
- Normalisation configurable (`global`, `per_source`, `none`) et split pondéré par source.
- Support de l’adaptation de topologie (padding/troncature) pour homogénéiser les batches cross-domain.

### 2.2 Configuration modulaire de l’expérimentation

Objectif : réduire la dette technique des hyperparamètres dispersés.

Livrable clé :
- `rca_pipeline/config.py`

Apports techniques :
- Dataclasses dédiées : `ModelConfig`, `TrainingConfig`, `DataConfig`, `PathConfig`, `ExperimentConfig`.
- Configurations prêtes à l’usage (`default`, `fast`, `thorough`).
- Gestion explicite des groupes de features par modalité.
- Ajustement adaptatif de la sparsité cible en fonction de $N$ :
$$
\text{target\_sparsity} = \max\left(\frac{1}{N},\ \text{target\_sparsity\_base}\right)
$$

### 2.3 Renforcement du cœur modèle et des fonctions de perte

Objectif : améliorer la robustesse sur grands graphes et incidents réalistes.

Livrables clés :
- `rca_pipeline/losses.py`
- `rca_pipeline/train_sta.py`
- `rca_pipeline/model_sta.py`
- `rca_pipeline/memory_module.py`

Apports techniques :
- GIB asymétrique renforcé (fenêtres saines vs anomales) avec contrôle de dispersion.
- Support explicite du multi-root via `multi_root_cause_loss` (BCE focal en mode multi-label, CE sinon).
- Ajout d’une branche mémoire (`MemoryAugmentedRCA`) :
  - banque mémoire différentiable,
  - attention multi-tête,
  - gate mémoire/requête,
  - tête auxiliaire novel vs recurrent.
- Amélioration du protocole STA :
  - warmup contrastif,
  - curriculum sur la sévérité,
  - annealing de température,
  - k-annealing maintenu pour stabiliser la sélection.

### 2.4 Génération de données simulées plus réalistes

Objectif : rapprocher les simulations des conditions d’exploitation AIOps.

Livrables clés :
- `rca_pipeline/simulated_data_scalable.py`
- `rca_pipeline/temporal_patterns.py`
- `rca_pipeline/anomaly_distribution.py`

Apports techniques :
- Version V2 du générateur avec :
  - distribution power-law des causes racines,
  - ratio de récurrence élevé,
  - patterns temporels variés (pas uniquement décroissance exponentielle),
  - metadata enrichie (`is_recurrent`, `pattern_type`, sévérité).
- Renforcement du bruit réaliste (distracteurs corrélés) pour limiter les shortcuts d’apprentissage.

### 2.5 Évaluation et benchmark : fiabilisation méthodologique

Objectif : rendre les comparaisons défendables scientifiquement.

Livrables clés :
- `rca_pipeline/benchmark.py`
- `rca_pipeline/evaluate.py`
- `audit_results_deep_dive.md`
- `benchmark_deep_dive_analysis.md`

Apports techniques :
- Baselines MLP/RF refactorées en mode node-wise pour éviter la mémorisation d’indices de nœuds.
- Correction des protocoles de contrastive learning (positifs par augmentation de la même fenêtre).
- Extension des métriques : Top-1/2/5, MRR, SW-MRR, Precision@k.
- Breakdown d’évaluation par type de pattern, récurrence et difficulté.
- Instrumentation plus explicite des métriques de latence et de stabilité multi-seed.

---

## 3. Résultats Consolidés Semaine 3

Les résultats deep-dive montrent une photographie plus réaliste en environnement durci (multi-modal, distracteurs plus forts, protocole anti-biais).

### 3.1 Tendances observées

- STA reste le modèle le plus robuste parmi les variantes sélectives :
  - N=20 : Top-1 67.8%
  - N=50 : Top-1 48.6%
  - N=100 : Top-1 41.9%
- Late Bottleneck (STGCN+GIB) montre une forte instabilité à grande échelle :
  - N=20 : 35.2%
  - N=50 : 22.7%
  - N=100 : 31.1% avec variance élevée.
- Les scores élevés de baselines globales observés précédemment sont confirmés comme biaisés lorsque l’implémentation autorise une mémorisation de position.

### 3.2 Lecture stratégique des résultats

Le signal principal de la semaine est méthodologique : les performances “brutes” sont moins importantes que leur validité expérimentale.

Le projet progresse sur deux axes essentiels :
- validité scientifique (suppression des biais transductifs),
- transférabilité (pipeline multi-sources, features multimodales, RCAEval).

---

## 4. Nettoyage, refactoring et livrables additionnels

### 4.1 Refactoring structurel

- Suppression de composants devenus obsolètes :
  - `rca_pipeline/simulated_data.py`
  - `rca_pipeline/simulated_data_large.py`
  - `rca_pipeline/simulated_dataset.csv`
- Ajout d’un script d’entraînement hybride :
  - `rca_pipeline/run_hybrid_training.sh`

### 4.2 Documentation et pilotage

- Ajout de documents d’analyse :
  - `audit_results_deep_dive.md`
  - `benchmark_deep_dive_analysis.md`
- Ajout d’un plan d’amélioration aligné SOTA 2026 :
  - `docs/RCA_IMPROVEMENT_PLAN_2026.md`
- Ajout de support bibliographique :
  - `docs/rca_aiops_sota.pdf`

---

## 5. Bilan par rapport aux objectifs de fin Semaine 2

| Objectif issu de Semaine 2 | Statut Semaine 3 | Commentaire |
|---|---|---|
| Validation sur données réelles K8s | **En cours avancé** | Pipeline de collecte réel implémenté (Prometheus/Jaeger/Loki), intégration entraînable via `unified_data.py`. |
| Alternatives / amélioration du GIB | **Partiellement atteint** | GIB adaptatif et pertes enrichies implémentés, mais supériorité systématique non encore démontrée. |
| Robustesse STA à grande échelle | **En cours** | STA stabilisé méthodologiquement, mais performances encore en retrait en contexte durci N=100. |
| Benchmark défendable académiquement | **Atteint (méthodo)** | Biais baselines identifiés, documentés et corrigés (node-wise, métriques étendues). |

---

## 6. Priorités recommandées pour la Semaine 4

1. Exécuter des campagnes complètes sur données réelles cluster avec labels d’alertes robustes.
2. Quantifier précisément le gain du module mémoire sur incidents récurrents vs incidents nouveaux.
3. Poursuivre le tuning STA avec $k$ adaptatif (fonction de la densité locale et non valeur fixe globale).
4. Ajouter des expériences cross-topology strictes (train sur un graphe, test sur un autre) pour valider l’inductivité réelle.
5. Finaliser un protocole de benchmark reproductible type “publication-ready” (seeds, splits, artefacts versionnés).

---

## 7. Conclusion

La Semaine 3 ne se limite pas à des gains de score : elle consolide le projet sur les fondations qui comptent pour un système RCA crédible en production et soutenable en recherche.

Le pipeline est désormais plus proche d’un standard AIOps moderne : multi-source, multi-modal, traçable, et orienté généralisation plutôt que performance opportuniste. Cette base permet d’aborder la prochaine phase avec un cadre expérimental beaucoup plus solide.
