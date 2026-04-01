# Audit final stage — RCA microservices (Software, ML, MLOps)

Date: 2026-04-01  
Périmètre audité: code + artefacts présents dans le dépôt uniquement.

## A. Résumé exécutif (max 10 lignes)
- Niveau global de préparation: 62/100.
- Forces majeures:
  1. Chaîne d’évaluation outillée et testée (23 tests passés), protocole scripté reproductible.
  2. Artefacts riches (métriques RCAEval, CI95, config exportée, drift raw vs filtered, couverture MM).
  3. Instrumentation avancée disponible (MM strict, comparaison baselines, split stratifié, normalisation train-only).
- Risques majeurs:
  1. Incohérence forte entre artefacts « officiels » et « prepublish » sur des scores clés (impact direct soutenance).
  2. Warmup contrastif STA instable sur réel (effondrement possible des performances selon configuration).
  3. Crédibilité expérimentale incomplète sur 3 contrôles obligatoires: per-service réel, per-failure-type réel, isopérimètre baselines publiées.

## B. Scorecard

| Dimension | Score/10 | Preuve | Risque | Priorité |
|---|---:|---|---|---|
| Qualité logicielle | 6.5 | 23 tests passés: commande pytest sur rca_pipeline/tests. Dette statique: 691 erreurs Ruff. | Maintenabilité et risques de régression silencieuse. | Haute |
| Validité expérimentale | 5.5 | Split stratifié, CI95 avec flag fiabilité seed-level, mais n=3/5 souvent non fiable. Warmup réel très sensible. | Surinterprétation des gains, variance mal cadrée. | Critique |
| Benchmarking & SOTA | 4.5 | Comparaison aux baselines publiées via constantes globales; incohérences baseline_official vs benchmark_full. | Comparaison potentiellement non isopérimètre. | Critique |
| Opérationnalisation & reproducibilité | 6.0 | validation_protocol + exports JSON détaillés; mais référence doc manquante et artefacts générés hétérogènes. | Difficulté de transfert et de défense jury. | Haute |

## C. Findings priorisés

| ID | Gravité | Problème | Preuve | Impact | Correctif | Effort | Quick Win |
|---|---|---|---|---|---|---|---|
| F01 | Critique | Fait: fortes divergences de scores entre artefacts simulés « benchmark_full_prepublish » et « baseline_official » (ex. N20 STA top1 ≈0.769 vs 0.9795). Interprétation: corpus de référence non unifié, comparabilité soutenance fragilisée. Recommandation: figer un artefact canonique unique et tracer provenance/commande/hash. | [rca_pipeline/benchmark_full_prepublish.json](rca_pipeline/benchmark_full_prepublish.json#L14), [rca_pipeline/baseline_official.json](rca_pipeline/baseline_official.json#L18), [rca_pipeline/baseline_official.json](rca_pipeline/baseline_official.json#L36), [rca_pipeline/baseline_official.json](rca_pipeline/baseline_official.json#L45) | Risque de rejet de crédibilité par le jury. | Créer manifest de vérité (JSON unique + script de génération + checksum). | M | Oui |
| F02 | Critique | Fait: STA réel varie drastiquement avec warmup contrastif (w0 avg1=0.7333 vs w10 avg1=0.0 sur même protocole léger). Interprétation: instabilité méthodologique sur réel, claim warmup non robuste. Recommandation: garder warmup=0 par défaut réel, ajouter garde-fou et benchmark A/B obligatoire avant diffusion. | [results_transmission/warmup_real_trainticket_w0.json](results_transmission/warmup_real_trainticket_w0.json#L4), [results_transmission/warmup_real_trainticket_w10.json](results_transmission/warmup_real_trainticket_w10.json#L4), [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py#L659), [rca_pipeline/ablation_prepublish.json](rca_pipeline/ablation_prepublish.json#L457) | Risque majeur de régression en démo soutenance. | Ajouter test de non-régression warmup réel et publier résultats A/B signés. | S | Oui |
| F03 | Critique | Fait: MM strict sur sockshop ne produit pas de métriques (stub/skipped) mais un artefact MM strict indique mm_effective=true avec traces_ratio=0.0. Interprétation: confusion possible entre « run MM exécuté » et « run MM réellement comparable ». Recommandation: marquer explicitement statut dataset-level (comparable/non comparable) et bloquer tout résumé global MM incluant sockshop. | [rca_pipeline/validation_protocol.py](rca_pipeline/validation_protocol.py#L107), [rca_pipeline/validation_runs/sockshop__full_mm_strict.json](rca_pipeline/validation_runs/sockshop__full_mm_strict.json#L3), [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](rca_pipeline/benchmark_real_long_mm_strict_prepublish.json#L206) | Mauvaise interprétation multimodale en soutenance. | Ajouter champ JSON: mm_dataset_status = comparable/skipped_with_reason. | S | Oui |
| F04 | Majeur | Fait: comparaison aux baselines publiées repose sur des constantes globales fixes (non dataset/protocole-specific). Interprétation: comparaison potentiellement non isopérimètre. Recommandation: fournir tableau d’alignement protocolaire explicite (splits, features, seeds, max_cases, preprocessing) avant claim SOTA. | [rca_pipeline/metrics_rcaeval.py](rca_pipeline/metrics_rcaeval.py#L258), [rca_pipeline/metrics_rcaeval.py](rca_pipeline/metrics_rcaeval.py#L266), [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py#L639) | Claim SOTA contestable. | Ajouter fichier COMPARABILITY_MATRIX.md généré automatiquement. | M | Oui |
| F05 | Majeur | Fait: métriques par service/root-cause exportées seulement pour ObservIT, pas pour RCAEval réel. Interprétation: contrôle déséquilibre classes incomplet sur benchmark réel principal. Recommandation: exporter métriques par service pour RCAEval (ou indiquer impossibilité avec justification et alternative). | [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py#L644), [datasets/observit_runs/benchmark_with_per_rc.json](datasets/observit_runs/benchmark_with_per_rc.json#L217) | Blind spot sur équité inter-services. | Étendre export per_root_cause à tous datasets quand labels disponibles. | M | Non |
| F06 | Majeur | Fait: incohérence interne ObservIT: avg1 global élevé (~0.912) mais per_root_cause = 0.0 pour tous services. Interprétation: bug d’agrégation ou mapping d’indices/labels. Recommandation: corriger top1_per_service pipeline et ajouter test de cohérence (global vs micro-agrégé). | [datasets/observit_runs/benchmark_with_per_rc.json](datasets/observit_runs/benchmark_with_per_rc.json#L4), [datasets/observit_runs/benchmark_with_per_rc.json](datasets/observit_runs/benchmark_with_per_rc.json#L217), [rca_pipeline/metrics_rcaeval.py](rca_pipeline/metrics_rcaeval.py#L298) | Données per-service actuellement non exploitables. | Ajouter test invariant: weighted(per_service_top1)=global_top1. | S | Oui |
| F07 | Majeur | Fait: performance par type de défaillance (hard/gray/cascade) bien présente en simulé mais absente des artefacts réels long. Interprétation: contrôle 6 non couvert sur le périmètre réel. Recommandation: injecter taxonomy compatible réel ou publier explicitement « non mesurable » + plan de collecte. | [rca_pipeline/ablation_prepublish.json](rca_pipeline/ablation_prepublish.json#L457), [rca_pipeline/benchmark_full_prepublish.json](rca_pipeline/benchmark_full_prepublish.json#L14), [rca_pipeline/benchmark_real_long_prepublish.json](rca_pipeline/benchmark_real_long_prepublish.json#L1) | Claim « robuste à tous types de pannes » non prouvé en réel. | Ajouter section real_failure_type_metrics dans JSON réel. | L | Non |
| F08 | Majeur | Fait: protocoles split multi-seed hétérogènes (simulé: split fixé par dataset_seed; réel: split dépend du seed de run). Interprétation: variance inter-seeds non comparable entre volets simulé/réel. Recommandation: expliciter deux modes (fixed-split vs varying-split) et publier les deux. | [rca_pipeline/benchmark.py](rca_pipeline/benchmark.py#L1297), [rca_pipeline/benchmark.py](rca_pipeline/benchmark.py#L1301), [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py#L423), [rca_pipeline/benchmark_real.py](rca_pipeline/benchmark_real.py#L433) | Analyse de stabilité partiellement biaisée. | Ajouter flag split_mode + reporting séparé. | M | Oui |
| F09 | Mineur | Fait: drift filtering présent mais impact nul sur artefacts RCAEval longs (drop_ratio=0, raw=filtered identiques). Interprétation: contrôle actif mais non informatif sur ce corpus; pas de conclusion robustesse drift. Recommandation: inclure un scénario drift stress-test dédié. | [rca_pipeline/benchmark_real_long_prepublish.json](rca_pipeline/benchmark_real_long_prepublish.json#L133), [rca_pipeline/benchmark_real_long_prepublish.json](rca_pipeline/benchmark_real_long_prepublish.json#L194) | Valeur démonstrative limitée du module drift. | Créer un benchmark drift_sensitivity dédié. | S | Oui |
| F10 | Mineur | Fait: dette qualité statique élevée (Ruff: 691 erreurs). Interprétation: coût de maintenance élevé, risque d’érosion qualité à moyen terme. Recommandation: plan de réduction par lots (imports, longueur lignes, variables inutilisées). | Commande: .venv/bin/python -m ruff check rca_pipeline --statistics → 691 erreurs. | Risque de régressions futures et lisibilité réduite. | Sprint qualité dédié + gate CI progressif. | M | Non |
| F11 | Opportunité | Fait: doc référencée PAPER_BENCHMARKS absente du dépôt courant. Interprétation: traçabilité de périmètre MM incomplète pour un évaluateur externe. Recommandation: ajouter ce document ou remplacer la référence par un doc présent. | [docs/BENCHMARK_ARTIFACTS.md](docs/BENCHMARK_ARTIFACTS.md#L3), recherche fichier: **/*PAPER_BENCHMARKS*.md → aucun résultat. | Risque de friction revue externe. | Publier doc de périmètre et versionner. | S | Oui |

## D. Audit de crédibilité expérimentale

### Claims validés
- Le pipeline exporte des métriques RCAEval complètes et CI95 avec signal de fiabilité seed-level.
- Les runs MM strict sur trainticket et online-boutique sont effectivement multimodaux (coverage logs/traces à 1.0).
- La séparation raw vs drift_filtered est bien exportée dans les JSON.

### Claims partiellement validés
- Robustesse multi-seed: validée en principe (std/CI exportés), mais puissance statistique faible sur n=3/5 et protocoles split hétérogènes.
- Comparaison à BARO: calculée correctement dans le code, mais isopérimètre externe non démontré exhaustivement.

### Claims non prouvés
- Performance par type de défaillance sur réel RCAEval (hard/gray/cascade).
- Analyse de déséquilibre root-cause par service sur réel RCAEval principal.
- Comparabilité MM incluant sockshop en strict (hors périmètre effectif).

### Claims invalidés
- « Warmup contrastif STA améliore de façon robuste le réel »: invalidé par test comparatif explicite w0 vs w10 sur trainticket (w10 effondre STA).

## E. Plan d’action

### Horizon 48h (indispensable avant soutenance)
1. Objectif: unifier la source de vérité des résultats.
   - Livrable: un JSON canonique + tableau de correspondance (old/new) + script de génération.
   - Critère mesurable: 0 divergence numérique non expliquée entre slides et artefact canonique.
   - Effort/Impact: M / Très fort.
2. Objectif: sécuriser STA réel contre régression warmup.
   - Livrable: défaut warmup=0 conservé + test CI A/B (w0 vs w10) + note protocole.
   - Critère mesurable: test échoue si Avg@1 STA chute >10 points sur protocole smoke fixé.
   - Effort/Impact: S / Très fort.
3. Objectif: verrouiller la sémantique multimodale.
   - Livrable: champ mm_dataset_status et blocage agrégation MM si dataset skipped.
   - Critère mesurable: impossible d’avoir « MM global » avec sockshop skipped sans avertissement bloquant.
   - Effort/Impact: S / Fort.
4. Objectif: corriger métriques per_root_cause ObservIT.
   - Livrable: patch + test de cohérence global/per-service.
   - Critère mesurable: écart absolu <= 1e-6 entre avg1 global et moyenne pondérée per-service.
   - Effort/Impact: S / Fort.

### Horizon 2 semaines (consolidations majeures)
1. Objectif: normaliser l’analyse multi-seed.
   - Livrable: double protocole reporté (fixed-split et varying-split) sur simulé et réel.
   - Critère mesurable: tableau de variance séparé par mode, reproductible par commande unique.
   - Effort/Impact: M / Fort.
2. Objectif: renforcer crédibilité statistique.
   - Livrable: preset paper 10 seeds exécuté sur jeux prioritaires.
   - Critère mesurable: seed_level_bootstrap_reliable=true pour métriques principales.
   - Effort/Impact: M / Fort.
3. Objectif: comparabilité isopérimètre SOTA.
   - Livrable: COMPARABILITY_MATRIX (hypothèses, données, preprocessing, seeds, budgets compute).
   - Critère mesurable: chaque claim baseline trace un protocole aligné ou un écart explicitement justifié.
   - Effort/Impact: M / Très fort.

### Horizon 1 à 2 mois (améliorations structurantes)
1. Objectif: couverture réelle par type de défaillance.
   - Livrable: schéma taxonomy réel + pipeline d’annotation/alignement.
   - Critère mesurable: publication systématique hard/gray/cascade sur réel.
   - Effort/Impact: L / Très fort.
2. Objectif: qualité code durable.
   - Livrable: réduction progressive Ruff (objectif <100), gate CI graduelle.
   - Critère mesurable: baisse continue du backlog lint sur 4 sprints.
   - Effort/Impact: M / Moyen.
3. Objectif: reproductibilité externe bout-en-bout.
   - Livrable: manifest complet (hash datasets, env lock, commandes, artefacts signés).
   - Critère mesurable: reproduction complète par une équipe tierce en <1 jour.
   - Effort/Impact: L / Très fort.

## F. Kit soutenance

### 10 questions agressives probables + réponse technique recommandée
1. Pourquoi vos chiffres « officiels » ne matchent pas vos prepublish ?
   - Réponse: nous avons identifié et isolé deux corpus; la soutenance s’appuie uniquement sur le corpus canonique signé (script + hash + commit).
2. Comment prouvez-vous que vos runs MM sont vraiment multimodaux ?
   - Réponse: on expose mm_coverage et mm_dataset_status par dataset; runs strict sans modalités requises sont exclus et marqués skipped.
3. Pourquoi sockshop MM est absent ?
   - Réponse: protocole strict exige traces valides; sockshop ne satisfait pas ce critère dans ce corpus, donc exclusion explicite pour éviter un faux comparatif.
4. Vos IC sont-elles fiables ?
   - Réponse: non en n=3/5; le JSON signale explicitement seed_level_bootstrap_reliable=false. On ne survend pas ces IC.
5. Le warmup contrastif aide-t-il réellement ?
   - Réponse: pas de façon robuste en réel; nos tests A/B montrent une dégradation possible sévère, donc warmup=0 par défaut réel.
6. Vos splits multi-seed sont-ils homogènes ?
   - Réponse: simulé fixe le split pour comparer les optimiseurs; réel varie le split par seed. Les deux modes seront reportés séparément.
7. Où sont les métriques par type de panne sur réel ?
   - Réponse: indisponibles aujourd’hui faute de taxonomy alignée; présentes en simulé, plan de collecte réel en cours.
8. Où sont les métriques par service sur RCAEval réel ?
   - Réponse: actuellement exportées seulement sur ObservIT; extension RCAEval priorisée avant publication finale.
9. Votre comparaison BARO est-elle isopérimètre ?
   - Réponse: comparaison indicative aujourd’hui; une matrice d’alignement protocolaire complète est livrée pour valider claim par claim.
10. Êtes-vous prêt pour transfert équipe ?
   - Réponse: oui sous conditions: corpus canonique, fixes per_root_cause, guardrails MM/warmup, et manifest reproductibilité.

### Points à ne pas survendre
- Ne pas annoncer « SOTA validée » sans matrice isopérimètre complète.
- Ne pas présenter les IC n=3/5 comme robustes statistiquement.
- Ne pas généraliser les résultats MM à sockshop en mode strict.
- Ne pas extrapoler hard/gray/cascade réel sans artefacts dédiés.

## G. Informations manquantes

### Ce qui manque pour conclure avec confiance élevée
1. Artefact canonique unique et traçable des scores soutenance (priorité P0).
2. Matrice d’alignement isopérimètre avec baselines publiées (P0).
3. Runs 10 seeds sur périmètre clé pour IC robustes (P1).
4. Métriques par service sur RCAEval réel (P1).
5. Métriques par type de défaillance sur réel (P1/P2 selon disponibilité labels).
6. Documentation de périmètre MM (fichier référencé mais absent) (P1).
7. Explication/patch de l’incohérence ObservIT global vs per_root_cause (P0).

### Ordre de priorité de collecte
1. P0: F01, F02, F03, F06.
2. P1: F04, F05, F08, F11.
3. P2: F07, F10.

## H. Décision finale
- Décision: Go sous conditions.

### Conditions minimales pour obtenir Go
1. Unifier et figer le corpus de résultats (plus aucune divergence chiffrée non expliquée).
2. Publier la comparaison warmup réel A/B et verrouiller warmup=0 par défaut réel.
3. Corriger la métrique per_root_cause ObservIT et prouver la cohérence avec avg1 global.
4. Rendre explicite le statut MM par dataset (sockshop excluded strict).
5. Fournir une matrice d’isopérimètre pour toute comparaison baseline publiée.

---

## Preuves d’exécution complémentaires (audit)
- Tests: `.venv/bin/python -m pytest -q rca_pipeline/tests` → 23 passed.
- Lint: `.venv/bin/python -m ruff check rca_pipeline --statistics` → 691 erreurs.
- Warmup réel comparatif exécuté:
  - `.venv/bin/python rca_pipeline/benchmark_real.py --dataset trainticket --seeds 42,123,7 --epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 0 --output-json results_transmission/warmup_real_trainticket_w0.json`
  - `.venv/bin/python rca_pipeline/benchmark_real.py --dataset trainticket --seeds 42,123,7 --epochs 20 --max-cases 20 --enable-drift --sta-contrastive-warmup 10 --output-json results_transmission/warmup_real_trainticket_w10.json`

---

## Addendum post-correctifs (même branche, sans commit)

Date: 2026-04-01 (après implémentation des P0/P1 demandés).

### État des findings

| ID | État actuel | Preuve mise à jour |
|---|---|---|
| F01 | **Fermé** (source canonique en place) | `results_transmission/canonical_results_manifest.json`, `results_transmission/canonical_results_manifest.json.sha256` |
| F02 | **Fermé** (guardrail + A/B publié) | `rca_pipeline/benchmark_real.py` (`_enforce_real_warmup_guard` + `--allow-real-warmup`), `results_transmission/warmup_real_trainticket_w0.json`, `results_transmission/warmup_real_trainticket_w10.json` |
| F03 | **Fermé** (statut MM dataset-level + blocage global) | `rca_pipeline/benchmark_real.py` (`mm_dataset_status`, `_mm_global_summary`), `rca_pipeline/validation_protocol.py` (stub `skipped_with_reason`) |
| F04 | **Fermé** (matrice isopérimètre outillée) | `rca_pipeline/comparability_matrix.py`, `docs/COMPARABILITY_MATRIX.md`, `results_transmission/comparability_matrix.json` |
| F05 | **Fermé** (per-root exporté sur réel RCAEval) | `rca_pipeline/benchmark_real.py` (`_attach_per_root_metrics` appliqué aux datasets réels), `results_transmission/smoke_pre_submission_check.json` |
| F06 | **Fermé** (invariant global vs per-service) | `rca_pipeline/metrics_rcaeval.py` (`top1_per_service_detailed.weighted_top1`), `rca_pipeline/benchmark_real.py` (check `max_abs_gap <= tol`) |
| F07 | **Ouvert** (partiellement) | métriques real_failure_type non publiées dans les artefacts réels longs |
| F08 | **Fermé** (split_mode explicite) | `rca_pipeline/benchmark.py` + `rca_pipeline/benchmark_real.py` (`split_mode`), `results_transmission/smoke_pre_submission_check.json` |
| F09 | **Ouvert** (mineur) | drift actif mais impact nul sur corpus actuel |
| F10 | **Ouvert** (mineur) | backlog Ruff toujours élevé (plan qualité requis) |
| F11 | **Fermé** | `docs/PAPER_BENCHMARKS.md` présent et référencé |

### Validation finale exécutée

- Tests complets: `.venv/bin/python -m pytest -q rca_pipeline/tests` → **35 passed**.
- Smoke pré-soumission régénéré: `results_transmission/smoke_pre_submission_check.json`.
- Archive finale prête à transmettre: `results_transmission/SOUTENANCE_PACKAGE_2026-04-01.tar.gz`.

### Décision opérationnelle mise à jour

- **Go sous conditions légères**:
   1. annoncer explicitement F07/F09/F10 comme limites connues,
   2. ne pas revendiquer d’isopérimètre MM externe (verdict `not_isoperimeter`),
   3. s’appuyer uniquement sur le manifeste canonique pour tous les chiffres de slides.
