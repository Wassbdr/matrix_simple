# Audit complet du projet et roadmap de stage

Ce document consolide une analyse technique, objective et critique des deux solutions du stage:
- `STA` (`SelectThenAnalyze`)
- `Late-Bottleneck` (`SpatioTemporalRCA + GIB`)

Il se base sur le code et les résultats récents présents dans le dépôt.

---

## 1) Périmètre audité (sources)

- `rca_pipeline/model_sta.py`
- `rca_pipeline/model.py`
- `rca_pipeline/train_sta.py`
- `rca_pipeline/train.py`
- `rca_pipeline/benchmark.py`
- `rca_pipeline/benchmark_real.py`
- `rca_pipeline/losses.py`
- `rca_pipeline/sta_fusion_utils.py`
- `rca_pipeline/cascade_aware_loss.py`
- `rca_pipeline/validation_protocol.py`
- `rca_pipeline/validate_observit_campaign.py`
- `rca_pipeline/loaders/rcaeval_loader.py`
- `rca_pipeline/benchmark_full.json`
- `rca_pipeline/benchmark_v2_full.json`
- `rca_pipeline/benchmark_real_v2_metrics.json`
- `rca_pipeline/benchmark_real_v2_mm.json`
- `rca_pipeline/baseline_official_compact.json`
- `docs/validation_comparison_v1_v2.md`
- `docs/Week3.md`

---

## 2) Lecture architecture: STA vs Late-Bottleneck

## STA (Select-Then-Analyze)

- **Design**: encodage léger zero-hop -> sélection top-k différentiable (`STESelector` ou `HierarchicalSelector`) -> extraction sous-graphe -> analyse deep temporelle/spatiale sur `k` nœuds.
- **Signal clé**: réduction précoce de la surface d’analyse, donc meilleure focalisation causale potentielle.
- **Forces**:
  - Architecture innovante et modulable (`selector_type`, `temporal_encoder`, fusion selector/scorer).
  - Bonne performance simulée v2 avec montée en échelle.
- **Faiblesses**:
  - Chaîne d’optimisation fragile (température, k-annealing, warmup contrastif, fusion).
  - Forte sensibilité au protocole.
  - Performance réelle RCAEval encore faible dans les résultats actuels.

## Late-Bottleneck (SpatioTemporalRCA + GIB tardif)

- **Design**: extraction ST sur tout le graphe -> GIB en fin de pipeline -> scoring.
- **Forces**:
  - Pipeline plus simple et robuste à entraîner.
  - Baseline stable et utile comme ancre d’évaluation.
- **Faiblesses**:
  - Sélection causale tardive (mélange de signaux possible avant filtrage).
  - Avantage de complexité moins clair quand `N` croît.

## Conclusion architecture

- **STA** est l’axe d’innovation/scalabilité.
- **Late-BN** est l’axe stabilité/référence.
- Le stage doit piloter les deux: Late-BN pour fiabilité des conclusions, STA pour contribution scientifique.

---

## 3) Résultats récents: ce qui est factuel

## Simulé (BA): progression nette STA v1 -> v2

Dans `benchmark_full.json` vs `benchmark_v2_full.json`, `STA Select-then-Analyze` progresse fortement:
- N=20: Top-1 ~`0.300 -> 0.677`, MRR ~`0.414 -> 0.754`
- N=50: Top-1 ~`0.183 -> 0.548`, MRR ~`0.276 -> 0.605`
- N=100: Top-1 ~`0.067 -> 0.518`, MRR ~`0.123 -> 0.575`

Lecture:
- Les corrections et le protocole v2 ont réellement amélioré STA en environnement contrôlé.
- STA devient compétitif à grand N.

## Réel (RCAEval): gap persistant STA

Dans `benchmark_real_v2_metrics.json`:
- STA Avg@1 ~`0.20` sur `trainticket`, `sockshop`, `online-boutique`
- GATv2 Avg@1: `0.95`, `0.77`, `0.72`

Dans `benchmark_real_v2_mm.json` (`--use-mm --mm-strict`):
- STA-MM reste à ~`0.20` (sur datasets où run effectif).
- GATv2-MM reste fort.
- `sockshop` incomplet en MM strict (`traces_ratio: 0.0`), donc comparabilité partielle.

Lecture:
- Le projet est méthodologiquement aligné (RCAEval, métriques ranking, baselines), mais STA n’atteint pas le niveau attendu sur réel.

---

## 4) Diagnostic critique du plateau STA ~0.20 (priorité P1)

Le plateau constant à `0.20` est trop régulier pour être ignoré. Plusieurs signaux techniques suggèrent un problème de pipeline/calibration en plus de la difficulté intrinsèque:

1. **Chemin de décision STA simplifié dans le benchmark réel**
   - Dans `benchmark_real.py`, la supervision STA est appliquée sur `selector_logits` uniquement.
   - En évaluation, le score collecté est aussi ce chemin selector (`sel`) et pas explicitement le chemin complet selector+scorer.
   - Risque: on n’exploite pas réellement la proposition full STA (analyse profonde + fusion), donc sous-performance structurelle.

2. **Comparabilité potentiellement hétérogène**
   - `validation_protocol.py` force par défaut des budgets courts (`epochs=20`, `max_cases=20`, seeds limitées) qui peuvent masquer ou accentuer des effets.
   - MM strict peut exclure des seeds et biaiser les comparaisons.

3. **Couverture multimodale incomplète**
   - Cas `sockshop` MM sans traces effectives sur runs stricts -> manque de comparabilité exhaustive.

4. **Risque de confusion entre jeux de résultats**
   - `baseline_official_compact.json` montre des scores simulés très élevés par rapport aux benchmarks v1/v2 actuels, probablement non isopérimètre.

Conclusion diagnostic:
- La faiblesse STA réel n’est probablement pas uniquement "STA est mauvais".
- Elle peut venir d’un mix: protocole, chemin d’inférence simplifié, calibration, couverture multimodale et variance statistique.

---

## 5) Dette technique / risques projet

- **Duplication d’orchestration** entre scripts (`train_*`, `benchmark*`, `validation_protocol`) -> risque d’écarts silencieux.
- **Hyperparamètres nombreux** et couplés côté STA (température, beta, warmup, k-anneal, fusion) -> surface de tuning large.
- **Robustesse statistique faible** (peu de seeds, pas d’IC systématiques dans toutes les conclusions).
- **MM strict binaire** (skip de seeds) -> parfois peu de runs exploitables.

---

## 6) Roadmap de stage (priorisée, opérationnelle)

## P1 — Diagnostiquer le gap réel STA (immédiat)

Objectif: expliquer quantitativement le `Avg@1 ~0.20`.

Actions:
- Instrumenter run sample-level STA sur RCAEval:
  - top-k selector,
  - probas selector,
  - score/fusion finale,
  - rang de la root cause.
- Comparer STA vs GATv2 sur exactement les mêmes seeds/splits/cases.
- Vérifier mapping labels/root-cause IDs et cohérence des classes présentes train/test.

Livrable:
- tableau "cause d’échec STA" par dataset.

## P2 — Protocole unifié STA/Late-BN/GATv2

Objectif: rendre toute comparaison strictement isopérimètre.

Actions:
- Fixer une matrice unique:
  - seeds (ex: `42,123,7,2026,77`),
  - epochs (court/moyen/long),
  - max_cases (court et full),
  - drift on/off,
  - MM strict/on-off.
- Générer un seul format JSON de sortie pour tous les runs.

Livrable:
- "table de vérité protocolaire" jointe à chaque benchmark.

## P3 — Renforcement statistique

Objectif: éviter les conclusions fragiles.

Actions:
- Augmenter le nombre de seeds.
- Ajouter IC95 bootstrap (ou percentile) pour Avg@1, MRR, NDCG.
- Définir des règles go/no-go formelles par dataset.

Livrable:
- section "robustesse statistique" dans le rapport final de stage.

## P4 — Ablations STA ciblées

Objectif: isoler les causes réelles de gain/perte.

Plan minimal:
- `selector_type`: `ste` vs `hierarchical`
- fusion: interne seule vs externe seule vs les deux
- warmup: A2 vs A3 (beta/température/durée)
- k: fixe vs anneal

Métriques obligatoires:
- Avg@1, MRR, NDCG
- root cause dans top-k selector
- latence moyenne

Livrable:
- matrice d’ablation courte mais complète (10-16 runs utiles).

## P5 — Late-BN comme baseline d’ancrage

Objectif: conserver un référentiel stable pendant l’itération STA.

Actions:
- Revalider gains/pertes de GIB asymétrique et cascade loss par type de panne.
- Garder Late-BN dans chaque campagne comme garde-fou méthodologique.

## P6 — MM réellement comparable

Objectif: rendre la conclusion multimodale défendable.

Actions:
- Traiter la couverture traces/logs (au moins un protocole comparable sur les 3 datasets).
- Rapporter les gains incrémentaux:
  - metrics-only
  - metrics+logs
  - metrics+traces
  - full MM

## P7 — Industrialisation minimale pour la fin de stage

Objectif: améliorer la reproductibilité et la maintenabilité.

Actions:
- Factoriser un noyau commun train/eval.
- Ajouter tests d’intégration protocole (seed/split/export).
- Écrire une note de décision "quand choisir STA vs Late-BN".

Livrable final stage:
- package reproductible + rapport avec conclusions robustes.

---

## 7) Plan d’exécution concret (6 semaines)

- **S1**: P1 diagnostic STA réel (instrumentation + premiers constats)
- **S2**: P2 protocole unifié + P3 stats
- **S3**: P4 ablations STA (lot 1)
- **S4**: P4 ablations STA (lot 2) + P5 Late-BN
- **S5**: P6 multimodal comparable
- **S6**: P7 industrialisation + rédaction finale

---

## 8) Réponse synthétique pour le stage

Tu as déjà une base solide et moderne (RCAEval, taxonomies, gray/cascade/drift, double architecture STA/Late-BN).  
Le travail restant est surtout de **convertir un succès simulé en preuve réelle robuste pour STA** via:
- diagnostic ciblé,
- protocole unifié,
- robustesse statistique,
- ablations propres,
- comparaisons multimodales strictes.

Position recommandée:
- **Late-BN** = baseline stable du mémoire.
- **STA** = contribution principale (innovation), avec critères d’acceptation explicites.
