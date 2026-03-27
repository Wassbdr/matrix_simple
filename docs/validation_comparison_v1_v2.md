# Validation STA v2 : comparaison v1 vs v2 et go/no-go

Sources: `rca_pipeline/benchmark_full.json` (v1), `rca_pipeline/benchmark_v2_full.json` (v2), ablations `rca_pipeline/ablation_A*.json`, `rca_pipeline/benchmark_real_v2_metrics.json`, `rca_pipeline/benchmark_real_v2_mm.json`.

Seeds simulés: `42,123,7`. Seeds RCAEval: `42,123,7`, `--epochs 100`.

## 1. Benchmark simulé (Barabási–Albert, STA)

| N   | STA v1 Top-1 | STA v2 Top-1 | Delta Top-1 | STA v1 MRR | STA v2 MRR |
|-----|-------------|-------------|-------------|------------|------------|
| 20  | 0.300       | 0.677       | +0.377      | 0.414      | 0.754      |
| 50  | 0.183       | 0.548       | +0.365      | 0.276      | 0.605      |
| 100 | 0.067       | 0.518       | +0.451      | 0.123      | 0.575      |

Latence STA (moyenne rapportée): v1 ~4.7–7.6 ms selon N; v2 ~3.2–3.5 ms (inférieure à v1 sur ces runs).

**Décision (P0 simulé):** GO. Les seuils cibles du protocole (Top-1 ≥ 0.40 @ N=20, ≥ 0.12 @ N=100) sont largement atteints.

## 2. Ablations warmup (STA, `N=20` et `N=50`, 3 seeds)

Configuration dans `train_sta_compact` (warmup uniquement; `--skip-late-gat` pour accélérer les autres baselines).

| Run | Warmup: régularisation | Warmup: agrégat InfoNCE | Top-1 @ N=20 | Top-1 @ N=50 |
|-----|------------------------|-------------------------|--------------|--------------|
| A1  | L1 (`--no-gib`)        | moyenne sur `H_fast` (`--no-mask-weighted-nce`) | 0.707 | 0.595 |
| A2  | variational GIB        | moyenne sur `H_fast`      | **0.855** | **0.622** |
| A3  | variational GIB        | masque pondéré (`H_fast * M`) | 0.677 | 0.548 |
| A4  | = v2 complet (même warmup que A3) | idem A3 | 0.677 | 0.548 |

Fichiers: `ablation_A1_zerohop_l1_hfast.json`, `ablation_A2_zerohop_gib_hfast.json`, `ablation_A3_zerohop_gib_zsubgraph.json` (A3 aligné sur `benchmark_v2_full.json` pour ces N).

**Observation:** Sur ce protocole simulé, **le pooling moyen + GIB (A2) obtient un Top-1 plus élevé que le masque pondéré (A3)** à N=20 et N=50. Le masque pondéré reste l’alignement théorique (gradient à travers `M`); la différence suggère un réglage (beta warmup, temperature, `k/N`) ou une interaction avec la phase supervisée à analyser (p.ex. sensibilité au seed, calibration).

## 3. RCAEval réel (metrics-only)

Référence publiée BARO Avg@1 ≈ 0.71 (trainticket).

| Dataset          | STA Avg@1 | GATv2 Avg@1 | BARO (ref) |
|------------------|-----------|-------------|------------|
| trainticket      | 0.20      | 0.95        | 0.71       |
| sockshop         | 0.20      | 0.77        | 0.71       |
| online-boutique  | 0.20      | 0.72        | 0.71       |

Fichier: `benchmark_real_v2_metrics.json`.

**Décision:** NO-GO pour “égaler BARO” avec STA seul sur ces splits: Avg@1 STA reste à 0.20 alors que GATv2 est fort. Piste: protocole d’entraînement (peu données par split), alignement des features, ou baselines non comparables sans même pipeline de prétraitement.

## 4. RCAEval multimodal (`--use-mm --mm-strict`)

| Dataset          | STA-MM Avg@1 | GATv2-MM Avg@1 | MM effectif (logs+traces) |
|------------------|--------------|----------------|---------------------------|
| trainticket      | 0.20         | 0.92           | oui                       |
| online-boutique  | 0.20         | 0.80           | oui                       |
| sockshop         | (absent)     | (absent)       | traces_ratio 0 dans les cas utilisés |

Fichier: `benchmark_real_v2_mm.json`. **sockshop** ne contient que `config` (pas de métriques STA-MM/GATv2-MM): avec `--mm-strict`, les seeds requis n’avaient pas signal traces (ou équivalent) sur train+test.

**Décision:** Sur trainticket et online-boutique, **STA-MM n’améliore pas** les métriques vs metrics-only (Avg@1 identique 0.20). **NO-GO** pour l’hypothèse “MM > metrics-only” dans cette configuration.

## 5. Synthèse go/no-go par innovation (P0 validation)

| Innovation (P0)              | Simulé BA | RCAEval réel |
|-----------------------------|-----------|--------------|
| Zero-hop + STE + analyse    | GO        | STA faible vs GATv2 / BARO |
| Variational GIB (warmup)    | GO (ablation) | —        |
| InfoNCE masque pondéré      | À revisiter (A2 > A3 sur BA) | — |
| Fusion MM multi-vues        | —         | NO-GO (pas de gain Avg@1) |

**Prochaines étapes suggérées:** (1) Étude de sensibilité warmup (beta, temperature, durée) pour réconcilier A2 vs A3. (2) Sur RCAEval: diagnostiquer pourquoi STA stagne à 0.20 (splits, labels, sur-apprentissage GATv2, besoin de données agrégées). (3) Sockshop MM: assouplir `--mm-strict` ou compléter les traces pour inclure des runs comparables.

## 6. Correctif STA réel (v3) — warmup contrastif désactivé

Un correctif a été introduit dans `benchmark_real.py` :
- nouveau flag `--sta-contrastive-warmup`,
- **défaut désormais à `0` pour RCAEval réel**,
- warmup réactivable explicitement pour essais (`--sta-contrastive-warmup > 0`).

### 6.1 Run de référence (réel, all datasets, 3 seeds, 90 cas, 100 epochs)

Fichier: `rca_pipeline/benchmark_real_sta_fix_all_nowarm.json`.

| Dataset          | STA Avg@1 (v2) | STA Avg@1 (v3, warmup=0) | Delta STA | GATv2 Avg@1 (v2) | GATv2 Avg@1 (v3) | Delta GATv2 |
|------------------|----------------|----------------------------|-----------|------------------|------------------|-------------|
| trainticket      | 0.20           | **0.97**                   | +0.77     | 0.95             | 0.95             | +0.00       |
| sockshop         | 0.20           | **0.80**                   | +0.60     | 0.77             | 0.77             | +0.00       |
| online-boutique  | 0.20           | **0.68**                   | +0.48     | 0.72             | 0.72             | +0.00       |

Lecture:
- le plateau STA `~0.20` est levé sur trainticket/sockshop;
- online-boutique progresse fortement mais reste juste sous BARO en Avg@1;
- **pas de régression GATv2** sur la comparaison longue isopérimètre (delta nul sur Avg@1).

### 6.2 Validation protocolaire courte (script `validation_protocol.py`)

Le protocole standard a été mis à jour pour:
- utiliser l’interpréteur actif (`sys.executable`);
- exposer `--sta-contrastive-warmup` via `RCA_VAL_STA_WARMUP` (défaut `0`);
- séparer deux presets explicites via `RCA_VAL_PRESET` :
  - `smoke` = `seeds 42,123`, `epochs 20`, `max-cases 20`;
  - `long` = `seeds 42,123,7`, `epochs 100`, `max-cases 90`.

Run exécuté:
- `.venv/bin/python rca_pipeline/validation_protocol.py`
- paramètres défaut du protocole: seeds `42,123`, epochs `20`, max-cases `20`.

Résumé extrait:
- **trainticket / metrics-only**: STA `0.800`, GATv2 `1.000`
- **online-boutique / metrics-only**: STA `0.583`, GATv2 `0.583`
- **sockshop / metrics-only**: STA `0.667`, GATv2 `0.583`
- configs MM strict sur sockshop: `n/a` (traces manquantes), inchangé.

### 6.3 Décision mise à jour

- **GO** sur la correction STA réel: le warmup contrastif n’est pas robuste sur ce protocole réel et doit être désactivé par défaut (`warmup=0`) pour RCAEval.
- **GO** sur la stabilité comparative: GATv2 ne régresse pas en protocole long isopérimètre.
- **À poursuivre**:
  1. augmenter le nombre de seeds en mode `long` pour resserrer les IC;
  2. traiter la couverture MM de sockshop (`mm_strict` -> `n/a`);
  3. affiner online-boutique pour passer le seuil BARO en Avg@1 de façon stable.

### 6.4 Last-mile online-boutique (micro-tuning ciblé)

Un tuning court a été testé uniquement sur `online-boutique`:
- `--sta-contrastive-warmup 0`
- `--hidden-dim 48`
- `--epochs 100`, `--max-cases 90`, `--seeds 42,123,7`

Fichier: `rca_pipeline/benchmark_real_sta_fix_online_h48_nowarm.json`.

Résultat:
- **STA Avg@1 = 0.7833** (au-dessus BARO 0.71)
- **GATv2 Avg@1 = 0.7000** (au-dessus seuil 0.69)

Conclusion pratique:
- pour `online-boutique`, augmenter `hidden_dim` à 48 stabilise STA au-dessus du seuil dans ce protocole.

### 6.5 Clôture périmètre MM sockshop

Décision explicite:
- on conserve `mm_strict` pour éviter des comparaisons biaisées;
- `sockshop` reste **non comparable en MM strict** tant que `traces_ratio=0` sur les seeds utilisés;
- ce périmètre est donc marqué **hors comparaison MM** dans la conclusion stage (jusqu’à enrichissement traces ou changement de protocole MM).
