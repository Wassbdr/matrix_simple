# Analyse Simulé vs Réel (STA vs GATv2)

## 1) Constat factuel

Le comportement de STA dépend fortement du régime de données:

- En simulé (benchmark BA historique, cf. `docs/internal/Week2.md`), les classements peuvent être favorables à des variantes non spatiales selon N et protocole.
- En réel RCAEval (runs `long` actuels, 5 seeds), STA et GATv2 sont proches sur `trainticket`, mais GATv2 garde un avantage sur `online-boutique`.

### Réel RCAEval - metrics only (5 seeds, 100 epochs)

| Dataset | STA Avg@1 | GATv2 Avg@1 | Delta STA-GATv2 |
|---|---:|---:|---:|
| trainticket | 0.960 | 0.950 | +0.010 |
| sockshop | 0.800 | 0.767 | +0.033 |
| online-boutique | 0.650 | 0.680 | -0.030 |

Source: `rca_pipeline/validation_runs/*__metrics_only.json`.

## 2) Pourquoi l’inversion sim/réel est plausible

### 2.1 Petit N réel (N=5)

Sur RCAEval RE2, les jeux utilisés ici sont à `N=5`. A cette échelle:

- le goulot `k` de STA est moins contraignant (souvent `k≈N`);
- l’avantage asymptotique `O(N + k²)` ne s’exprime pas;
- la qualité du score dépend surtout de la séparabilité temporelle des signaux.

Conclusion: à petit N, un modèle full-graph (GATv2) ou temporal-only peut rivaliser avec STA, voire le dépasser selon le dataset.

### 2.2 Variabilité selon distribution de pannes

`trainticket`/`sockshop` et `online-boutique` n’ont pas la même difficulté:

- `trainticket`: STA et GATv2 sont tous deux très forts;
- `online-boutique`: le gap est plus sensible et GATv2 est devant en metrics-only.

Cela suggère que la structure relationnelle utile n’est pas uniforme selon dataset, même à N identique.

## 3) Résultat T5: temporal-only sur réel

Nous avons exécuté l’ablation dédiée:

- Script: `rca_pipeline/ablation_temporal_only_real.py`
- Artefact: `rca_pipeline/validation_runs/ablation_temporal_real.json`
- Protocole: 5 seeds, 100 epochs, mêmes splits/seed pour STA et temporal-only.

### STA vs Temporal-only (Avg@1)

| Dataset | STA | Temporal-only | Delta STA-Temporal |
|---|---:|---:|---:|
| trainticket | 0.960 | 0.940 | +0.020 |
| sockshop | 0.850 | 0.860 | -0.010 |
| online-boutique | 0.650 | 0.690 | -0.040 |

Lecture:

- Le graphe aide légèrement sur `trainticket`.
- Le graphe n’apporte pas de gain net sur `sockshop`/`online-boutique` dans ce protocole.

## 4) Positionnement STA (thèse pragmatique)

1. STA reste compétitif sur réel et dépasse BARO sur `trainticket`/`sockshop`.
2. L’hypothèse “le graphe aide toujours” est trop forte; le gain dépend du dataset et du régime de taille.
3. Pour `N` très petit, l’argument principal de STA n’est pas la complexité, mais la robustesse/inductif biais; les gains peuvent être faibles.

## 5) Implications mémoire/rapport

- Formuler la contribution de manière conditionnelle:
  - STA est efficace et stable en réel,
  - mais l’apport spatial est dataset-dépendant à petit N.
- Mettre explicitement l’ablation temporal-only comme résultat de limitation honnête.
