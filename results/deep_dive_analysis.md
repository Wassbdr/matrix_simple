# Analyse approfondie des resultats benchmark RCA

Date: 2026-03-21
Source unique: artefacts actuellement disponibles dans le dossier results.

## 1. Perimetre et etat de completude

- Tables disponibles dans benchmark_full.csv:
  - main: 75 lignes
  - modality: 30 lignes
  - source: 5 lignes
- Echelles effectivement couvertes en main: N=20, N=50, N=100
- Ablations non executees: ablation_full.csv est vide
- La table source ne contient que synthetic_only (pas de rcaeval_only ni unified)

Conclusion de completude:
- Le benchmark principal est exploitable pour une analyse comparative de premier niveau.
- Les conclusions sur robustesse methodologique restent partielles sans les ablations A-E.

## 2. Resultats consolides (table main)

### N=20 (Top-1 moyen)

1. MLP: 0.3713 (+- 0.0069)
2. STA: 0.3037 (+- 0.0783)
3. LateBottleneck: 0.1162 (+- 0.0373)
4. RandomForest: 0.0721 (+- 0.0062)
5. FullGraphGCN: 0.0691 (+- 0.0136)

Lecture:
- MLP domine nettement en Top-1 avec tres faible variance.
- STA est 2e mais avec variance elevee entre seeds.

### N=50 (Top-1 moyen)

1. RandomForest: 0.0739 (+- 0.0147)
2. STA: 0.0557 (+- 0.0627)
3. MLP: 0.0287 (+- 0.0039)
4. FullGraphGCN: 0.0113 (+- 0.0066)
5. LateBottleneck: 0.0035 (+- 0.0019)

Lecture:
- Chute generale tres marquee des performances.
- RandomForest devient meilleur en Top-1, mais au prix de la latence la plus elevee.
- STA devient tres instable (ecart-type > moyenne).

### N=100 (Top-1 moyen)

1. RandomForest: 0.0348 (+- 0.0130)
2. MLP: 0.0259 (+- 0.0022)
3. FullGraphGCN: 0.0243 (+- 0.0111)
4. LateBottleneck: 0.0243 (+- 0.0318)
5. STA: 0.0032 (+- 0.0034)

Lecture:
- Degradation severe a grande echelle.
- STA s effondre presque completement a N=100.

## 3. Latence et compromis precision/cout

Latence mediane (ms, infer 100):
- MLP: ~0.106 a 0.128 (le plus rapide)
- LateBottleneck: ~1.6 a 1.8
- FullGraphGCN: ~2.1 a 2.25
- STA: ~3.8 a 6.2
- RandomForest: ~27.7 a 30.3 (le plus lent)

Implication:
- Si contrainte latency stricte: MLP reste le plus viable.
- Si contrainte Top-1 brute en N>=50: RF gagne, mais son cout inference est tres eleve.

## 4. Analyse modalite (N=50, STA et LateBottleneck)

- STA:
  - metrics: Top-1 0.0374
  - metrics_traces: Top-1 0.0278
  - full: Top-1 0.0557
- LateBottleneck:
  - metrics: Top-1 0.0026
  - metrics_traces: Top-1 0.0009
  - full: Top-1 0.0035

Interpretation:
- L enrichissement modal (full) aide STA de facon claire.
- Le meme enrichissement n apporte qu un gain marginal a LateBottleneck, qui reste tres faible.

## 5. Recurrent vs novel (gap MRR recurrent - novel)

Signaux observes:
- N=20: gap positif pour STA, MLP, GCN, Late; negatif pour RF.
- N=50: gap faible ou negatif pour la plupart, sauf STA legerement positif.
- N=100: gap redevient positif pour la plupart, sauf STA legerement negatif.

Interpretation:
- Le comportement sur la recurrence n est pas stable selon l echelle.
- STA perd son avantage recurrence a N=100.

## 6. Signaux d anomalie methodologique (importants)

1. Profil MLP atypique
- Excellent a N=20, puis forte chute des que N augmente.
- Cela peut indiquer un apprentissage de raccourcis non transferables.

2. Effondrement global a N=50/N=100
- Plusieurs modeles proches du hasard pour Top-1.
- Risque de mismatch entre objectif de ranking de root cause et structure reelle des donnees generees.

3. Incompletude du protocole final
- Absence des resultats d ablation empeche de valider les causes de degradation.
- La table source est incomplete (source_mode unique present: synthetic_only).

## 7. Hypotheses causales les plus probables

1. Sensibilite forte a l augmentation de cardinalite (N)
- Les representations apprises a petite echelle ne generalisent pas en grande topologie.

2. Distribution des incidents fortement desequilibree
- Top-1 peut etre trompeur si quelques root causes dominent massivement.

3. Training dynamics instables pour STA/Late
- Les logs montrent des phases supervisees avec acc parfois tres faible et early stop frequents.

4. Qualite du signal supervisé insuffisante pour certaines configurations
- Les modeles graphes peuvent manquer de signal discriminant robuste quand N augmente.

## 8. Recommandations prioritaires (court terme)

1. Completer les ablations A-E avant toute conclusion finale de recherche.
2. Ajouter des baselines de controle:
- majority-root baseline
- random weighted baseline
3. Ajouter des metriques macro:
- Macro-MRR par root cause
- Recall par root cause frequente vs rare
4. Comparer strictement a mix de source constant entre N=20/50/100.
5. Verrouiller les diagnostics d instabilite:
- courbes train/val par seed
- calibration logits
- sensibilite a k, batch size et class imbalance

## 9. Conclusion executive

Les resultats actuels montrent un benchmark principal exploitable mais encore fragile:
- MLP est tres performant a N=20 et tres rapide, mais ne passe pas l epreuve d echelle.
- RF devient meilleur en Top-1 a N>=50 mais son cout inference est prohibitif.
- STA reste competitif a N=20 mais perd fortement en robustesse et en performance quand N augmente.

A ce stade, il est prudent de traiter ces resultats comme une photographie intermediaire, pas comme une conclusion finale sur la superiorite architecturale des modeles.
