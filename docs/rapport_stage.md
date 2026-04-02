# Rapport de stage - RCA microservices

Date de consolidation: 2026-04-01

---

## Résumé

Ce rapport présente la transformation d'un prototype de Root Cause Analysis (RCA) en une plateforme technique défendable pour un contexte microservices industriel. Le travail couvre toute la chaîne: collecte de télémétrie, construction de jeux de données, entraînement de modèles graphe+temps, benchmark simulé/réel, et production d'artefacts auditable.

Le résultat central du stage n'est pas un score isolé, mais un système reproductible qui:
- exécute des protocoles explicites sur simulé, RCAEval réel et ObservIT,
- expose les limites de comparabilité externe,
- publie un manifeste canonique signé (checksum) pour verrouiller les chiffres de soutenance.

Contributions principales du stage:
1. formalisation opérationnelle du RCA comme problème inverse sous contraintes de parcimonie,
2. industrialisation d'un pipeline unifié collecte/train/eval/export,
3. comparaison structurée de familles de modèles (Late BN, STA, GATv2, MM),
4. mise en place d'une gouvernance des claims (manifest + comparability matrix + audit final).

Références structurantes:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)
- [rca_pipeline/benchmark_full_prepublish.json](../rca_pipeline/benchmark_full_prepublish.json)
- [rca_pipeline/benchmark_real_long_prepublish.json](../rca_pipeline/benchmark_real_long_prepublish.json)
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

---

## 1. Contexte industriel et enjeu RCA microservices

### 1.1 Problème opérationnel

Dans une architecture microservices, un incident visible (hausse latence, erreur 5xx, saturation) se propage souvent en cascade. Les symptômes sont distribués, hétérogènes et parfois décalés temporellement:
- métriques (Prometheus),
- traces distribuées,
- logs applicatifs.

Le diagnostic manuel est coûteux: plusieurs services paraissent anormaux alors qu'un seul est cause racine. Cette ambiguïté augmente le MTTR, le risque d'actions correctives inutiles, et la dette opérationnelle SRE.

### 1.2 Contraintes réelles du contexte

Le stage est contraint par des conditions classiques d'exploitation:
- labels causes racines incomplets ou coûteux en production,
- topologies évolutives et forte variabilité inter-services,
- dérive de distribution entre environnements simulés et réels,
- nécessité de décisions explicables et reproductibles pour la gouvernance.

### 1.3 Enjeu de valeur

L'enjeu n'est pas uniquement d'augmenter un Top-1 offline, mais de construire une chaîne RCA défendable:
- robuste aux changements de corpus,
- vérifiable par artefacts,
- utilisable pour soutenance technique, transfert équipe, puis publication externe.

---

## 2. Formulation mathématique du problème inverse

### 2.1 Système dynamique

On modélise le système microservices comme un graphe dirigé:
$$
\mathcal{G}=(\mathcal{V},\mathcal{E}),\quad |\mathcal{V}|=N
$$

État latent à l'instant $t$:
$$
X^{(t)}\in\mathbb{R}^{N\times d}
$$

Dynamique:
$$
X^{(t+1)} = f_\theta(A, X^{(t)}) + B U^{(t)}
$$

où:
- $A$ est la structure de dépendances,
- $f_\theta$ capte la propagation nominale,
- $U^{(t)}$ représente une perturbation exogène (anomalie), supposée parcimonieuse.

### 2.2 RCA comme problème inverse

On observe $Y^{(t)}$ (métriques/logs/traces), pas directement $X^{(t)}$ ni $U^{(t)}$. Le RCA consiste à retrouver le support minimal de la perturbation:
$$
\min_U \|U\|_0 \quad \text{sous contrainte} \quad Y^{(t)}\approx \hat{Y}^{(t)}
$$

Interprétation: identifier le plus petit ensemble de services capable d'expliquer la dérive observée.

### 2.3 Relaxation différentiable et objectif d'apprentissage

La contrainte $\ell_0$ est relaxée par un masque différentiable $M\in[0,1]^N$:
$$
Z^{(t)} = M\odot H^{(t)}
$$

avec objectif pratique:
$$
\mathcal{L}_{total}=\mathcal{L}_{InfoNCE}+\beta\|M\|_1
$$

et, selon les variantes, régularisation GIB variationnelle (KL Bernoulli) dans [rca_pipeline/losses.py](../rca_pipeline/losses.py).

### 2.4 Métriques de décision

Le problème est traité en ranking multiclasses (service cause racine):
- Avg@1, Avg@3, Avg@5,
- MRR,
- NDCG@3, NDCG@5.

Ces métriques sont harmonisées via [rca_pipeline/metrics_rcaeval.py](../rca_pipeline/metrics_rcaeval.py).

---

## 3. Évolution du projet (Phase 1 -> Phase 4)

### 3.1 Phase 1 - Fondations

Objectif: poser le cadre scientifique et le pipeline minimal.

Acquis:
- formalisation du RCA en problème inverse,
- premier pipeline graphe + temporel,
- premières métriques Top-K/MRR.

Référence: [docs/Week1.md](Week1.md)

### 3.2 Phase 2 - Passage à l'échelle

Objectif: sortir du cas micro-topologie et tester la scalabilité.

Acquis:
- introduction de STA (Select-Then-Analyze),
- générateur Barabasi-Albert scalable,
- hygiène expérimentale (split, normalisation, multi-seeds),
- vague d'ablations structurelles.

Référence: [docs/Week2.md](Week2.md)

### 3.3 Phase 3 - Validation réelle et multimodale

Objectif: confronter les hypothèses au réel.

Acquis:
- benchmark réel unifié (RCAEval + ObservIT),
- extension metrics/logs/traces,
- analyses simulé vs réel et multimodal.

Références:
- [docs/Week3.md](Week3.md)
- [docs/ANALYSIS_SIM_VS_REAL.md](ANALYSIS_SIM_VS_REAL.md)
- [docs/ANALYSIS_MULTIMODAL.md](ANALYSIS_MULTIMODAL.md)

### 3.4 Phase 4 - Hardening pré-soumission

Objectif: passer en mode soutenance, avec priorité à la crédibilité.

Acquis:
- guardrails critiques (warmup réel, statut MM dataset-level),
- export cohérence per-root-cause vs global,
- split_mode explicite simulé/réel,
- manifeste canonique + matrice de comparabilité.

Références:
- [docs/Week4.md](Week4.md)
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)

---

## 4. Architecture finale (Late BN, STA, MM)

### 4.1 Vue d'ensemble

La plateforme finale est organisée en couches:
- Données: collecte, fenetrage, alignement labels,
- Modèles: Late Bottleneck, STA, variantes GAT/MM,
- Entraînement/Évaluation: scripts benchmark simulé/réel/ablation,
- Gouvernance: validation protocol, manifest, comparability matrix.

### 4.2 Couche données

Composants principaux:
- [rca_pipeline/dataset_builder.py](../rca_pipeline/dataset_builder.py)
- [rca_pipeline/graph_builder.py](../rca_pipeline/graph_builder.py)
- [rca_pipeline/loaders/rcaeval_loader.py](../rca_pipeline/loaders/rcaeval_loader.py)
- [rca_pipeline/loaders/observit_loader.py](../rca_pipeline/loaders/observit_loader.py)
- [rca_pipeline/simulated_data_scalable.py](../rca_pipeline/simulated_data_scalable.py)

### 4.3 Modèle Late Bottleneck (Late BN)

Principe:
- encoder le graphe complet,
- appliquer ensuite le goulot d'information (GIB) pour filtrer.

Implémentations:
- [rca_pipeline/model.py](../rca_pipeline/model.py) (Late STGCN+GIB)
- [rca_pipeline/model_late_gat.py](../rca_pipeline/model_late_gat.py) (Late GATv2)

Atout: baseline forte sur certains corpus.

Limite: sensibilité possible à la régularisation sur grands graphes selon contexte de données.

### 4.4 Modèle STA (Select-Then-Analyze)

Principe:
- sélection différenciable de sous-graphe causal,
- analyse approfondie uniquement sur ce sous-ensemble.

Implémentation:
- [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py)
- [rca_pipeline/train_sta.py](../rca_pipeline/train_sta.py)

Intérêt: efficacité structurelle et meilleure focalisation causale.

### 4.5 Extensions multimodales (MM)

Principe:
- fusion metrics/logs/traces,
- évaluation stricte possible via mode mm_strict.

Implémentations:
- [rca_pipeline/model_sta_mm.py](../rca_pipeline/model_sta_mm.py)
- [rca_pipeline/model_late_gat_mm.py](../rca_pipeline/model_late_gat_mm.py)

Point critique: la comparabilité MM doit rester dataset-level explicite (exclusions possibles).

### 4.6 Orchestration entraînement/évaluation

Scripts clés:
- [rca_pipeline/benchmark.py](../rca_pipeline/benchmark.py)
- [rca_pipeline/benchmark_real.py](../rca_pipeline/benchmark_real.py)
- [rca_pipeline/benchmark_ablation.py](../rca_pipeline/benchmark_ablation.py)
- [rca_pipeline/validation_protocol.py](../rca_pipeline/validation_protocol.py)

---

## 5. Protocole expérimental (simulé, RCAEval réel, observit)

### 5.1 Volet simulé

But: évaluer la robustesse à l'échelle sur graphes BA.

Artefact principal:
- [rca_pipeline/benchmark_full_prepublish.json](../rca_pipeline/benchmark_full_prepublish.json)

Propriétés:
- évaluation multi-modèles,
- split tracé,
- reporting harmonisé (Avg@k, MRR, NDCG).

### 5.2 Volet réel RCAEval (metrics-only)

But: mesurer la généralisation sur datasets de référence.

Artefact principal:
- [rca_pipeline/benchmark_real_long_prepublish.json](../rca_pipeline/benchmark_real_long_prepublish.json)

Config dominante:
- seeds: 42,123,7
- epochs: 100
- max_cases: 90
- split_mode: varying_split_by_run_seed
- warmup STA réel: 0 (guardrail)

### 5.3 Volet réel RCAEval multimodal strict

But: évaluer en mode strict de couverture multimodale.

Artefact principal:
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

Règle: ne pas agréger globalement un dataset marqué skipped_with_reason.

### 5.4 Volet ObservIT

But: boucler la chaîne sur des campagnes cluster réelles contrôlées.

Composants:
- collecte/campagne: [rca_pipeline/dataset_builder.py](../rca_pipeline/dataset_builder.py)
- scripts: [scripts/run_observit_dataset.sh](../scripts/run_observit_dataset.sh), [scripts/run_observit_benchmark.sh](../scripts/run_observit_benchmark.sh)
- validation campagne: [rca_pipeline/validate_observit_campaign.py](../rca_pipeline/validate_observit_campaign.py)

### 5.5 Qualité statistique et limites de protocole

Les IC95 seed-level sont exportés, mais restent peu robustes quand $n=3$ (flag seed_level_bootstrap_reliable=false). Cette limite est explicitement conservée dans les artefacts et reprise dans l'analyse.

---

## 6. Résultats consolidés + limites explicites

### 6.1 Benchmark simulé (Top1)

Source canonique:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)

| N | STA | Late BN (STGCN+GIB) | Late BN (GATv2) | Full-Graph GCN |
|---|---:|---:|---:|---:|
| 20 | 0.7691 | 0.8145 | 0.8223 | 0.8379 |
| 50 | 0.6278 | 0.4561 | 0.5752 | 0.5263 |
| 100 | 0.5757 | 0.4208 | 0.5012 | 0.4799 |

Lecture:
- pas de vainqueur universel à toutes les échelles,
- Full-Graph GCN est meilleur à N=20,
- STA est meilleur à N=50 et N=100 dans ce snapshot canonique.

### 6.2 Benchmark réel metrics-only (RCAEval long)

Source:
- [rca_pipeline/benchmark_real_long_prepublish.json](../rca_pipeline/benchmark_real_long_prepublish.json)

| Dataset | STA Avg@1 | STA MRR | GATv2 Avg@1 | GATv2 MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9667 | 0.9833 | 0.9500 | 0.9750 |
| sockshop | 0.8000 | 0.8775 | 0.7667 | 0.8472 |
| online-boutique | 0.6833 | 0.7953 | 0.7167 | 0.8197 |

Lecture:
- STA domine sur trainticket et sockshop,
- GATv2 reste meilleur sur online-boutique,
- prudence statistique obligatoire (n=3 seeds).

### 6.3 Benchmark réel MM strict

Source:
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

| Dataset | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9167 | 0.9472 | 0.9167 | 0.9500 |
| online-boutique | 0.6167 | 0.7792 | 0.8000 | 0.8839 |

Cas sockshop:
- mm_coverage traces incomplet,
- statut dataset-level: skipped_with_reason dans la matrice de comparabilité,
- exclusion nécessaire de toute comparaison isopérimètre MM globale.

### 6.4 Test critique warmup réel (A/B)

Sources:
- [results_transmission/warmup_real_trainticket_w0.json](../results_transmission/warmup_real_trainticket_w0.json)
- [results_transmission/warmup_real_trainticket_w10.json](../results_transmission/warmup_real_trainticket_w10.json)

Résultat:
- warmup=0 -> STA Avg@1 = 0.7333
- warmup=10 -> STA Avg@1 = 0.0000

Conclusion:
- warmup contrastif réel instable dans cette configuration,
- guardrail warmup réel désactivé par défaut justifié.

### 6.5 Ablations (snapshot prepublish)

Source:
- [rca_pipeline/ablation_prepublish.json](../rca_pipeline/ablation_prepublish.json)

Points saillants:
- A3 spatial-only est nettement en retrait à N=100 (Top1 ~0.4775),
- A4 temporal-only est très élevé sur ce snapshot à N=100 (Top1 ~0.9563),
- sensibilité à k (STA, N=100): Top1 passe de ~0.9243 (k=10) à ~0.9492 (k=30),
- protocole D (warmup/k-annealing) montre une forte dépendance au contexte: sur ce snapshot, D2/D3 dépassent D1.

Interprétation:
- les conclusions d'architecture doivent être formulées comme des tendances conditionnelles au corpus, pas comme des lois universelles.

### 6.6 Limites explicites

Menaces internes:
1. puissance statistique limitée (souvent n=3),
2. forte sensibilité à certains hyperparamètres/protocoles,
3. dérive encore peu informative sur corpus long (drop ratio proche de zero).

Menaces externes:
1. comparabilité baseline publiée seulement partielle,
2. MM strict non isopérimètre sur tous les datasets,
3. extension des métriques failure-type réel encore incomplète.

---

## 7. Analyse transversale et discussion

### 7.1 Lecture scientifique: ce que disent vraiment les résultats

Les résultats consolidés montrent une réalité importante: il n'existe pas de modèle universellement dominant.

- Sur simulé, l'ordre des modèles varie avec l'échelle du graphe (N=20 vs N=50/100).
- Sur réel metrics-only, STA est devant sur 2 datasets (trainticket, sockshop), mais GATv2 reste devant sur online-boutique.
- Sur MM strict, les performances restent compétitives mais la couverture multimodale inégale (ex. sockshop) interdit les conclusions globales isopérimètres.

Interprétation: la bonne question n'est pas "quel est le meilleur modèle absolu", mais "quel modèle est le plus robuste pour un contexte de données donné".

### 7.2 Analyse de transfert simulé -> réel

Le stage confirme un point classique en ML système: les gains observés en simulé ne se transfèrent pas mécaniquement au réel.

Facteurs explicatifs identifiés dans le projet:
1. topologies et régimes de bruit différents entre simulation BA et traces d'exploitation,
2. qualité/coverage hétérogène des modalités (surtout traces) selon dataset,
3. sensibilité des protocoles d'entraînement (warmup, k-sélection, split mode) au corpus.

Conséquence méthodologique: les claims de performance doivent être présentées par scénario de données, et non comme un classement global unique.

### 7.3 Analyse de robustesse et sensibilité protocolaire

Trois points de robustesse émergent:
1. warmup réel STA: le test A/B (w0 vs w10) montre une dégradation drastique avec warmup actif, ce qui justifie le guardrail warmup=0 en réel;
2. sensibilité à k (STA): un k trop agressif dégrade la performance, ce qui impose un tuning explicite par échelle/dataset;
3. précision statistique: avec n=3 seeds, les IC95 seed-level existent mais ne suffisent pas pour des claims externes forts.

Lecture opérationnelle: le système est prêt pour des décisions techniques de soutenance, mais la robustesse publication-grade exige une campagne multi-seeds plus large.

### 7.4 Analyse des risques de décision et mitigations

Risques principaux:
1. inflation de claim: sur-interpréter un résultat local comme conclusion générale;
2. erreur de triage opérationnel: promouvoir un modèle unique sur tous les corpus;
3. confusion de comparabilité: mélanger des périmètres non isopérimètres (notamment MM strict).

Mitigations déjà implémentées:
1. manifest canonique signé et checksums pour figer les chiffres;
2. comparability matrix pour qualifier explicitement ce qui est comparable ou non;
3. export per-root-cause/global pour détecter les incohérences de synthèse.

### 7.5 Cadre de décision recommandé pour l'exploitation

| Contexte | Constat empirique | Décision recommandée |
|---|---|---|
| RCAEval metrics-only | STA gagne sur 2/3 datasets | STA par défaut, GATv2 en challenger actif |
| Dataset fortement hétérogène (online-boutique) | GATv2 devant en metrics-only et MM | Conserver un routage par dataset/corpus |
| MM coverage partielle | statut skipped_with_reason sur certains jeux | Interdire l'agrégation globale MM et fallback metrics-only |
| Training réel sensible | warmup réel peut faire chuter les scores | Garder warmup=0 tant qu'aucune preuve robuste contraire |

Ce cadre transforme le benchmark en règles de pilotage concrètes pour un usage SRE/MLOps.

---

## 8. Gouvernance, reproductibilité, comparabilité

### 8.1 Source de vérité et intégrité

Le manifeste canonique verrouille:
- la liste d'artefacts,
- les commandes de génération,
- les hash SHA256,
- le commit de référence.

Références:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/canonical_results_manifest.json.sha256](../results_transmission/canonical_results_manifest.json.sha256)

### 8.2 Protocole de rerun

Le paquet de reproduction décrit les commandes de rerun simulé/réel/MM/ablation/manifest:
- [results_transmission/README_resultats.txt](../results_transmission/README_resultats.txt)

### 8.3 Comparabilité externe

La matrice dédiée qualifie explicitement les claims:
- metrics-only vs baselines publiées: partially_comparable,
- MM strict vs baselines publiées: not_isoperimeter.

Référence:
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)

### 8.4 Validation logicielle et garde-fous

Éléments de crédibilité logicielle:
- tests pré-soumission passés: 35,
- guardrails critiques implantés (warmup réel, statut MM dataset-level, cohérence per-root/global),
- audit final formalisé.

### 8.5 Composants complémentaires implantés

Au-delà des modules principaux décrits dans les sections précédentes,
le pipeline intègre plusieurs briques logicielles additionnelles qui
participent à la robustesse et à la profondeur du système.

#### 8.5.1 Encodeur temporel S4 (Structured State Space)

Le modèle STA dispose d'un encodeur temporel alternatif fondé sur les
State Space Models (SSM), implanté dans la classe `S4TemporalBlock`.
Contrairement à l'encodeur Conv1d (filtre à réponse finie), le S4
modélise explicitement la transition d'état formalisée dans la
section 2:

    h(t+1) = A_s * h(t) + B_s * x(t)
    y(t)   = C_s * h(t) + D_s * x(t)

L'initialisation s'appuie sur les paramètres HiPPO-LegS pour les
dépendances longue-portée. Le calcul s'effectue en mode convolutif
via FFT en O(W log W), où W est la taille de la fenêtre temporelle.
Le noyau de convolution est construit à partir des puissances de la
matrice d'état discrétisée A_bar = exp(A * dt).

Implémentation: [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py) (classe `S4TemporalBlock`, lignes 195-286).

#### 8.5.2 Sélecteur hiérarchique (HierarchicalSelector)

Le pipeline propose également un sélecteur à deux étages, alternatif
au `STESelector` standard:

- Étage 1 (grossier): un MLP léger note les N nœuds et sélectionne
  les top-N/2 via Gumbel-sigmoid.
- Étage 2 (fin): un MLP plus expressif (2 couches + ReLU) affine la
  sélection sur les survivants pour produire les top-k finaux.

Le flux de gradient traverse les deux étages conjointement
(produit des probabilités coarse * fine au STE final).
La complexité est O(N) + O(N/2) + O(k^2) — identique à la borne
asymptotique du STE simple, mais avec un meilleur ensemble de
candidats lorsque N >> k.

Implémentation: [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py) (classe `HierarchicalSelector`, lignes 432-525).

#### 8.5.3 Perte sensible à la cascade (CascadeAwareLoss)

Trois variantes de régularisation cascade sont implantées:

1. `CascadeAwareLoss`: contrainte hinge sur la distance topologique
   (hop). Le score du nœud cause-racine doit dépasser celui des
   voisins à 1-hop d'une marge m1, et ceux-ci doivent dépasser les
   2-hops d'une marge m2. Inspirée de Chain-of-Event (FSE 2024).
2. `CausalCascadeAwareLoss`: les marges sont proportionnelles aux
   poids causaux des arêtes (issues du CausalGraphBuilder), de sorte
   que les liens causalement forts imposent un écart de score plus large.
3. `LearnedPropagationCascadeLoss`: la matrice de propagation B
   est un `nn.Parameter` apprenable, initialisée à partir du graphe
   causal. Cela relie directement la perte à la formalisation
   X(t+1) = f(A, X(t)) + B*U(t) de la section 2.

Implémentation: [rca_pipeline/cascade_aware_loss.py](../rca_pipeline/cascade_aware_loss.py).

#### 8.5.4 Détection de dérive (DriftDetector ADWIN)

Le module `drift_detector.py` implante l'algorithme ADWIN
(Bifet & Gavalda, SDM 2007) pour détecter les changements de
distribution dans les métriques Kubernetes en streaming.

Principe:
- Chaque métrique est surveillée indépendamment par un détecteur ADWIN(delta).
- Si une fraction configurable (par défaut 50 %) des métriques
  dérivent simultanément, un événement `suspected_deployment` est déclaré.
- L'analyse de cause-racine est alors suspendue pendant une fenêtre
  de cooldown (30 pas par défaut) pour éviter les faux positifs.

Le détecteur repose sur une représentation compressée en buckets
(complexité O(log n) amortie par mise à jour, espace O(log n)).

Le module inclut également un mécanisme de Test-Time Adaptation (TTA):
lorsqu'une dérive est détectée, les statistiques BatchNorm sont
réinitialisées et recalibrées sur les fenêtres les plus récentes,
sans réentraîner le modèle complet.

Implémentation: [rca_pipeline/drift_detector.py](../rca_pipeline/drift_detector.py).

#### 8.5.5 Détection de pannes grises (GrayFailureDetector)

Inspiré de GrayScope (FSE 2024) et GREYBOUND (ATC 2025), ce module
détecte les défaillances où les métriques individuelles restent dans
les seuils SLA mais où leur comportement conjoint se dégrade.

Quatre mécanismes sont proposés:
1. Ratio P99/P50: détecte les micro-stutters (CPU steal, contention L3,
   pauses GC, throttle CFS).
2. Décorrélation inter-métriques: chute de la corrélation de Pearson
   (cpu, latence) sur fenêtre glissante.
3. Tendance monotone: test de Mann-Kendall sur la mémoire, détectant
   les Metastable Failures naissantes (Bronson et al., OSDI 2022).
4. Signal Limplock: ratio queue_depth_rate / cpu_usage élevé
   (Google 2023) — la file d'attente croît tandis que le CPU reste plat.

Implémentation: [rca_pipeline/gray_failure_detector.py](../rca_pipeline/gray_failure_detector.py).

#### 8.5.6 Construction de graphe causal (CausalGraphBuilder)

Ce module construit un DAG causal à partir des séries temporelles
en mode sain, avec trois algorithmes et un mécanisme de fallback
automatique:

1. PC (causallearn): tests d'indépendance conditionnelle (alpha=0.05),
   avec test de violation de couverture de Markov. Bascule sur FCI
   si plus de 20 % des tests sont rejetés.
2. Granger (statsmodels): causalité lag-based, utilisé en fallback
   si moins de 50 fenêtres saines sont disponibles.
3. NOTEARS-LR (Zheng et al., NeurIPS 2018): optimisation continue
   de la contrainte d'acyclicité tr(e^(W⊙W)) - N = 0, limitée
  à N <= 200 nœuds.

En mode hybride (par défaut), l'edge_index final est l'union des
arêtes structurelles (BA) et causales, pondérées respectivement
à 1.0 et 2.0.

Implémentation: [rca_pipeline/causal_graph_builder.py](../rca_pipeline/causal_graph_builder.py).

#### 8.5.7 Taxonomie d'anomalies (AnomalyTaxonomy)

Module d'enrichissement de labels implantant trois classifications
complémentaires:

1. Chandola (ACM CS 2009): point / contextuel / collectif.
2. Zamanzadeh Darban (ACM CS 2024): extensions temporelles
   (global/local/trend/seasonal) et inter-métriques
   (global/local/temporal-intermetric).
3. CRI-Model (Reichert 2018): détection de phase
   Change / Rupture / Impact.

Ce module sert de feature engineering pour le détecteur de pannes
grises et d'enrichissement de labels pour le générateur simulé.

Implémentation: [rca_pipeline/anomaly_taxonomy.py](../rca_pipeline/anomaly_taxonomy.py).

#### 8.5.8 Extracteurs multimodaux (logs et traces)

Deux modules d'extraction de features assurent la transformation
des données brutes textuelles en vecteurs numériques:

- `log_feature_extractor.py`: extrait 8 features par nœud
  (error_freq, warn_freq, log_rate, error_burst, unique_errors,
  oom_flag, restart_flag, timeout_flag) à partir de lignes de log
  brutes. Méthode inspirée de Chain-of-Event (FSE 2024).
- `trace_feature_extractor.py`: extrait 6 features par nœud
  (call_depth, error_span_ratio, p99_span_ms, upstream_errors,
  downstream_errors, retry_count) à partir de spans distribués.
  Méthode inspirée de RUN (ASE 2024).

Chaque extracteur retourne np.zeros lorsque la modalité est
indisponible (fallback gracieux préservant la compatibilité
avec les modèles unimodaux).

Implémentations:
- [rca_pipeline/log_feature_extractor.py](../rca_pipeline/log_feature_extractor.py)
- [rca_pipeline/trace_feature_extractor.py](../rca_pipeline/trace_feature_extractor.py)

#### 8.5.9 Perte multi-root-cause

La fonction de perte standard suppose une cause-racine unique
par fenêtre (cross-entropy, InfoNCE). Une extension BCE multi-label
est également implantée dans `losses.py` pour supporter les cas
||U||_0 > 1 (anomalies à causes racines multiples simultanées).

Implémentation: [rca_pipeline/losses.py](../rca_pipeline/losses.py) (lignes 147-201).

#### 8.5.10 Infrastructure Kubernetes et génération de charge

Le déploiement du pipeline en conditions réelles s'appuie sur:

- `scripts/`: 8 scripts d'automatisation couvrant le déploiement
  des microservices cibles (TrainTicket, SockShop, Online Boutique),
  l'injection de pannes via Chaos Mesh, la collecte Prometheus,
  et la génération de datasets.
- `k8s/`: manifestes Kubernetes pour le déploiement de la stack
  d'observabilité (Prometheus, Chaos Mesh).
- `locustfile.py`: script Locust pour générer du trafic utilisateur
  réaliste avec des profils de charge variables, nécessaire pour
  obtenir des métriques de latence significatives pendant la
  collecte de données réelles.

---

## 9. Conclusion générale

Ce stage a permis de concevoir, d'implémenter et de valider un pipeline
complet d'analyse de cause-racine (RCA) pour les architectures
microservices, depuis la collecte des métriques Kubernetes jusqu'à
la production de diagnostics explicables.

Le travail réalisé couvre quatre axes principaux:

Premièrement, une formalisation mathématique rigoureuse du problème
de RCA sous la forme d'un système dynamique X(t+1) = f_theta(A, X(t))
+ B*U(t), où la localisation de la cause-racine se traduit par
l'identification du support de la perturbation U(t). Cette formulation
a donné naissance à deux familles d'architectures complémentaires:
le Late Bottleneck (STGCN+GIB et GATv2) qui encode d'abord le graphe
complet avant de filtrer par goulot d'information, et le Select-then-
Analyze (STA) qui sélectionne d'abord un sous-graphe causal avant
de l'analyser en profondeur. La complexité du STA en O(N + k^2)
le rend structurellement plus adapté aux topologies de grande taille
que l'approche Late BN en O(N^2).

Deuxièmement, une couverture expérimentale multi-volets: 
benchmarks simulés sur des topologies Barabasi-Albert de N = 5 à 200
nœuds, benchmarks sur trois jeux de données réels RCAEval (TrainTicket,
SockShop, Online Boutique), et collecte de données propres via
ObservIT. Les résultats confirment qu'il n'existe pas de vainqueur
universel: le Late BN domine à petite échelle (N = 20), tandis que le
STA s'impose progressivement à mesure que la topologie grandit
(N >= 50). Les extensions multimodales (métriques + logs + traces)
améliorent les performances sur les datasets où la couverture
des modalités est suffisante, mais leur comparabilité reste
limitée par l'absence de baselines publiées multimodales.

Troisièmement, une suite d'outils d'analyse et de robustesse:
détection de dérive ADWIN pour filtrer les faux positifs liés aux
déploiements, détection de pannes grises (GrayScope/Limplock),
construction de graphes causaux (PC/FCI/Granger/NOTEARS), taxonomie
d'anomalies (Chandola, Zamanzadeh, CRI), et pertes sensibles à la
cascade (hinge topologique et propagation apprenable).

Quatrièmement, un cadre de gouvernance des résultats conçu pour
garantir la reproductibilité et l'honnêteté des claims: manifeste
canonique signé (SHA256 + commit), matrice de comparabilité
isopérimètre qualifiant explicitement les verdicts (partially_comparable,
not_isoperimeter), guardrails automatisés (warmup réel, statut MM
dataset-level, cohérence per-root/global), et protocole de validation
à trois paliers (smoke, long, paper).

Le pipeline totalise environ 15 000 lignes de code Python structurées
en 61 modules, accompagnées de 35 tests unitaires couvrant les
invariants critiques. L'ensemble des artefacts de résultats est
tracé, versionné et reproductible via des commandes documentées.

Les limitations reconnues ont été explicitées tout au long du
rapport: puissance statistique limitée avec n = 3 seeds sur certains
runs, absence de taxonomie de type de défaillance sur les jeux réels,
comparabilité externe indicative plutôt que strictement isopérimètre,
et instabilité du warmup contrastif STA sur données réelles (justifiant
le guardrail warmup = 0 par défaut). Ces limites, loin d'affaiblir
le travail, témoignent d'une démarche de recherche exigeante où la
transparence méthodologique est considérée comme un prérequis à la
crédibilité des résultats.

---

## Annexes - Cartographie des composants principaux

- Entraînement/eval:
  - [rca_pipeline/train.py](../rca_pipeline/train.py)
  - [rca_pipeline/train_sta.py](../rca_pipeline/train_sta.py)
  - [rca_pipeline/evaluate.py](../rca_pipeline/evaluate.py)
- Modèles:
  - [rca_pipeline/model.py](../rca_pipeline/model.py)
  - [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py) (inclut S4TemporalBlock, HierarchicalSelector)
  - [rca_pipeline/model_late_gat.py](../rca_pipeline/model_late_gat.py)
  - [rca_pipeline/model_sta_mm.py](../rca_pipeline/model_sta_mm.py)
  - [rca_pipeline/model_late_gat_mm.py](../rca_pipeline/model_late_gat_mm.py)
- Pertes:
  - [rca_pipeline/losses.py](../rca_pipeline/losses.py)
  - [rca_pipeline/cascade_aware_loss.py](../rca_pipeline/cascade_aware_loss.py)
- Analyse et détection:
  - [rca_pipeline/drift_detector.py](../rca_pipeline/drift_detector.py)
  - [rca_pipeline/gray_failure_detector.py](../rca_pipeline/gray_failure_detector.py)
  - [rca_pipeline/causal_graph_builder.py](../rca_pipeline/causal_graph_builder.py)
  - [rca_pipeline/anomaly_taxonomy.py](../rca_pipeline/anomaly_taxonomy.py)
- Extracteurs multimodaux:
  - [rca_pipeline/log_feature_extractor.py](../rca_pipeline/log_feature_extractor.py)
  - [rca_pipeline/trace_feature_extractor.py](../rca_pipeline/trace_feature_extractor.py)
- Gouvernance:
  - [rca_pipeline/results_manifest.py](../rca_pipeline/results_manifest.py)
  - [rca_pipeline/comparability_matrix.py](../rca_pipeline/comparability_matrix.py)
  - [rca_pipeline/validation_protocol.py](../rca_pipeline/validation_protocol.py)

