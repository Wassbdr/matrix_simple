# Memoire de stage - RCA microservices

Date de consolidation: 2026-04-01

---

## Resume

Ce memoire presente la transformation d'un prototype de Root Cause Analysis (RCA) en une plateforme technique defendable pour un contexte microservices industriel. Le travail couvre toute la chaine: collecte de telemetry, construction de jeux de donnees, entrainement de modeles graphe+temps, benchmark simule/reel, et production d'artefacts auditable.

Le resultat central du stage n'est pas un score isole, mais un systeme reproductible qui:
- execute des protocoles explicites sur simule, RCAEval reel et ObservIT,
- expose les limites de comparabilite externe,
- publie un manifeste canonique signe (checksum) pour verrouiller les chiffres de soutenance.

Contributions principales du stage:
1. formalisation operationnelle du RCA comme probleme inverse sous contraintes de parcimonie,
2. industrialisation d'un pipeline unifie collecte/train/eval/export,
3. comparaison structuree de familles de modeles (Late BN, STA, GATv2, MM),
4. mise en place d'une gouvernance des claims (manifest + comparability matrix + audit final).

References structurantes:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)
- [rca_pipeline/benchmark_full_prepublish.json](../rca_pipeline/benchmark_full_prepublish.json)
- [rca_pipeline/benchmark_real_long_prepublish.json](../rca_pipeline/benchmark_real_long_prepublish.json)
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

---

## 1. Contexte industriel et enjeu RCA microservices

### 1.1 Probleme operationnel

Dans une architecture microservices, un incident visible (hausse latence, erreur 5xx, saturation) se propage souvent en cascade. Les symptomes sont distribues, heterogenes et parfois decales temporellement:
- metriques (Prometheus),
- traces distribuees,
- logs applicatifs.

Le diagnostic manuel est couteux: plusieurs services paraissent anormaux alors qu'un seul est cause racine. Cette ambiguite augmente le MTTR, le risque d'actions correctives inutiles, et la dette operationnelle SRE.

### 1.2 Contraintes reelles du contexte

Le stage est contraint par des conditions classiques d'exploitation:
- labels causes racines incomplets ou couteux en production,
- topologies evolutives et forte variabilite inter-services,
- derive de distribution entre environnements simules et reels,
- necessite de decisions explicables et reproductibles pour la gouvernance.

### 1.3 Enjeu de valeur

L'enjeu n'est pas uniquement d'augmenter un Top-1 offline, mais de construire une chaine RCA defendable:
- robuste aux changements de corpus,
- verifiable par artefacts,
- utilisable pour soutenance technique, transfert equipe, puis publication externe.

---

## 2. Formulation mathematique du probleme inverse

### 2.1 Systeme dynamique

On modelise le systeme microservices comme un graphe dirige:
$$
\mathcal{G}=(\mathcal{V},\mathcal{E}),\quad |\mathcal{V}|=N
$$

Etat latent a l'instant $t$:
$$
X^{(t)}\in\mathbb{R}^{N\times d}
$$

Dynamique:
$$
X^{(t+1)} = f_\theta(A, X^{(t)}) + B U^{(t)}
$$

ou:
- $A$ est la structure de dependances,
- $f_\theta$ capte la propagation nominale,
- $U^{(t)}$ represente une perturbation exogene (anomalie), supposee parcimonieuse.

### 2.2 RCA comme probleme inverse

On observe $Y^{(t)}$ (metriques/logs/traces), pas directement $X^{(t)}$ ni $U^{(t)}$. Le RCA consiste a retrouver le support minimal de la perturbation:
$$
\min_U \|U\|_0 \quad \text{sous contrainte} \quad Y^{(t)}\approx \hat{Y}^{(t)}
$$

Interpretation: identifier le plus petit ensemble de services capable d'expliquer la derive observee.

### 2.3 Relaxation differentiable et objectif d'apprentissage

La contrainte $\ell_0$ est relaxee par un masque differentiable $M\in[0,1]^N$:
$$
Z^{(t)} = M\odot H^{(t)}
$$

avec objectif pratique:
$$
\mathcal{L}_{total}=\mathcal{L}_{InfoNCE}+\beta\|M\|_1
$$

et, selon les variantes, regularisation GIB variationnelle (KL Bernoulli) dans [rca_pipeline/losses.py](../rca_pipeline/losses.py).

### 2.4 Metriques de decision

Le probleme est traite en ranking multiclasses (service cause racine):
- Avg@1, Avg@3, Avg@5,
- MRR,
- NDCG@3, NDCG@5.

Ces metriques sont harmonisees via [rca_pipeline/metrics_rcaeval.py](../rca_pipeline/metrics_rcaeval.py).

---

## 3. Evolution du projet (Phase 1 -> Phase 4)

### 3.1 Phase 1 - Fondations

Objectif: poser le cadre scientifique et le pipeline minimal.

Acquis:
- formalisation du RCA en probleme inverse,
- premier pipeline graphe + temporel,
- premieres metriques Top-K/MRR.

Reference: [docs/Week1.md](Week1.md)

### 3.2 Phase 2 - Passage a l'echelle

Objectif: sortir du cas micro-topologie et tester la scalabilite.

Acquis:
- introduction de STA (Select-Then-Analyze),
- generateur Barabasi-Albert scalable,
- hygiene experimentale (split, normalisation, multi-seeds),
- vague d'ablations structurelles.

Reference: [docs/Week2.md](Week2.md)

### 3.3 Phase 3 - Validation reelle et multimodale

Objectif: confronter les hypotheses au reel.

Acquis:
- benchmark reel unifie (RCAEval + ObservIT),
- extension metrics/logs/traces,
- analyses simule vs reel et multimodal.

References:
- [docs/Week3.md](Week3.md)
- [docs/ANALYSIS_SIM_VS_REAL.md](ANALYSIS_SIM_VS_REAL.md)
- [docs/ANALYSIS_MULTIMODAL.md](ANALYSIS_MULTIMODAL.md)

### 3.4 Phase 4 - Hardening pre-soumission

Objectif: passer en mode soutenance, avec priorite a la credibilite.

Acquis:
- guardrails critiques (warmup reel, statut MM dataset-level),
- export coherence per-root-cause vs global,
- split_mode explicite simule/reel,
- manifeste canonique + matrice de comparabilite.

References:
- [docs/Week4.md](Week4.md)
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)

---

## 4. Architecture finale (Late BN, STA, MM)

### 4.1 Vue d'ensemble

La plateforme finale est organisee en couches:
- Donnees: collecte, fenetrage, alignement labels,
- Modeles: Late Bottleneck, STA, variantes GAT/MM,
- Entrainement/Evaluation: scripts benchmark simule/reel/ablation,
- Gouvernance: validation protocol, manifest, comparability matrix.

### 4.2 Couche donnees

Composants principaux:
- [rca_pipeline/dataset_builder.py](../rca_pipeline/dataset_builder.py)
- [rca_pipeline/graph_builder.py](../rca_pipeline/graph_builder.py)
- [rca_pipeline/loaders/rcaeval_loader.py](../rca_pipeline/loaders/rcaeval_loader.py)
- [rca_pipeline/loaders/observit_loader.py](../rca_pipeline/loaders/observit_loader.py)
- [rca_pipeline/simulated_data_scalable.py](../rca_pipeline/simulated_data_scalable.py)

### 4.3 Modele Late Bottleneck (Late BN)

Principe:
- encoder le graphe complet,
- appliquer ensuite le goulot d'information (GIB) pour filtrer.

Implementations:
- [rca_pipeline/model.py](../rca_pipeline/model.py) (Late STGCN+GIB)
- [rca_pipeline/model_late_gat.py](../rca_pipeline/model_late_gat.py) (Late GATv2)

Atout: baseline forte sur certains corpus.

Limite: sensibilite possible a la regularisation sur grands graphes selon contexte de donnees.

### 4.4 Modele STA (Select-Then-Analyze)

Principe:
- selection differenciable de sous-graphe causal,
- analyse approfondie uniquement sur ce sous-ensemble.

Implementation:
- [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py)
- [rca_pipeline/train_sta.py](../rca_pipeline/train_sta.py)

Interet: efficacite structurelle et meilleure focalisation causale.

### 4.5 Extensions multimodales (MM)

Principe:
- fusion metrics/logs/traces,
- evaluation stricte possible via mode mm_strict.

Implementations:
- [rca_pipeline/model_sta_mm.py](../rca_pipeline/model_sta_mm.py)
- [rca_pipeline/model_late_gat_mm.py](../rca_pipeline/model_late_gat_mm.py)

Point critique: la comparabilite MM doit rester dataset-level explicite (exclusions possibles).

### 4.6 Orchestration entrainement/evaluation

Scripts clefs:
- [rca_pipeline/benchmark.py](../rca_pipeline/benchmark.py)
- [rca_pipeline/benchmark_real.py](../rca_pipeline/benchmark_real.py)
- [rca_pipeline/benchmark_ablation.py](../rca_pipeline/benchmark_ablation.py)
- [rca_pipeline/validation_protocol.py](../rca_pipeline/validation_protocol.py)

---

## 5. Protocole experimental (simule, RCAEval reel, observit)

### 5.1 Volet simule

But: evaluer la robustesse a l'echelle sur graphes BA.

Artefact principal:
- [rca_pipeline/benchmark_full_prepublish.json](../rca_pipeline/benchmark_full_prepublish.json)

Proprietes:
- evaluation multi-modeles,
- split trace,
- reporting harmonise (Avg@k, MRR, NDCG).

### 5.2 Volet reel RCAEval (metrics-only)

But: mesurer la generalisation sur datasets de reference.

Artefact principal:
- [rca_pipeline/benchmark_real_long_prepublish.json](../rca_pipeline/benchmark_real_long_prepublish.json)

Config dominante:
- seeds: 42,123,7
- epochs: 100
- max_cases: 90
- split_mode: varying_split_by_run_seed
- warmup STA reel: 0 (guardrail)

### 5.3 Volet reel RCAEval multimodal strict

But: evaluer en mode strict de couverture multimodale.

Artefact principal:
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

Regle: ne pas agreger globalement un dataset marque skipped_with_reason.

### 5.4 Volet ObservIT

But: boucler la chaine sur des campagnes cluster reelles controlees.

Composants:
- collecte/campagne: [rca_pipeline/dataset_builder.py](../rca_pipeline/dataset_builder.py)
- scripts: [scripts/run_observit_dataset.sh](../scripts/run_observit_dataset.sh), [scripts/run_observit_benchmark.sh](../scripts/run_observit_benchmark.sh)
- validation campagne: [rca_pipeline/validate_observit_campaign.py](../rca_pipeline/validate_observit_campaign.py)

### 5.5 Qualite statistique et limites de protocole

Les IC95 seed-level sont exportes, mais restent peu robustes quand $n=3$ (flag seed_level_bootstrap_reliable=false). Cette limite est explicitement conservee dans les artefacts et reprise dans l'analyse.

---

## 6. Resultats consolides + limites explicites

### 6.1 Benchmark simule (Top1)

Source canonique:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)

| N | STA | Late BN (STGCN+GIB) | Late BN (GATv2) | Full-Graph GCN |
|---|---:|---:|---:|---:|
| 20 | 0.7691 | 0.8145 | 0.8223 | 0.8379 |
| 50 | 0.6278 | 0.4561 | 0.5752 | 0.5263 |
| 100 | 0.5757 | 0.4208 | 0.5012 | 0.4799 |

Lecture:
- pas de vainqueur universel a toutes les echelles,
- Full-Graph GCN est meilleur a N=20,
- STA est meilleur a N=50 et N=100 dans ce snapshot canonique.

### 6.2 Benchmark reel metrics-only (RCAEval long)

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

### 6.3 Benchmark reel MM strict

Source:
- [rca_pipeline/benchmark_real_long_mm_strict_prepublish.json](../rca_pipeline/benchmark_real_long_mm_strict_prepublish.json)

| Dataset | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9167 | 0.9472 | 0.9167 | 0.9500 |
| online-boutique | 0.6167 | 0.7792 | 0.8000 | 0.8839 |

Cas sockshop:
- mm_coverage traces incomplet,
- statut dataset-level: skipped_with_reason dans la matrice de comparabilite,
- exclusion necessaire de toute comparaison isoperimetre MM globale.

### 6.4 Test critique warmup reel (A/B)

Sources:
- [results_transmission/warmup_real_trainticket_w0.json](../results_transmission/warmup_real_trainticket_w0.json)
- [results_transmission/warmup_real_trainticket_w10.json](../results_transmission/warmup_real_trainticket_w10.json)

Resultat:
- warmup=0 -> STA Avg@1 = 0.7333
- warmup=10 -> STA Avg@1 = 0.0000

Conclusion:
- warmup contrastif reel instable dans cette configuration,
- guardrail warmup reel desactive par defaut justifie.

### 6.5 Ablations (snapshot prepublish)

Source:
- [rca_pipeline/ablation_prepublish.json](../rca_pipeline/ablation_prepublish.json)

Points saillants:
- A3 spatial-only est nettement en retrait a N=100 (Top1 ~0.4775),
- A4 temporal-only est tres eleve sur ce snapshot a N=100 (Top1 ~0.9563),
- sensibilite a k (STA, N=100): Top1 passe de ~0.9243 (k=10) a ~0.9492 (k=30),
- protocole D (warmup/k-annealing) montre une forte dependance au contexte: sur ce snapshot, D2/D3 depassent D1.

Interpretation:
- les conclusions d'architecture doivent etre formulees comme des tendances conditionnelles au corpus, pas comme des lois universelles.

### 6.6 Limites explicites

Menaces internes:
1. puissance statistique limitee (souvent n=3),
2. forte sensibilite a certains hyperparametres/protocoles,
3. derive encore peu informative sur corpus long (drop ratio proche de zero).

Menaces externes:
1. comparabilite baseline publiee seulement partielle,
2. MM strict non isoperimetre sur tous les datasets,
3. extension des metriques failure-type reel encore incomplete.

---

## 7. Analyse transversale et discussion

### 7.1 Lecture scientifique: ce que disent vraiment les resultats

Les resultats consolides montrent une realite importante: il n'existe pas de modele universellement dominant.

- Sur simule, l'ordre des modeles varie avec l'echelle du graphe (N=20 vs N=50/100).
- Sur reel metrics-only, STA est devant sur 2 datasets (trainticket, sockshop), mais GATv2 reste devant sur online-boutique.
- Sur MM strict, les performances restent competitives mais la couverture multimodale inegale (ex. sockshop) interdit les conclusions globales isoperimetres.

Interpretation: la bonne question n'est pas "quel est le meilleur modele absolu", mais "quel modele est le plus robuste pour un contexte de donnees donne".

### 7.2 Analyse de transfert simule -> reel

Le stage confirme un point classique en ML systeme: les gains observes en simule ne se transferent pas mecaniquement au reel.

Facteurs explicatifs identifies dans le projet:
1. topologies et regimes de bruit differents entre simulation BA et traces d'exploitation,
2. qualite/coverage heterogene des modalites (surtout traces) selon dataset,
3. sensibilite des protocoles d'entrainement (warmup, k-selection, split mode) au corpus.

Consequence methodologique: les claims de performance doivent etre presentes par scenario de donnees, et non comme un classement global unique.

### 7.3 Analyse de robustesse et sensibilite protocolaire

Trois points de robustesse emergent:
1. warmup reel STA: le test A/B (w0 vs w10) montre une degradation drastique avec warmup actif, ce qui justifie le guardrail warmup=0 en reel;
2. sensibilite a k (STA): un k trop agressif degrade la performance, ce qui impose un tuning explicite par echelle/dataset;
3. precision statistique: avec n=3 seeds, les IC95 seed-level existent mais ne suffisent pas pour des claims externes forts.

Lecture operationnelle: le systeme est pret pour des decisions techniques de soutenance, mais la robustesse publication-grade exige une campagne multi-seeds plus large.

### 7.4 Analyse des risques de decision et mitigations

Risques principaux:
1. inflation de claim: sur-interpreter un resultat local comme conclusion generale;
2. erreur de triage operationnel: promouvoir un modele unique sur tous les corpus;
3. confusion de comparabilite: melanger des perimetres non isoperimetres (notamment MM strict).

Mitigations deja implementees:
1. manifest canonique signe et checksums pour figer les chiffres;
2. comparability matrix pour qualifier explicitement ce qui est comparable ou non;
3. export per-root-cause/global pour detecter les incoherences de synthese.

### 7.5 Cadre de decision recommande pour l'exploitation

| Contexte | Constat empirique | Decision recommandee |
|---|---|---|
| RCAEval metrics-only | STA gagne sur 2/3 datasets | STA par defaut, GATv2 en challenger actif |
| Dataset fortement heterogene (online-boutique) | GATv2 devant en metrics-only et MM | Conserver un routage par dataset/corpus |
| MM coverage partielle | statut skipped_with_reason sur certains jeux | Interdire l'agregation globale MM et fallback metrics-only |
| Training reel sensible | warmup reel peut faire chuter les scores | Garder warmup=0 tant qu'aucune preuve robuste contraire |

Ce cadre transforme le benchmark en regles de pilotage concretes pour un usage SRE/MLOps.

---

## 8. Gouvernance, reproductibilite, comparabilite

### 8.1 Source de verite et integrite

Le manifeste canonique verrouille:
- la liste d'artefacts,
- les commandes de generation,
- les hash SHA256,
- le commit de reference.

References:
- [results_transmission/canonical_results_manifest.json](../results_transmission/canonical_results_manifest.json)
- [results_transmission/canonical_results_manifest.json.sha256](../results_transmission/canonical_results_manifest.json.sha256)

### 8.2 Protocole de rerun

Le paquet de reproduction decrit les commandes de rerun simule/reel/MM/ablation/manifest:
- [results_transmission/README_resultats.txt](../results_transmission/README_resultats.txt)

### 8.3 Comparabilite externe

La matrice dediee qualifie explicitement les claims:
- metrics-only vs baselines publiees: partially_comparable,
- MM strict vs baselines publiees: not_isoperimeter.

Reference:
- [results_transmission/comparability_matrix.json](../results_transmission/comparability_matrix.json)

### 8.4 Validation logicielle et garde-fous

Elements de credibilite logicielle:
- tests pre-soumission passes: 35,
- guardrails critiques implantes (warmup reel, statut MM dataset-level, coherence per-root/global),
- audit final formalise.

Reference:
- [results_transmission/AUDIT_FINAL_SOUTENANCE_2026-04-01.md](../results_transmission/AUDIT_FINAL_SOUTENANCE_2026-04-01.md)

---

## 9. Conclusion et perspectives

### 9.1 Conclusion generale

Le projet est passe d'une logique "modele de recherche" a une logique "plateforme RCA defendable".

Bilan en trois axes:
- Software: pipeline explicite, teste, avec garde-fous,
- ML: evaluation simule/reel/MM structuree et interpretable,
- MLOps: gouvernance des resultats par manifeste canonique et matrice de comparabilite.

Conclusion operationnelle:
- etat adequat pour soutenance technique,
- claims externes a maintenir qualifies,
- trajectoire d'amelioration deja priorisee.

Horizon 30 jours:
1. preflight unique (tests + artefacts + checks coherence),
2. stabilisation des tableaux claims depuis l'artefact canonique,
3. fermeture des derniers ecarts per-root-cause reel.

Horizon 60 jours:
1. passage en preset 10 seeds sur runs externes prioritaires,
2. renforcement de la comparaison protocolaire baseline par baseline,
3. campagne drift-stress dediee.

Horizon 90 jours:
1. extension metriques failure-type en reel,
2. consolidation MM stricte dataset-level avec criteres explicites,
3. paquet de publication externe (papier + artefacts + script public de rerun).

---

## Annexes - Cartographie des composants principaux

- Entrainement/eval:
  - [rca_pipeline/train.py](../rca_pipeline/train.py)
  - [rca_pipeline/train_sta.py](../rca_pipeline/train_sta.py)
  - [rca_pipeline/evaluate.py](../rca_pipeline/evaluate.py)
- Modeles:
  - [rca_pipeline/model.py](../rca_pipeline/model.py)
  - [rca_pipeline/model_sta.py](../rca_pipeline/model_sta.py)
  - [rca_pipeline/model_late_gat.py](../rca_pipeline/model_late_gat.py)
  - [rca_pipeline/model_sta_mm.py](../rca_pipeline/model_sta_mm.py)
  - [rca_pipeline/model_late_gat_mm.py](../rca_pipeline/model_late_gat_mm.py)
- Gouvernance:
  - [rca_pipeline/results_manifest.py](../rca_pipeline/results_manifest.py)
  - [rca_pipeline/comparability_matrix.py](../rca_pipeline/comparability_matrix.py)
  - [rca_pipeline/validation_protocol.py](../rca_pipeline/validation_protocol.py)
