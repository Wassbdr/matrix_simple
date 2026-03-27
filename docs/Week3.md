# Compte rendu — Travaux sur l’architecture STA (sélection puis analyse), validation et pipelines

Ce document résume **ce qui a été fait** : évolutions du modèle, de l’entraînement, des benchmarks et des conclusions tirées des expériences. Il se lit comme un **compte rendu d’activité**, sans juxtaposition à un autre rapport ou à un état de dépôt particulier.

---

## 1. Objectifs poursuivis

- **Aligner l’implémentation** avec une formalisation de type « problème inverse » : signal exogène creux, sélection de nœuds **avant** analyse profonde, limitation du mélange de signaux (over-smoothing) sur tout le graphe.
- **Renforcer le protocole d’entraînement** : warmup contrastif cohérent, régularisation du masque de sélection, réutilisation d’utilitaires communs (early stopping, planificateurs).
- **Mesurer** les effets sur données **simulées** (Barabási–Albert) et **réelles** (RCAEval), y compris en **multimodal** (métriques, logs, traces).
- **Isoler par ablations** le rôle de la régularisation (L1 vs GIB variationnelle) et du mode de pooling pour InfoNCE (moyenne sur l’encodeur léger vs pondération par le masque).
- **Se rapprocher des pratiques de l’état de l’art** : métriques de référence (RCAEval), modèles de comparaison (GATv2, graphe causal), adaptation au déploiement (dérive, multimodal).

---

## 2. Travaux réalisés (code et artefacts) — inventaire détaillé

### 2.1 Cœur STA (`model_sta.py`)

- **Encodeur léger « zero-hop »** : plus de convolution de graphe **avant** la sélection ; traitement **local temporel** (Conv1d, attention) puis projection — l’adjacence n’entre qu’**après** le masque, sur le sous-graphe. Répond à la critique « **over-smoothing** avant localisation de la cause » souvent invoquée dans les travaux sur GNN pour RCA.
- **Sélection** : `STESelector` (Gumbel, top-k, STE) ; variante **`HierarchicalSelector`** (raffinement en deux étapes) exposée via **`selector_type`** — rapproche les idées de **sélection hiérarchique / raffinage** du support.
- **Chemin profond** : blocs temporel + GCN sur le sous-graphe uniquement ; encodeur temporel profond en **`conv`** ou **`s4`** (bloc type **SSM** / state-space model), aligné avec la littérature sur le **modèle d’état** pour séries temporelles longues.
- **Fusion selector / scorer** : paramètre **`residual_alpha`** (mélange sur le top-k pour que le scorer affine sans contredire totalement le sélecteur) — évite que la tête profonde **détruise** la décision du sélecteur (cohérent avec les architectures « two-stage » en détection).
- **Pertes côté pipeline** : usage de la **GIB variationnelle** (KL Bernoulli) en substitution d’une simple L1 sur le masque, selon les scripts — plus proche d’un **bottleneck informationnel** formulé en probabilités qu’une pénalité de parcimonie ad hoc.

### 2.2 Entraînement (`train_sta.py`, `training_utils.py`)

- **Phase contrastive** : InfoNCE sur embeddings obtenus par **pondération par `M_soft`** (gradient à travers le masque) + **GIB variationnelle** ; docstring alignée sur ce protocole — standard **représentation contrastive** (InfoNCE) couplé à une contrainte sur \(I(Z;X)\).
- **Phase supervisée** : CE sur les logits du sélecteur + CE secondaire sur le scorer lorsque la cause est dans le top-k + GIB sur `M_soft`.
- **`training_utils.py`** : `EarlyStopping` (sur **MRR**, comme recommandé pour le ranking à grand \(N\)), `CosineAnnealingLR` / `CosineAnnealingWarmRestarts`, `TwoPhaseTrainer` — pratiques de **stabilisation** et **curriculum** proches des pipelines de référence en apprentissage profond.

### 2.3 Benchmark simulé (`benchmark.py`)

- Warmup STA aligné sur le même schéma que `train_sta` (GIB + pooling masqué).
- **Flags d’ablation** en ligne de commande : `--no-gib`, `--no-mask-weighted-nce`, `--selector-type`, `--temporal-encoder`.
- **`train_sta_compact`** étendu pour rejouer les variantes A1–A3 sans dupliquer le code.
- **Late Bottleneck** : `train_v1_compact` avec phase contrastive + **GIB asymétrique** adaptatif (`adaptive_gib_beta`), phase supervisée sur **logits GIB bruts** (ranking) — aligné avec les usages de **GIB** en graphe.
- **Late GATv2** : entraînement deux phases (InfoNCE puis CE), baseline **sans masque binaire** — se rapproche des méthodes type **attention sur graphe** pour RCA (voir commentaires dans `model_late_gat.py`).
- **Métriques homogènes** : `RCAEvalMetrics` sur les sorties simulées (Avg@k, MRR, NDCG) — **même formalisme** que sur le benchmark réel.
- **Régression** : références `REFERENCE_TOP1` + `_regression_flag` pour détecter les régressions entre runs — **gouvernance qualité** des expériences.
- **Graphe causal / hybride** : `CausalGraphBuilder`, `graph_type` structural | causal | hybrid, pondération d’arêtes pour Late BN / GAT.
- **Perte cascade** : `CausalCascadeAwareLoss` / `LearnedPropagationCascadeLoss` optionnelle — biais structurel **propagation** comme dans les échecs en cascade.
- **Drift** : filtre ADWIN + cooldown sur flux simulé — aligné avec les scénarios **changement de distribution** en AIOps.
- **Fusion STA** : `gated_selector_scorer_fusion` / utilitaires associés (voir `sta_fusion_utils.py`).

### 2.4 Pertes et extensions (`losses.py`)

- `variational_gib_loss` : borne KL sur **Bernoulli** pour le masque.
- **Multi-root** : `multi_root_cause_loss`, `multi_root_cause_metrics` — cas **plusieurs causes** (plus réaliste qu’un seul argmax).
- `info_nce_loss` : contraste standard.

### 2.5 Données simulées (`simulated_data_scalable.py`)

- Graphes **Barabási–Albert** à grande échelle, anomalies multi-types (**hard / gray / cascade**).
- **Logs et traces synthétiques** alignés sur les **dimensions et conventions** des extracteurs réels — permet d’entraîner **STA-MM** sans dépendre uniquement du disque RCAEval.

#### Causes racines et **nouveaux** types de défaillance — intégration et raisons

En plus d’une **cause racine unique** par fenêtre anomale (`root_cause` : nœud tiré dans le graphe), le générateur attache un **`failure_type`** à chaque échantillon PyG : `healthy` | `gray` | `hard` | `cascade`. Ce n’est pas une multiplication des « classes de cause » (qui restent les **N** services) : c’est une **taxonomie de la forme du signal** d’anomalie, calquée sur la littérature ops / AIOps.

| Type | Rôle dans le simulateur | Pourquoi l’avoir ajouté |
|------|-------------------------|-------------------------|
| **hard** | Signal **fort**, décroissance rapide du transitoire sur la cause racine ; symptômes type panne « dure » (pics nets). | Modélise les pannes **visibles** (erreurs 5xx, saturation) — cas le plus simple pour un classifieur. |
| **gray** | Signal **plus faible**, décroissance **lente** ; bruit modéré sur métriques — « collective anomaly » côté métriques. | Reflète les **gray failures** (souvent majoritaires en prod : latence, micro-rallentissements sous SLA). Sans ce mode, le modèle serait **biaisé** vers des pics faciles. |
| **cascade** | **Propagation** le long du graphe (jusqu’à `max_cascade_hops` hops) : nœuds **downstream** reçoivent une partie du signal ; liste `cascade_nodes` pour logs/traces. | Isole le cas **cause → victimes** (microservices) : la cause racine n’est pas le seul nœud « bruyant » — indispensable pour tester **localisation** vs simple corrélation spatiale. |
| *(reste)* | Si les ratios `gray_failure_ratio` + `hard_failure_ratio` ne couvrent pas 100 % du tirage, le reste est **cascade** (par construction du `rng`). | Répartition contrôlée (défaut typique : 15 % gray, 35 % hard, **50 % cascade**). |

**Intégration dans le pipeline :**

1. **Champs sur `torch_geometric.data.Data`** : `root_cause`, `failure_type`, et pour cascade les indices utiles à la génération logs/traces — le **benchmark** et **`evaluate.py`** peuvent **filtrer** ou **agréger** les métriques par type.
2. **`failure_type_utils.split_logits_by_failure_type`** : découpe `y_true` / logits en paquets `hard`, `gray`, `cascade`, `other` — pour répondre à la question : *le modèle échoue-t-il surtout sur les grays ?*
3. **`cascade_aware_loss.py`** : encode la **priorité** des scores le long des arêtes (cause au-dessus des descendants) — cohérent avec la sémantique **cascade**.
4. **`gray_failure_detector.py`** + **`sta_fusion_utils.apply_gray_prior`** : détection **heuristique** style GrayScope / corrélations (voir références en tête de fichier) ; le prior **rehausse** les logits des nœuds détectés « gray » — **hybridation** règles + réseau.
5. **`anomaly_taxonomy.py`** : taxonomies **Chandola**, **Zamanzadeh**, **CRI** — utilisées pour enrichir les **labels** ou les **features** orientées détection de gray (documentation et ingénierie des signaux, pas seulement un seul score global).

**Causes multi-racines (plusieurs nœuds)** : côté **pertes**, `losses.py` expose des utilitaires **multi-root** (BCE multi-label, métriques ensemblistes) pour les scénarios où \(\|U\|_0 > 1\) — le générateur scalable reste surtout **une** cause par fenêtre ; l’extension multi-label prépare **l’alignement** avec des jeux réels plus riches.

### 2.6 Benchmark réel (`benchmark_real.py`, chargeurs, données RCAEval)

- Évaluation sur jeux **RCAEval** : STA vs GATv2, métriques **Avg@1/3/5, MRR, NDCG@3/5** — **même famille** que les articles qui référencent RCAEval / BARO.
- **Comparaison** avec baselines publiées dans `metrics_rcaeval` / `compare_with_rcaeval_baselines` (BARO, MicroRCA, etc. — **ancrage SOTA** explicite).
- **Split train/test** stratifié avec **graine** par exécution pour des comparaisons reproductibles entre seeds.
- **Warmup (réel)** : le warmup contrastif STA est **désactivé par défaut** sur RCAEval réel (`--sta-contrastive-warmup 0`), car il induisait un plateau de performance ; le warmup reste pertinent côté **simulé**.
- Runs **metrics-only** et **multimodaux** (`SelectThenAnalyze_MM`, GATv2-MM), avec `--mm-strict`, `--mm-trace-prior`, etc.
- **Détection de dérive** (optionnelle) : flux type **streaming** avec filtre avant entraînement.

#### 2.6.1 Évaluation sur cluster (observ-it) — boucle *collecte → dataset → benchmark*

En plus de RCAEval, le projet inclut un flux “**cluster réel contrôlé**” (observ-it) : on contrôle l’injection (Chaos Mesh), la charge (Locust) et l’étiquetage (cause racine), puis on applique le **même** benchmark `benchmark_real.py`.

- **Orchestration & collecte** : `rca_pipeline/dataset_builder.py`
  - Lance **Locust** (traffic), attend une période **baseline**, applique un YAML **Chaos Mesh**, attend la période **anomaly**, supprime le chaos.
  - Interroge Prometheus en `query_range` et exporte un CSV `dataset_<run_id>.csv`.
  - (Robustesse) si l’URL Prometheus est `127.0.0.1:9090` et ne répond pas, le script tente un `kubectl port-forward` vers `svc/rca-prometheus-server`.

- **Benchmark** : `rca_pipeline/benchmark_real.py --observit-csv <csv_ou_dossier>`
  - Supporte un **fichier** CSV unique ou un **répertoire** contenant plusieurs `dataset_*.csv` (fusion en `_merged_observit.csv`).
  - Construit les fenêtres temporelles et évalue **STA** / **GATv2** avec les mêmes métriques RCAEval (Avg@k, MRR, NDCG).
  - Normalisation : `StandardScaler` **fit sur train uniquement**, puis appliqué au test (alignement pratique “pas de fuite”).

Procédure pas-à-pas et scripts : `docs/observit_dataset.md` + `scripts/run_observit_dataset.sh` / `scripts/run_observit_benchmark.sh`.

#### 2.6.2 Dataset réel observ-it — composition et schéma

Le dataset observ-it est un export tabulaire pivoté au format “timeseries par pod” avant conversion en fenêtres PyG.

- **Granularité** :
  - une ligne = \((timestamp, pod)\) avec un vecteur de métriques;
  - les fenêtres sont ensuite construites sur l’axe temps, puis empilées en tenseur `.x` **[N, F, W]**.

- **Services / nœuds** : graphe fixe 5 services (noms typiques) : `frontend`, `auth`, `backend`, `cache`, `database`.  
  La cause racine est alignée sur cet espace (flag `--root-cause` au moment de l’injection).

- **Colonnes minimales (schéma)** :
  - `timestamp`, `pod`,
  - `cpu_usage`, `ram_usage`, `network_tx`, `network_rx`, `error_rate`,
  - `label` (0/1),
  - `root_cause` (string si `label==1`, vide sinon).
  - Colonnes supportées en plus : `latency_p99` (utile pour gray / tail latency si disponible).

- **Sémantique des labels** :
  - `label=1` uniquement **pendant la fenêtre de chaos** (anomaly period),
  - `root_cause` défini pendant l’anomalie, vide sur baseline,
  - champs additionnels possibles : `failure_type` (`hard|gray|healthy`) et `scenario` (nom de campagne).

- **Conversion en dataset PyG** :
  - `rca_pipeline/graph_builder.py` construit des snapshots puis des fenêtres glissantes,
  - `rca_pipeline/loaders/observit_loader.py` filtre les fenêtres sans `root_cause` (baseline-only) et renvoie `(List[Data], node_names, edge_index)` compatible RCAEval.

- **Contrôles qualité (campagne)** :
  - `rca_pipeline/validate_observit_campaign.py` vérifie:
    - présence du schéma minimal,
    - volume minimal de lignes labelisées (>=200),
    - diversité des causes (>=3) et des types (>=2 si `failure_type` présent).

### 2.7 Multimodal (`model_sta_mm.py`, `model_late_gat_mm.py`)

- Encodeurs **séparés** par modalité (métriques temporelles, logs, traces).
- **Sélecteurs par modalité** + **fusion cross-modale** (attention) — formalisation **multi-vues** du problème inverse (consensus sur `supp(U)`).
- GATv2-MM : même idée que la ligne **GATv2** mais avec **têtes multimodales**.

### 2.8 Graphe causal et structure (`causal_graph_builder.py`)

- Méthodes **PC, FCI, Granger**, et piste **NOTEARS** (optimisation différentiable sous contrainte d’acyclicité) — rapprochement des travaux **structure learning** pour dépendances entre services.

### 2.9 Pertes cascade (`cascade_aware_loss.py`)

- Pénalisation des ordres de score **incohérents** avec la propagation sur le graphe ; variante avec **matrice B apprise** — lien avec les modèles **propagation** en RCA.

### 2.10 Détection et adaptation (`drift_detector.py`, `gray_failure_detector.py`, `sta_fusion_utils.py`)

- **ADWIN** pour dérive ; **TestTimeAdaptation** (recalibration BN / petit adaptateur) — pratiques **test-time adaptation** et **concept drift** en production.
- **Prior gray** : `apply_gray_prior` pour injecter une **heuristique** sur les nœuds « gray failure » dans les logits du sélecteur — pont entre **règles** et **réseau**.

### 2.11 Évaluation et protocole (`evaluate.py`, `validation_protocol.py`, `failure_type_utils.py`)

- Métriques **RCAEval** + découpage par **type de panne** (hard / gray / cascade) — évaluation **fine** comme dans les benchmarks avancés.
- `validation_protocol.py` : cadre pour **répéter** expériences avec critères de succès.

### 2.12 Métriques et sélection de k (`metrics_rcaeval.py`, `k_selection.py`)

- Implémentation **Avg@k, MRR, NDCG** alignée sur RCAEval.
- **`get_optimal_k(N)`** : règle unique partagée benchmark / entraînement ; **`adaptive_gib_beta`** pour éviter une **sur-sparsification** à grand \(N\) — **scalabilité** de la régularisation.

### 2.13 Features et intégration (`log_feature_extractor.py`, `trace_feature_extractor.py`, `prometheus_connector.py`)

- Extracteurs **tabulaires** pour logs/traces — **réutilisables** entre synthétique et réel.
- **Prometheus** : brique pour **données réelles** en cluster.

### 2.14 Tests (`rca_pipeline/tests/`)

- Tests unitaires sur `k_selection`, `failure_type_utils`, `rcaeval_loader`, `sta_fusion_utils` — **réduction des régressions** lors des évolutions.

### 2.15 Résultats numériques enregistrés

- Fichiers JSON pour benchmarks simulés et réels, ablations **A1–A3**, et synthèse dans **`docs/validation_comparison_v1_v2.md`** (les libellés « v1 / v2 » y désignent **deux jeux de runs enregistrés**, pas une période de travail).

---

## 3. Rapprochement du SOTA : en quoi les changements y contribuent

Le « SOTA » en RCA opérationnelle combine souvent : **(i)** un **benchmark reconnu** (ex. RCAEval), **(ii)** des **métriques de ranking** comparables aux articles (Avg@k, MRR), **(iii)** des **architectures** qui exploitent **graphe + temps + multimodalité**, **(iv)** des **biais** (causalité, cascade), **(v)** **robustesse** (dérive). Le tableau ci-dessous relie **les changements** à ces **dimensions**.

| Dimension SOTA typique | Où c’est adressé dans le dépôt | Effet recherché |
|------------------------|--------------------------------|-----------------|
| **Benchmark standardisé** | `benchmark_real.py`, `loaders/rcaeval_loader.py`, jeux RE1/RE2/RE3 | Même **protocole d’évaluation** que les références publiées (BARO, etc.). |
| **Métriques de ranking** | `metrics_rcaeval.py`, usage dans `benchmark.py` / `evaluate.py` | **Avg@k, MRR, NDCG** — **comparabilité** avec la littérature RCAEval. |
| **GNN + attention** | `model_late_gat.py` (GATv2), commentaires DynaCausal / soft importance | Baseline **forte** sur graphe complet, proche des approches **attention** pour le RCA. |
| **Information bottleneck / parcimonie** | GIB `model.py`, GIB variationnelle `losses.py` + STA | **Sélection informative** de nœuds (Late BN en fin de pipeline ; STA en début). |
| **Auto-supervision + supervisé** | InfoNCE + phases 1/2 dans `train_sta`, `benchmark`, `train_v1_compact` | Aligné sur les pipelines **pré-entraînement contrastif** puis **fine-tuning**. |
| **Causal structure** | `causal_graph_builder.py` (PC, Granger, NOTEARS) | **Graphe causal** ou hybride au lieu du seul graphe de déploiement. |
| **Cascade failures** | `cascade_aware_loss.py`, types dans `simulated_data_scalable.py` | Modélisation **propagation** (microservices). |
| **Multimodal** | `model_sta_mm.py`, extracteurs, `benchmark_real.py --use-mm` | Tendance **métriques + logs + traces** des plateformes AIOps. |
| **Concept drift / adaptation** | `drift_detector.py` (ADWIN, TTA) | **Robustesse** aux changements de distribution (sujet SOTA en prod). |
| **Scalabilité** | STA O(N + k²), sélection précoce | Argument **production** pour grands \(N\) (vs O(N²) sur tout le graphe). |
| **Gray failures** | `gray_failure_detector.py`, `sta_fusion_utils.apply_gray_prior` | Pont **heuristique + apprentissage** (cas difficiles en RCA). |

### Où le rapprochement SOTA est **réellement observé** (dans les runs enregistrés)

- **Données simulées (BA)** : la configuration STA actuelle (warmup + GIB + sélection) obtient des **Top-1 et MRR** nettement **plus élevés** que les runs archivés dans `benchmark_full.json` aux mêmes N et seeds — **qualité** du ranking sur un **protocole contrôlé** et **comparable** (régression, multi-seed, métriques RCAEval homogènes).
- **Données réelles (RCAEval)** : après correctif (warmup STA désactivé par défaut sur réel), **STA** remonte fortement (campagne longue documentée) et devient **compétitif** ; **GATv2** reste fort et sert de borne supérieure dans le pipeline. Le rapprochement SOTA se fait **à la fois** par méthodologie (RCAEval + métriques standard) **et** par amélioration de score STA sur réel.

### Synthèse honnête

- **Méthodologiquement** : le travail **se rapproche du SOTA** par **alignement sur RCAEval**, **métriques standard**, **baselines fortes** (GATv2), **multimodal** et **biais structurels** (causal, cascade, drift).
- **Empiriquement sur simulé** : **STA** montre des **gains** importants sur les runs archivés — **soutient** la thèse « sélection précoce + GIB + contraste ».
- **Empiriquement sur réel** : l’écart observé initialement sur STA provenait en grande partie d’un **protocole warmup inadapté** ; après correctif, il reste surtout du **last-mile** (stabilité multi-seed, cas `online-boutique`, périmètre MM strict).

### Traçabilité `docs/sota_anomalies.pdf` (§ → fichier → preuve / gap)

Document interne d’état de l’art (taxonomie cause/signal, datasets, gray, causalité). Tableau de correspondance **court** — les « preuves » sont des **implémentations** ou **mesures** dans ce dépôt, pas une revue exhaustive.

| § PDF (thème) | Fichier(s) principal(aux) | Preuve / gap |
|---------------|-----------------------------|--------------|
| §1 Taxonomie **axe cause** (hard, gray, cascade, contention, metastable…) | `simulated_data_scalable.py`, `failure_type_utils.py`, `cascade_aware_loss.py`, `gray_failure_detector.py` | **Preuve** : types hard / gray / cascade en simulation, métriques par type, pertes cascade, heuristiques P99/P50, décorrélation, tendance (metastable), Limplock. **Gap** : pas de modèle explicite **Byzantine** ni taxonomie anti-patterns (Cerny). |
| §2 **Axe signal** (Chandola, Zamanzadeh MTS, CRI) | `anomaly_taxonomy.py` | **Preuve** : classifieurs heuristiques alignés sur les définitions citées. **Gap** : pas d’évaluation sur benchmarks AD « classiques » (hors RCA) ; implémentations simplifiées vs articles sources. |
| §2.4 **Drift** vs anomalie (ADWIN amont) | `drift_detector.py` | **Preuve** : ADWIN, TTA. **Gap** : impact chiffré sur RCAEval à consigner par run. |
| §2.5 Anomalies de **graphe** (topologie) | `model_sta.py`, `model_late_gat*.py`, `benchmark_real.py` | **Preuve** : RCA sur graphe de services. **Gap** : pas de formalisation des typologies type défi ISWC (7 fautes). |
| §3 **Datasets** de référence | `loaders/`, `benchmark_real.py`, JSON réels | **Preuve** : **RCAEval** + testbeds RE (TrainTicket, SockShop, OnlineBoutique). **Gap** : jeux industriels massifs (Alibaba, Azure VMNoise, etc.) non intégrés. |
| §5 **Gray failures** & micro-stutters (GrayScope, Limplock) | `gray_failure_detector.py`, `sta_fusion_utils.py` | **Preuve** : mécanismes et références alignés sur le PDF ; prior sur logits. **Gap** : inspiration, pas reproduction **GrayScope** / **GREYBOUND**. |
| §6 **Inférence causale** (PC, FCI, Granger, NOTEARS) | `causal_graph_builder.py` | **Preuve** : construction graphe causal / hybride. **Gap** : pas d’intégration des méthodes nommées **DynaCausal**, **Chain-of-Event**, **RCACopilot**, **T2C**, etc. |
| §7 **Limites** & gap benchmark vs prod | `docs/validation_comparison_v1_v2.md` | **Preuve** : plateau STA réel (**~0,20**) observé initialement, puis **levé** après correctif (warmup=0 par défaut sur réel) avec des scores élevés sur la campagne longue. **Gap** : conserver la robustesse statistique (IC, plus de seeds) et fermer les cas limites (MM strict sockshop, tuning online-boutique). |

---

## 4. Modèles et rôles dans les pipelines actuels

| Composant | Fichier(s) | Rôle |
|-----------|------------|------|
| **STA** | `model_sta.py` | Zero-hop → sélection (STE ou hiérarchique) → sous-graphe → STGCN réduit → scorer + mélange α. |
| **STA-MM** | `model_sta_mm.py` | Métriques + logs + traces ; votes par modalité ; fusion ; puis sélection et analyse profonde. |
| **Late Bottleneck STGCN+GIB** | `model.py` / benchmark | Traitement graphe complet puis masque GIB. |
| **Late GATv2** | `model_late_gat.py`, `model_late_gat_mm.py` | Attention sur graphe complet ; baseline très forte sur RCAEval dans les scripts actuels. |
| **Baselines** | `benchmark.py` | GCN plein graphe, MLP, forêt aléatoire, etc. |

**Paramètres** : de l’ordre de **15,5k** paramètres pour STA dans les configurations récentes. **Complexité** ciblée : **O(N + k²)** pour la partie sélection + sous-graphe, à comparer aux approches **O(N²)** sur tout le graphe.

---

## 5. Résultats et conclusions (synthèse)

### 5.1 Données simulées (Barabási–Albert)

- Les runs consignés dans **`benchmark_v2_full.json`** montrent une **forte amélioration** du Top-1 et du MRR par rapport aux runs enregistrés dans **`benchmark_full.json`** (mêmes échelles N, mêmes seeds), avec une **latence** STA du même ordre ou inférieure selon les mesures.
- Les **objectifs de validation** fixés pour le protocole simulé (Top-1 suffisant à N=20 et à grand N) sont **atteints** sur ces expériences.

### 5.2 Ablations warmup (N = 20 et 50)

- **A1** : L1 + pooling moyen sur l’encodeur léger.
- **A2** : GIB variationnelle + pooling moyen → **meilleur Top-1** que la variante masquée sur ces runs.
- **A3** : GIB + pooling **pondéré par le masque** → correspond au **comportement « complet »** documenté pour le warmup sur ces tailles.

**Conclusion :** le masque pondéré reste **cohérent** avec le gradient à travers la sélection, mais **numériquement** la configuration « moyenne + GIB » (A2) est supérieure ici — **réglage** (β, température, durée) à poursuivre plutôt qu’un rejet a priori.

### 5.3 Données réelles (RCAEval)

- **STA (avant correctif warmup)** stagnait autour de **0,20 Avg@1** dans `benchmark_real_v2_metrics.json`, tandis que **GATv2** restait très fort.
- **STA (après correctif)** : la stagnation est levée en désactivant le warmup contrastif sur réel (voir §5.4 et `docs/validation_comparison_v1_v2.md`). Sur la campagne longue (3 seeds, 90 cas, 100 epochs), STA atteint **0,97** (trainticket), **0,80** (sockshop), **0,68** (online-boutique) en Avg@1.
- **STA-MM** : l’évaluation multimodale reste sensible à la **couverture** des modalités (ex. sockshop en `mm_strict` peut être **non comparable** si `traces_ratio=0`).

**Conclusion :** l’écart STA réel observé initialement était en grande partie un **problème de protocole warmup** sur RCAEval réel. Après correctif, STA devient compétitif (souvent au-dessus des seuils BARO), et la question se déplace vers: (i) stabilité statistique multi-seed, (ii) last-mile sur online-boutique, (iii) périmètre MM strict.

### 5.4 Correctif STA réel (warmup=0) — “ce qui a été fait depuis”

**Problème** : le warmup contrastif (initialement activé par défaut dans `benchmark_real.py`) entraînait un comportement dégénéré sur RCAEval réel (plateau ~0,20 Avg@1).  
**Correctif** : ajout du flag `--sta-contrastive-warmup` et **défaut fixé à 0** pour RCAEval réel.

- **Code** : `rca_pipeline/benchmark_real.py` (nouveau flag + défaut `0`), `rca_pipeline/validation_protocol.py` (utilise `sys.executable`, passe warmup via env, presets `smoke` vs `long`).
- **Résultats** :
  - Campagne longue: `rca_pipeline/benchmark_real_sta_fix_all_nowarm.json` (STA remonte fortement; GATv2 stable en isopérimètre).
  - Last-mile online-boutique: `rca_pipeline/benchmark_real_sta_fix_online_h48_nowarm.json` (STA Avg@1 **0,783** avec `hidden_dim=48`).
- **Documentation** : `docs/validation_comparison_v1_v2.md` contient le tableau avant/après (STA et GATv2), les presets de validation, et la clôture explicite du périmètre MM sockshop en `mm_strict`.
- **Audit stage** : `docs/stage_audit_sta_latebn.md` consolide l’analyse technique STA vs Late-Bottleneck et la roadmap (diagnostic, protocole, stats, ablations, MM, industrialisation).

---

## 6. Pistes suivantes

1. **Warmup** : sensibilité à β, température, nombre d’epochs, pour rapprocher les performances A2 et A3.
2. **Réel** : analyser les splits, le prétraitement et la comparabilité avec les méthodes publiées ; enrichir ou assouplir les contraintes MM selon les jeux.
3. **Simulé** : poursuivre des échelles N plus grandes pour tester le gain **O(N + k²)** là où le full-graph devient coûteux.
4. **Documentation des expériences** : conserver les JSON et une synthèse décisionnelle à jour (`validation_comparison_v1_v2.md`) pour tracer les décisions.

---

## 7. Documents associés

- Synthèse des métriques et décisions **go / no-go** : `docs/validation_comparison_v1_v2.md`.
- État de l’art **anomalies / RCA cloud** (taxonomie, datasets, gray, causalité) : `docs/sota_anomalies.pdf` — traçabilité § → code dans la section **3** ci-dessus.
- **Dataset réel cluster observ-it** (collecte + benchmark STA/GATv2) : `docs/observit_dataset.md`.
- Audit stage (analyse longue + roadmap STA vs Late-Bottleneck) : `docs/stage_audit_sta_latebn.md`.
- Audit ou notes de projet (si présent) : `rca_pipeline/audit_report.md`.
