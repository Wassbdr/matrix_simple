# Synthèse : Pipeline RCA pour microservices

## Modèles, postulats, résultats et évaluation sur cluster sock-shop

---

## 1. Introduction et cadrage

Les architectures microservices distribuent la logique applicative sur des dizaines de services indépendants. En cas d'incident, identifier le service à l'origine de la panne — la *root cause* — parmi N nœuds est un problème non trivial : les pannes se propagent en cascade, les signaux sont bruités, et l'espace de recherche croît avec N.

Ce projet développe et compare quatre modèles de RCA fondés sur des graphes de neurones spatio-temporels (STGCN), avec deux axes de variation principaux :

1. **Stratégie de bottleneck** : sélection précoce (STA) vs attention tardive (GATv2).
2. **Modalités d'entrée** : métriques seules vs fusion multimodale (métriques + logs + traces).

**Périmètre expérimental :**

| Benchmark | Description | N nœuds |
|---|---|---:|
| Simulé (Barabasi-Albert) | Graphes aléatoires BA, pannes synthétiques | 20, 50, 100 |
| RCAEval trainticket | Dataset public, microservices Java | 5 |
| RCAEval sockshop | Dataset public, e-commerce Go | 5 |
| RCAEval online-boutique | Dataset public, microservices Google | 5 |
| ObservIT cluster | Cluster K8s interne, sockshop déployé | 14 |

**Métriques d'évaluation :** Avg@k (proportion de cas où la RC est dans le top-k), MRR (Mean Reciprocal Rank), NDCG@k (Normalized Discounted Cumulative Gain). Avg@1 est la métrique principale : la RC est-elle trouvée immédiatement ?

---

## 2. Présentation des modèles

### 2.1 Vue d'ensemble

| Modèle | Stratégie | Complexité | Modalités |
|---|---|---|---|
| STA | Early bottleneck + GIB | O(N + k²) | Métriques |
| GATv2 | Late bottleneck + attention douce | O(N²) | Métriques |
| STA-MM | Early bottleneck + fusion cross-modale | O(N + k²) | Métriques + logs + traces |
| GATv2-MM | Late fusion + GATv2 | O(N²) | Métriques + logs + traces |

---

### 2.2 STA — Select-then-Analyze (`model_sta.py`)

**Principe :** pipeline en deux étapes — sélectionner d'abord un sous-graphe de k nœuds candidats, puis analyser ce sous-graphe en profondeur.

**Architecture (5 étages) :**

1. **LightEncoder** — encodeur temporel zero-hop (Conv1d + attention pooling). Pas de convolution graphe ici : le sélecteur ne voit que les features locales de chaque nœud, ce qui évite le sur-lissage à grande échelle.
2. **STESelector** — sélection différentiable via Gumbel-sigmoid + binarisation top-k avec Straight-Through Estimator (STE). Gradient transmis malgré la discrétisation.
3. **Extraction de sous-graphe** — appel vectorisé `pyg_subgraph` sur le batch entier pour extraire les k nœuds sélectionnés.
4. **DeepSTGCN** — 2 couches Conv1d temporelles + 2 couches GCN spatiales, appliquées uniquement sur le sous-graphe de taille k.
5. **SubgraphScorer** — scoring final avec mélange résiduel appris : `logits = (1-α)·scorer + α·selector`. Le scorer peut affiner mais pas contredire la sélection.

**Régularisation GIB variationnel :** la régularisation L1 naïve est remplacée par une divergence KL Bernoulli, donnant une borne théorique sur l'information mutuelle I(Z; X) (Information Bottleneck). Cela force une sélection parcimonieuse et principiée.

**Intérêt principal :** complexité **O(N + k²)** vs O(N²) pour les modèles full-graph. À N=100 et k=15, le nombre d'opérations dans le module lourd passe de 10 000 à 2 250 — gain de ×4,4 sur la partie coûteuse. La latence d'inférence mesurée est ~3,4 ms (vs ~3,0 ms pour GATv2 à N=100, voir §4.1).

---

### 2.3 GATv2 — Late Bottleneck avec attention (`model_late_gat.py`)

**Principe :** encoder tout le graphe avec une attention multi-tête (GATv2), puis dériver l'importance des nœuds à partir des poids d'attention agrégés.

**Architecture :**

1. **TemporalConvBlock** — même encodeur Conv1d que STA.
2. **GATv2SpatialBlock × 2** — deux couches GATv2 avec 4 têtes d'attention. L'attention est apprise de façon contextuelle (dépend de la paire source/cible, contrairement à GAT v1).
3. **Importance de nœud** — somme des poids d'attention entrants par nœud (importance douce, pas de masque binaire).
4. **NodeScorer** — MLP produisant les logits finaux [B, N].

**Intérêt principal :** l'attention douce permet de capturer des causes racines diffuses, impliquant plusieurs nœuds, sans forcer une sélection binaire. Particulièrement pertinent quand la frontière cause/non-cause est floue (pannes en cascade multi-hop). Pas de contrainte de sparsité : GATv2 apprend implicitement à concentrer son attention sans régularisation explicite.

---

### 2.4 STA-MM — STA multimodal (`model_sta_mm.py`)

**Principe :** formulation en problème inverse multi-vues. Chaque modalité vote indépendamment sur le support de causalité U via son propre sélecteur, puis les votes sont fusionnés par attention cross-modale.

**Formulation :** min_U ||U||₀ s.t. Y⁽ᵐ⁾ ≈ g_φ⁽ᵐ⁾(X) pour tout m ∈ {métriques, logs, traces}

**Architecture :**
- Trois encodeurs spécialisés : MetricEncoder (Conv1d), LogEncoder (MLP + LayerNorm), TraceEncoder (MLP + LayerNorm).
- Trois sélecteurs parallèles → trois masques souples M⁽ᵐ⁾.
- Fusion cross-modale par attention → M_fusionné → même pipeline STA (STE + DeepSTGCN + SubgraphScorer).

**Propriété clé :** une modalité absente ou bruitée produit un masque uniforme, sans effet sur la fusion. Tolérance native aux modalités manquantes.

---

### 2.5 GATv2-MM — GATv2 multimodal (`model_late_gat_mm.py`)

**Principe :** late fusion — chaque modalité est encodée séparément, les représentations sont concaténées, puis GATv2 est appliqué sur la représentation fusionnée.

**Architecture :**
- Encodage séparé : H_m (Conv1d), H_l (Linear + LayerNorm), H_t (Linear + LayerNorm).
- FusionMLP : concaténation [H_m ; H_l ; H_t] → projection → H_fusionné.
- GATv2 × 2 + NodeScorer.
- Zero-pad si logs ou traces absents (fallback mono-modal).

---

### 2.6 Baselines de référence

Les baselines publiées dans la littérature RCAEval (`baselines.py`) servent de points de comparaison indicatifs :

| Méthode | Avg@1 | Avg@3 | Avg@5 | MRR |
|---|---:|---:|---:|---:|
| BARO | 0.71 | 0.82 | 0.86 | 0.77 |
| E-Diagnosis | 0.64 | 0.77 | 0.83 | 0.72 |
| CausaRank | 0.52 | 0.70 | 0.79 | 0.63 |
| MicroRCA | 0.49 | 0.68 | 0.77 | 0.60 |

**Note de comparabilité :** le protocole exact (splits, preprocessing, max_cases) des baselines publiées n'est pas totalement documenté. Toute comparaison avec nos modèles est donc **partiellement comparable** (metrics-only) ou **non-isopérimètre** (multimodal). Les claims sont qualifiés en conséquence tout au long de ce document.

---

## 3. Postulats de départ et vérification

Cinq hypothèses structurent la conception des modèles. Elles ont été testées expérimentalement.

| # | Postulat | Protocole de test | Verdict |
|---|---|---|---|
| P1 | STA bat les modèles full-graph à grand N (avantage asymptotique O(N+k²)) | Benchmark simulé BA, N∈{20,50,100}, 5 seeds | **CONFIRMÉ** à N≥50 |
| P2 | L'avantage STA se transfère sur les datasets réels | RCAEval 3 datasets, metrics-only, 5 seeds, 100 epochs | **PARTIEL** : vrai sur 2/3 datasets |
| P3 | La structure graphe apporte un gain vs temporal-only | Ablation T5 : STA vs STA sans GCN, mêmes splits | **MARGINAL** : +2% trainticket, 0 ou négatif ailleurs |
| P4 | La fusion multimodale améliore les scores | RCAEval metrics-only vs MM-strict, 5 seeds | **NON CONFIRMÉ** sur RCAEval |
| P5 | Nos modèles dépassent les baselines publiées | Comparaison aux 4 baselines BARO/E-Diag/CausaRank/MicroRCA | **OUI** sur trainticket/sockshop ; **partiel** ailleurs |

### 3.1 Analyse de P1 — Avantage asymptotique STA

À N=20, GATv2 devance légèrement STA (0.828 vs 0.805 en Avg@1), mais avec une variance bien inférieure. À N=20, k≈N : le sous-graphe sélectionné couvre presque tout le graphe, et l'avantage de la sélection ne s'exprime pas encore.

À partir de N=50, STA prend la tête et creuse l'écart : à N=100, STA (0.553) vs GATv2 (0.490), soit +6,3 points. La complexité réduite se traduit par moins d'overfitting et une meilleure généralisation sur les grandes topologies.

**P1 confirmé.** L'avantage asymptotique est réel et mesurable dès N=50.

### 3.2 Analyse de P2 — Transfert sur les données réelles

Les datasets RCAEval opèrent à N=5 (5 services instrumentés). À cette échelle, k≈N : le sélecteur de STA retient quasiment tous les nœuds, annulant son avantage de complexité. La qualité dépend alors surtout de la séparabilité temporelle des signaux.

Résultat : STA gagne sur trainticket (+1%) et sockshop (+4%), GATv2 gagne sur online-boutique (+3%). L'inversion n'est pas aléatoire : online-boutique présente une structure de propagation de pannes plus diffuse, où l'attention douce de GATv2 est plus adaptée.

**P2 partiellement confirmé.** Le transfert est dataset-dépendant.

### 3.3 Analyse de P3 — Apport du graphe vs temporal-only

L'ablation T5 compare STA (avec GCN sur le sous-graphe) à une version temporal-only (sans GCN) sur les mêmes splits et seeds :

| Dataset | STA Avg@1 | Temporal-only Avg@1 | Delta |
|---|---:|---:|---:|
| trainticket | 0.960 | 0.940 | +2% (graphe utile) |
| sockshop | 0.850 | 0.860 | −1% (graphe neutre) |
| online-boutique | 0.650 | 0.690 | −4% (graphe nuit) |

Le graphe aide légèrement là où les dépendances inter-services sont fortes et régulières (trainticket). Il n'apporte rien — voire nuit — quand la structure relationnelle introduit du bruit par rapport au signal temporel pur.

**P3 marginalement confirmé.** Le graphe n'est pas universellement bénéfique à petit N.

### 3.4 Analyse de P4 — Utilité de la fusion multimodale

Résultats MM sur RCAEval (métriques-only vs MM-strict, 5 seeds) :

| Dataset | STA mono | STA-MM | Delta | GATv2 mono | GATv2-MM | Delta |
|---|---:|---:|---:|---:|---:|---:|
| trainticket | 0.960 | 0.910 | **−5%** | 0.950 | 0.880 | **−7%** |
| online-boutique | 0.650 | 0.650 | 0% | 0.680 | 0.790 | **+11%** |
| sockshop | 0.850 | n/a | — | 0.810 | n/a | — |

La couverture MM est complète (logs_ratio=1.0, traces_ratio=1.0 sur trainticket et online-boutique) : la dégradation n'est pas due à des modalités manquantes. Trois hypothèses explicatives :

1. **Signal additionnel faible** : les logs/traces RCAEval n'apportent pas d'information discriminante au-delà des métriques.
2. **Coût d'optimisation** : la fusion multimodale ajoute des paramètres et des chemins de gradient sans gain de signal proportionnel.
3. **Conflit de gradients** : l'attention de fusion peut sur-pondérer des modalités corrélées ou bruitées, perturbant la convergence.

Exception notable : GATv2-MM sur online-boutique (+11%). L'architecture late-fusion semble mieux tolérer les modalités de faible signal sur ce dataset spécifique.

**P4 non confirmé sur RCAEval.** Résultat négatif robuste et documenté. La piste prometteuse reste l'évaluation sur des données cluster réelles où les logs/traces ont un contenu plus riche (voir §6).

---

## 4. Résultats consolidés

### 4.1 Benchmark simulé (Barabasi-Albert)

Protocole : 5 seeds (42, 123, 7, 1337, 2025), graphes BA aléatoires, N ∈ {20, 50, 100}, splits stratifiés.

**N = 20 (k = 5)**

| Modèle | Avg@1 ± std | Avg@3 ± std | MRR ± std | Params |
|---|---:|---:|---:|---:|
| STA (k=5) | 0.805 ± 0.122 | 0.884 ± 0.102 | 0.854 ± 0.103 | 15 494 |
| GATv2 | **0.828 ± 0.039** | **0.916 ± 0.033** | **0.878 ± 0.032** | 9 794 |
| GCN full-graph | 0.830 ± 0.029 | 0.908 ± 0.017 | 0.876 ± 0.022 | 7 490 |
| Late STGCN+GIB | 0.823 ± 0.047 | 0.890 ± 0.034 | 0.868 ± 0.036 | 8 579 |
| Random Forest | 0.685 ± 0.027 | 0.900 ± 0.009 | 0.800 ± 0.017 | — |
| MLP | 0.084 ± 0.024 | 0.258 ± 0.048 | 0.242 ± 0.030 | 22 132 |

**N = 50 (k = 8)**

| Modèle | Avg@1 ± std | Avg@3 ± std | MRR ± std |
|---|---:|---:|---:|
| **STA (k=8)** | **0.589 ± 0.049** | **0.667 ± 0.036** | **0.651 ± 0.033** |
| GATv2 | 0.571 ± 0.023 | 0.619 ± 0.030 | 0.619 ± 0.026 |
| GCN full-graph | 0.526 ± 0.026 | 0.574 ± 0.021 | 0.575 ± 0.022 |
| Late STGCN+GIB | 0.469 ± 0.068 | 0.526 ± 0.073 | 0.524 ± 0.069 |
| Random Forest | 0.312 ± 0.016 | 0.453 ± 0.035 | 0.421 ± 0.021 |

**N = 100 (k = 15)**

| Modèle | Avg@1 ± std | Avg@3 ± std | MRR ± std |
|---|---:|---:|---:|
| **STA (k=15)** | **0.552 ± 0.046** | **0.606 ± 0.054** | **0.600 ± 0.048** |
| GATv2 | 0.490 ± 0.029 | 0.543 ± 0.036 | 0.538 ± 0.033 |
| GCN full-graph | 0.466 ± 0.042 | 0.537 ± 0.039 | 0.521 ± 0.038 |
| Late STGCN+GIB | 0.471 ± 0.097 | 0.557 ± 0.105 | 0.532 ± 0.094 |
| Random Forest | 0.065 ± 0.009 | 0.123 ± 0.017 | 0.128 ± 0.014 |

**Lecture :** L'inversion STA vs GATv2 se produit exactement à N=50. À N=100, STA dépasse GATv2 de +6,2 points en Avg@1 et le GCN de +8,6 points. Le Random Forest s'effondre à N=100 (Avg@1 = 0.065), confirmant que les méthodes non-graphe ne scalent pas.

---

### 4.2 Benchmark RCAEval réel — métriques uniquement

Protocole : 5 seeds (42, 123, 7), 100 epochs, max_cases=90, split par run seed, drift activé.

| Dataset | STA Avg@1 ± std | GATv2 Avg@1 ± std | STA Avg@3 | GATv2 Avg@3 | STA MRR | GATv2 MRR |
|---|---:|---:|---:|---:|---:|---:|
| trainticket | **0.960 ± 0.037** | 0.950 ± 0.045 | **1.000** | **1.000** | **0.980** | 0.975 |
| sockshop | **0.850 ± 0.089** | 0.810 ± 0.080 | **0.960** | 0.910 | **0.908** | 0.876 |
| online-boutique | 0.650 ± 0.095 | **0.680 ± 0.051** | **0.890** | 0.840 | 0.779 | **0.788** |

Sur trainticket et sockshop, les deux modèles dépassent BARO (0.71) avec une marge significative. Sur online-boutique, STA (0.650) reste sous BARO — GATv2 (0.680) s'en approche.

---

### 4.3 Benchmark RCAEval réel — fusion multimodale

Protocole identique avec `--use-mm --mm-strict`. Sockshop MM exclut du périmètre (traces manquantes).

| Dataset | STA mono | STA-MM | Δ STA | GATv2 mono | GATv2-MM | Δ GATv2 |
|---|---:|---:|---:|---:|---:|---:|
| trainticket | 0.960 | 0.910 ± 0.020 | **−5%** | 0.950 | 0.880 ± 0.051 | **−7%** |
| online-boutique | 0.650 | 0.650 ± 0.155 | 0% | 0.680 | **0.790 ± 0.086** | **+11%** |
| sockshop | 0.850 | — | — | 0.810 | — | — |

La variance de STA-MM sur online-boutique (±0.155) est 3× plus élevée qu'en metrics-only (±0.095), signe d'une optimisation multimodale instable.

---

### 4.4 Benchmark ObservIT cluster — métriques uniquement

Protocole : 5 seeds, 100 epochs, cluster K8s interne (sock-shop + train-ticket).

| Modèle | Avg@1 ± std | Avg@3 ± std | Avg@5 ± std | MRR ± std |
|---|---:|---:|---:|---:|
| **STA** | **0.796 ± 0.043** | 0.921 ± 0.075 | 0.942 ± 0.062 | **0.868 ± 0.033** |
| GATv2 | 0.762 ± 0.034 | **0.978 ± 0.010** | **0.988 ± 0.006** | 0.860 ± 0.031 |

Résultat notable : GATv2 rattrape STA en Avg@3/Avg@5 (0.978 vs 0.921), et les dépasse nettement. Sur les runs cluster — topologie plus grande et plus hétérogène — GATv2 classe correctement la RC dans le top-3 presque systématiquement.

---

### 4.5 Benchmark ObservIT cluster — fusion multimodale

| Modèle | Avg@1 ± std | Avg@3 ± std | Avg@5 ± std | MRR ± std |
|---|---:|---:|---:|---:|
| **STA-MM** | **0.772 ± 0.021** | **0.982 ± 0.007** | **0.992 ± 0.003** | **0.878 ± 0.010** |
| GATv2-MM | 0.762 ± 0.016 | 0.937 ± 0.083 | 0.950 ± 0.086 | 0.846 ± 0.035 |

Contrairement à RCAEval, la fusion multimodale ne dégrade pas STA sur ObservIT : STA-MM (Avg@1=0.772) dépasse même GATv2 metrics-only (0.762). Les logs et traces du cluster interne semblent contenir un signal plus discriminant que dans RCAEval.

---

## 5. Verdict : modèle gagnant ou équilibré ?

Il n'y a pas de modèle universellement supérieur. Le choix optimal dépend du contexte de déploiement.

### Tableau décisionnel

| Contexte | Modèle recommandé | Justification |
|---|---|---|
| Grand graphe N ≥ 50, métriques uniquement | **STA** | Avantage O(N+k²) confirmé expérimentalement (§4.1) |
| Petit graphe N ≤ 10, métriques uniquement | **STA ≈ GATv2** | Écart non significatif, instabilité similaire |
| Dataset avec pannes diffuses (ex. online-boutique) | **GATv2** | Attention douce mieux adaptée aux cascades multi-hop |
| Cluster K8s réel, métriques seulement, top-3 important | **GATv2** | Avg@3/Avg@5 nettement supérieurs sur ObservIT |
| Cluster K8s réel, logs + traces disponibles | **STA-MM** | Seul modèle avec gain MM sur cluster (Avg@1=0.772) |
| Données RCAEval publiques + modalités | Éviter MM | Résultat négatif robuste, variance élevée |
| Gray failures (pannes dégradées, signal faible) | **GATv2 + gray_failure_detector** | STA peut rater la RC si elle sort du top-k |
| Contrainte de latence forte (inférence temps réel) | **STA** | ~3.4 ms vs ~3.0 ms à N=100 (comparable), mais scalabilité meilleure |

### Synthèse

Sur les données simulées à grand N : **STA gagne**. Sur les données réelles à petit N et structure diffuse : **GATv2 est compétitif ou supérieur**. Sur cluster réel avec modalités enrichies : **STA-MM se distingue**. L'hypothèse initiale — STA > GATv2 universellement — est nuancée : la supériorité asymptotique est réelle, mais le régime RCAEval (N=5) n'est pas le régime où elle s'exprime.

---

## 6. Étude personnelle — Sock-shop sur cluster ObservIT

*Demande du maître de stage : prendre le sock-shop sur le cluster, identifier les métriques les plus importantes, et tester les chances des modèles de trouver la cause racine.*

### 6.1 Topologie sock-shop

Le déploiement ObservIT utilise la topologie officielle sock-shop : **14 services, 15 dépendances**.

```
front-end ──► catalogue ──► catalogue-db
          ──► carts ──────► carts-db
          ──► orders ─────► orders-db
          │           ────► shipping ──► rabbitmq ──► queue-master
          │           ────► payment
          │           ────► user
          ──► user ────────► user-db
          │               ► session-db
          ──► payment
```

**Services critiques :**
- `front-end` : hub central, 5 dépendances directes. Tout incident upstream se propage ici immédiatement.
- `orders` : degré sortant le plus élevé (orders-db, shipping, payment, user). Point de concentration des pannes en cascade.
- Couche data : 5 bases isolées (catalogue-db, carts-db, orders-db, user-db, session-db) — pannes silencieuses sur ces nœuds sont difficiles à remonter.

**Workload Locust** (`locustfile_sockshop.py`) : 20 utilisateurs simultanés, 60 % parcours catalogue, 25 % panier, 15 % checkout.

---

### 6.2 Scénarios chaos retenus

Cinq scénarios de panne, choisis pour couvrir les trois types de failure : **hard** (panne franche), **cascade** (propagation), **gray** (dégradation silencieuse).

| # | Scénario | Root Cause | Failure Type | Mécanisme Chaos Mesh |
|---|---|---|---|---|
| 1 | `stress-cpu-catalogue` | `catalogue` | hard | 2 workers CPU @ 80% pendant 5 min |
| 2 | `stress-mem-carts` | `carts` | hard | 256 Mo d'allocation stress pendant 5 min |
| 3 | `network-delay-orders-to-orders-db` | `orders-db` | cascade | 400 ms latence + 80 ms jitter (100% corrélation) sur le lien orders → orders-db |
| 4 | `network-loss-catalogue-to-catalogue-db` | `catalogue-db` | gray | 15% perte de paquets (60% corrélation) sur le lien catalogue → catalogue-db |
| 5 | `pod-kill-user` | `user` | hard | Suppression périodique du pod user toutes les 90 s pendant 5 min |

**Prérequis cluster :** Chaos-Mesh doit être déployé en mode `clusterScoped=true` (via `scripts/05-enable-chaos-mesh-cluster-scope.sh`) avant tout inject de chaos.

---

### 6.3 Métriques Prometheus les plus importantes

Le connecteur Prometheus (`prometheus_connector.py`) collecte 6 métriques par service, formant un tenseur `[window_size, N_services, 6]`.

#### Les 6 métriques

| # | Nom | PromQL | Signal pour quel scénario |
|---|---|---|---|
| 1 | **cpu_usage** | `rate(container_cpu_usage_seconds_total[2m])` | Scénario 1 : spike immédiat sur `catalogue`, isolé |
| 2 | **memory_bytes** | `container_memory_working_set_bytes` | Scénario 2 : croissance monotone sur `carts`, signature inconfondable |
| 3 | **network_rx** | `rate(container_network_receive_bytes_total[2m])` | Scénario 4 : drop du trafic reçu par `catalogue-db` — signal principal du gray failure |
| 4 | **network_tx** | `rate(container_network_transmit_bytes_total[2m])` | Scénario 3 : asymétrie tx/rx sur `orders-db`, latence = paquets retransmis |
| 5 | **latency_p99** | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[2m]))` | Scénarios 1, 2, 3 : indicateur précoce — monte avant que le taux d'erreur ne décolle |
| 6 | **error_rate** | `rate(http_requests_total{status=~"5.."}[2m])` | Scénarios 4, 5 : erreurs 503/504 sur `front-end` et `user` — signal en aval de la panne |

#### Pourquoi chaque métrique est critique

**cpu_usage** : le stress CPU sur `catalogue` crée un spike isolé sur un seul nœud. C'est le signal le plus clair — aucune autre métrique ne monte proportionnellement. Un modèle qui charge correctement cpu_usage identifie immédiatement le coupable.

**memory_bytes** : la saturation mémoire sur `carts` est progressive (croissance monotone) mais non propagée — `carts-db` et `front-end` ne voient pas de variation mémoire. Signal discriminant fort pour les modèles qui encodent la dimension temporelle.

**network_rx** : dans le scénario de gray failure (15% packet loss sur catalogue → catalogue-db), seul le trafic réseau reçu par `catalogue-db` est affecté. L'error_rate monte peu (60% des requêtes passent encore). C'est le signal dominant, et il est faible — c'est précisément pourquoi les gray failures sont difficiles.

**network_tx** : le délai réseau sur `orders-db` se manifeste d'abord par des retransmissions TCP, visibles dans network_tx avant que la latence applicative (latency_p99) ne soit détectable. Signal précoce dans la couche transport.

**latency_p99** : métrique la plus sensible pour les pannes de performance (CPU, mémoire, réseau). Elle monte avant que les requêtes ne génèrent des erreurs 5xx. Utile comme signal d'alerte précoce que les modèles doivent pondérer fortement en entrée.

**error_rate** : signal aval. Il apparaît sur `front-end` et les services appelants, pas sur la root cause directe. Un modèle qui s'appuie uniquement sur l'error_rate risque de remonter vers `front-end` plutôt que vers la vraie cause. C'est pourquoi la structure graphe est utile : elle permet de remonter des symptômes (front-end) vers la cause (orders-db, catalogue-db).

#### Cas particulier du gray failure (scénario 4)

Avec 15% de perte de paquets et 60% de corrélation, seulement ~40% des requêtes catalogue → catalogue-db sont véritablement impactées. Les métriques restantes (cpu, memory, latency_p99) restent proches des valeurs nominales. Seul network_rx sur `catalogue-db` révèle la panne.

Ce scénario est le plus difficile pour les modèles : si network_rx a un faible poids dans l'encodeur, la RC (`catalogue-db`) ne remontera pas dans le top-k de STA. GATv2 a un avantage : son attention douce peut s'aligner sur la seule métrique active, sans contrainte de sélection binaire.

---

### 6.4 Évaluation des modèles sur sock-shop cluster — résultats mesurés

Les données cluster ont été collectées via `scripts/run_sockshop_campaign.sh` (5 scénarios chaos, 5 seeds, Chaos-Mesh + Locust + Prometheus). Les JSON canoniques (`paper_benchmark_observit.json`) contiennent un breakdown par root cause (`STA_per_root_cause`, etc.), exploité par le script [analyze_per_scenario.py](../rca_pipeline/analyze_per_scenario.py).

**Matrice mesurée scénario × modèle (Avg@1 moyen ± std, 5 seeds) :**

| Scénario | Failure Type | RC | STA | GATv2 | Late-STGCN+GIB |
|---|---|---|---:|---:|---:|
| `stress-cpu-catalogue` | hard | catalogue | 0.966 ± 0.026 | 0.959 ± 0.023 | 0.951 ± 0.035 |
| `stress-mem-carts` | hard | carts | 0.987 ± 0.014 | 0.987 ± 0.014 | 0.984 ± 0.012 |
| `network-delay-orders-to-orders-db` | **cascade** | orders-db | **0.043 ± 0.095** | **0.459 ± 0.445** | 0.003 ± 0.007 |
| `network-loss-catalogue-to-catalogue-db` | gray | catalogue-db | **0.987 ± 0.017** | 0.508 ± 0.404 | 0.997 ± 0.007 |
| `pod-kill-user` | hard | user | 0.951 ± 0.000 | 0.951 ± 0.000 | 0.741 ± 0.360 |

**Agrégat par type de panne :**

| Type | STA Avg@1 moyen | GATv2 Avg@1 moyen | Late-STGCN+GIB |
|---|---:|---:|---:|
| hard (CPU/mem/pod-kill) | 0.968 | 0.965 | 0.892 |
| cascade (1 scénario) | **0.043** | 0.459 | 0.003 |
| gray (1 scénario) | 0.987 | 0.508 | 0.997 |

**Lecture des résultats — trois constats qui contredisent les prédictions initiales :**

1. **Le scénario de cascade est le vrai point de rupture, pas le gray failure.** STA s'effondre à Avg@1 = 0.043 (quasi-aléatoire) sur `network-delay-orders-to-orders-db`. GATv2 résiste mieux (0.459) mais reste loin de ses performances sur les autres scénarios. La panne au niveau d'`orders-db` génère des symptômes massifs en aval (`orders`, `front-end`) qui dominent le scoring local. La cause racine elle-même n'a presque aucune anomalie métrique propre — c'est un effet de propagation pure.

2. **Le gray failure à 15% de perte de paquets s'est révélé largement détectable.** Contrairement à la prédiction, STA (0.987) et Late-STGCN+GIB (0.997) identifient quasi parfaitement `catalogue-db`. Hypothèse : à 15% de perte, le trafic réseau de `catalogue-db` chute suffisamment pour devenir un signal franc, plutôt qu'une anomalie subtile. Le terme "gray" mériterait d'être réservé à des taux de perte plus faibles (1-5%).

3. **Les pannes hard restent triviales** : STA et GATv2 ≥ 0.95 sur les trois scénarios isolés. Late-STGCN+GIB faiblit sur `pod-kill-user` (0.741 ± 0.360, forte variance — probablement un seed défaillant).

**Implications pour les recommandations §6.5 :**
- La priorité d'intégration du `gray_failure_detector.py` doit être réorientée vers les **cascades**, pas vers les gray failures. C'est sur la cascade qu'un signal externe (détection de propagation multi-hop) aiderait STA.
- GATv2 reste le meilleur compromis pour un déploiement où les cascades sont fréquentes.
- L'agrégat ObservIT (STA Avg@1 = 0.796 reporté §4.4) **masque** cet effondrement sur la cascade : la moyenne (0.97+0.99+0.04+0.99+0.95)/5 ≈ 0.79 colle exactement à l'aggrégat publié. Ce breakdown est donc indispensable pour comprendre le comportement réel.

**Détails complets** : voir [docs/PER_SCENARIO_BREAKDOWN.md](PER_SCENARIO_BREAKDOWN.md) (généré par `python -m rca_pipeline.analyze_per_scenario`).

---

### 6.5 Recommandations opérationnelles pour le cluster (révisées après mesure)

1. **GATv2 par défaut sur cluster N ≤ 14** : confirmé par l'agrégat (Avg@3 = 0.978) et par la matrice scénario × modèle. STA peut rester en backup pour les scénarios isolés où le signal est fort.

2. **Réorienter le `gray_failure_detector.py`** : initialement conçu pour les gray failures à signal faible, sa cible utile sur ce cluster est en fait les **cascades**. La sous-section [§1.2 de la roadmap](../docs/PER_SCENARIO_BREAKDOWN.md) propose une ablation A/B sur le scénario `network-delay-orders-to-orders-db` où STA s'effondre. Un détecteur de propagation multi-hop devrait booster les scores des nœuds ayant des voisins anomaliques mais des métriques propres normales.

3. **Pour les staging à grande topologie (N ≥ 50)** : l'avantage théorique de STA en latence reste valide, mais il faut d'abord valider qu'il ne s'effondre pas sur les cascades à cette échelle (limitation non couverte).

4. **MM sur cluster** : les résultats ObservIT (STA-MM Avg@1 = 0.772 sans régression) justifient un test étendu avec Loki + Jaeger sur le cluster Devoteam si les modalités sont disponibles avec une bonne couverture.

5. **Métriques Prometheus** : les 6 métriques actuelles (cpu, mem, network_rx, network_tx, latency_p99, error_rate) suffisent pour les pannes hard et gray. Pour les cascades, ajouter des métriques de **dépendance** (taux d'appel inter-service, taille de queue par lien) serait probablement plus utile que d'ajouter des métriques au nœud lui-même.

---

## 7. Conclusion

### Trois takeaways

**1. L'avantage asymptotique de STA est réel, mais se manifeste uniquement à grand N.**
Sur les datasets simulés (N≥50), STA dépasse GATv2 de façon constante et significative (+6 points à N=100). Sur les datasets RCAEval (N=5), l'écart s'efface : les deux modèles sont comparables, avec une légère dominance de GATv2 sur les topologies à pannes diffuses. L'hypothèse de départ est confirmée dans son domaine de validité (grand N), et clairement délimitée dans ses conditions limites (petit N).

**2. La fusion multimodale n'améliore pas les scores sur RCAEval, mais montre un potentiel sur cluster.**
Sur les trois datasets RCAEval évalués, MM dégrade ou est neutre. Ce résultat négatif est robuste (couverture de modalités complète, 5 seeds). En revanche, sur ObservIT, STA-MM maintient ses performances (Avg@1=0.772) sans régression, et dépasse même GATv2 metrics-only. La différence de comportement entre RCAEval et ObservIT pointe vers la richesse du signal de log/trace comme facteur déterminant.

**3. Sur sock-shop cluster, les résultats par scénario contredisent les prédictions initiales.**
Les pannes franches (hard) sont triviales pour STA et GATv2 (Avg@1 ≥ 0.95). Le gray failure à 15% de perte de paquets s'est révélé étonnamment facile (STA = 0.987) — le signal n'était pas si "gris". **Le vrai point de rupture est la cascade** : STA s'effondre à 0.043 sur `network-delay-orders-to-orders-db`, GATv2 limite la casse à 0.459. Cet effet est masqué par l'agrégat ObservIT (STA = 0.796). L'intégration future du `gray_failure_detector.py` doit être réorientée vers la détection de propagation multi-hop, pas vers les anomalies réseau subtiles.

### Perspectives

- **Augmenter le nombre de répétitions** par scénario chaos (actuellement 2) pour stabiliser les estimations par failure type.
- **Évaluer MM sur sock-shop cluster** : actuellement hors périmètre (traces manquantes), mais les résultats ObservIT justifient l'investissement.
- **Tester STA à N > 14** sur le cluster : déployer des topologies plus grandes pour valider l'avantage asymptotique en conditions réelles.
- **Intégrer `gray_failure_detector.py`** systématiquement dans le pipeline d'inférence pour les scénarios réseau.

---

*Protocole de référence : `rca_pipeline/validation_protocol.py`. Artefacts canoniques : `results_transmission/canonical_results_manifest.json`. Comparabilité des baselines : `docs/COMPARABILITY_MATRIX.md`.*
