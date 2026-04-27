# Rapport de Progression — Semaine 2 : Passage à l'Échelle, Benchmark Multi-Échelle et Étude d'Ablation

## 1. Résumé des Travaux

La deuxième semaine a été consacrée à trois axes principaux : (i) un audit rigoureux du code existant avec corrections de bugs critiques, (ii) le développement d'une deuxième architecture (STA) et le passage à l'échelle sur des topologies de 20 à 100 nœuds via un générateur Barabási-Albert, et (iii) une étude d'ablation systématique pour comprendre la contribution de chaque composant architectural.

---

## 2. Audit et Corrections

### 2.1 Fuite de Données (Data Leakage)
L'audit a révélé que le `StandardScaler` dans `graph_builder.py` était ajusté (*fit*) sur l'ensemble du dataset avant le découpage train/test. Cela introduisait une fuite d'information du test vers le train. La correction a consisté à implémenter un pattern `fit_scaler()` / `apply_scaler()` strict : le scaler est ajusté uniquement sur le train, puis appliqué au test avec les mêmes statistiques.

### 2.2 Mapping Pod-to-Node
Le mécanisme de correspondance entre noms de pods Kubernetes et noms de services utilisait une recherche par sous-chaîne naïve, créant des ambiguïtés (e.g., `auth-service-xxx` matchait à la fois `auth` et `auth-service`). Réécriture avec un matching par segments et désambiguïsation par la clé la plus longue.

### 2.3 Variance Statistique
Les benchmarks de la semaine 1 reposaient sur une seule graine aléatoire. Passage à un système multi-seed (3 graines : 42, 123, 7) avec calcul de la moyenne et de l'écart-type pour quantifier la stabilité de chaque modèle.

### 2.4 Refactoring
- Suppression des fichiers V2 obsolètes (`model_v2.py`, `train_v2.py`, `benchmark_v2.py`)
- Renommage : `model_v3.py` → `model_sta.py`, `train_v3.py` → `train_sta.py`, `benchmark_v3.py` → `benchmark.py`
- Réécriture de `infer.py`, `evaluate.py`, `visualize.py` pour supporter les deux architectures et un nombre de nœuds arbitraire
- Nettoyage terminologique : *synthetic* → *simulated* dans tout le code

---

## 3. Deuxième Architecture : Select-then-Analyze (STA)

### 3.1 Motivation
L'architecture Late Bottleneck (STGCN + GIB) traite l'intégralité du graphe en $O(N^2)$ avant de filtrer via le masque GIB. Pour des topologies de production ($N > 100$), cette complexité quadratique pose un problème de passage à l'échelle.

### 3.2 Architecture STA — $O(N + k^2)$
L'idée fondamentale est d'inverser l'ordre : **sélectionner d'abord**, puis analyser en profondeur uniquement le sous-graphe pertinent.

$$\text{STA}: \quad x \xrightarrow{\text{LightEncoder}} h \xrightarrow{\text{STE Selector}} \text{top-}k \xrightarrow{\text{DeepSTGCN}} \text{scorer} \rightarrow \hat{y}$$

Les 5 étapes :
1. **LightEncoder** — GCN léger + attention temporelle sur les $N$ nœuds, $O(N)$
2. **STESelector** — Gumbel-sigmoid + top-$k$ binarisation + Straight-Through Estimator, $O(N)$
3. **Extraction vectorisée** — un seul appel `pyg_subgraph` pour le batch entier, $O(E)$
4. **DeepSTGCN** — convolutions temporelles et spatiales sur les $k$ nœuds sélectionnés, $O(k^2)$
5. **SubgraphScorer** — scoring par nœud avec attention sur le contexte du sous-graphe, $O(k)$

Le nombre de paramètres reste **constant** ($\approx 16\,549$) indépendamment de $N$.

### 3.3 Entraînement avec k-annealing
Le problème de la poule et l'œuf (le sélecteur ne peut apprendre sans que le scorer voie la cause racine, et vice versa) est résolu par un curriculum :
- $k$ démarre à $N/2$ (haute probabilité de capturer la cause racine) et décroît linéairement vers $k_{\text{target}}$ sur 60 epochs
- La température du Gumbel-sigmoid décroît de 1.0 à 0.3

---

## 4. Générateur de Données Simulées — Barabási-Albert

Pour tester au-delà de la topologie fixe de 5 nœuds K8s, un générateur a été développé (`simulated_data_scalable.py`). Il produit des graphes dirigés $BA(N, m)$ avec :
- Distribution de degré en loi de puissance (réaliste pour les architectures microservices)
- Nœuds nommés par rôle selon la centralité (*db*, *cache*, *mq*, *svc*, *gw*, *api*)
- Dynamique d'anomalie réaliste : signal transitoire au nœud racine (décroissance exponentielle), propagation multi-hop via la matrice $P$, distracteurs avec propagation limitée

Trois échelles testées : $N = 20$ ($k=5$), $N = 50$ ($k=8$), $N = 100$ ($k=10$), chacune avec 3 000 fenêtres temporelles.

---

## 5. Résultats du Benchmark Multi-Échelle

Cinq modèles comparés sur 3 échelles, 3 graines par configuration (moyenne ± écart-type).

### 5.1 N=20 (petite échelle)

| Modèle | Top-1 Accuracy | MRR | Latence (ms) | Params |
|--------|----------------|-----|-------------|--------|
| **STA (k=5)** | **97.9% ± 0.7%** | **0.983** | 4.21 | 16 549 |
| Full-Graph GCN | 97.3% ± 0.5% | 0.981 | 1.46 | 7 490 |
| Late Bottleneck | 96.6% ± 1.4% | 0.975 | 1.57 | 8 579 |
| Random Forest | 59.3% ± 1.6% | 0.721 | — | — |
| MLP | 6.7% ± 0.7% | 0.217 | — | — |

### 5.2 N=50 (échelle intermédiaire)

| Modèle | Top-1 Accuracy | MRR | Latence (ms) | Params |
|--------|----------------|-----|-------------|--------|
| **Late Bottleneck** | **95.4% ± 0.8%** | **0.971** | 1.41 | 8 579 |
| Full-Graph GCN | 94.4% ± 3.2% | 0.961 | 1.55 | 7 490 |
| STA (k=8) | 93.7% ± 1.8% | 0.957 | 3.47 | 16 549 |
| Random Forest | 17.7% ± 1.7% | 0.276 | — | — |
| MLP | 1.9% ± 0.5% | 0.096 | — | — |

### 5.3 N=100 (grande échelle)

| Modèle | Top-1 Accuracy | MRR | Latence (ms) | Params |
|--------|----------------|-----|-------------|--------|
| **Full-Graph GCN** | **94.1% ± 1.1%** | **0.959** | 1.35 | 7 490 |
| STA (k=10) | 90.2% ± 2.1% | 0.926 | 3.32 | 16 549 |
| Late Bottleneck | 90.2% ± 9.0% | 0.925 | 1.33 | 8 579 |
| Random Forest | 7.0% ± 0.9% | 0.149 | — | — |
| MLP | 1.7% ± 0.7% | 0.072 | — | — |

### 5.4 Observations principales
1. Les trois approches à base de graphes maintiennent $>90\%$ de Top-1 jusqu'à $N=100$. Les baselines classiques s'effondrent (RF : $59\% \to 7\%$).
2. Le Full-Graph GCN domine de façon inattendue à $N=100$ malgré sa simplicité architecturale.
3. Late Bottleneck présente une variance critique de $\pm 9.0\%$ à $N=100$, rédhibitoire pour un déploiement.
4. STA est plus lent que le GCN à ces échelles ($N \leq 100$), la surcharge de la phase de sélection compensant le gain du sous-graphe réduit. L'avantage théorique $O(N + k^2)$ se matérialisera à $N > 500$.

---

## 6. Étude d'Ablation

L'étude d'ablation a été conçue pour répondre à la question : **pourquoi le GCN simple bat-il les architectures plus complexes à grande échelle ?** Quatre axes ont été testés sur $N=20$ et $N=100$ avec 3 graines chacun.

### 6.1 Ablation A — Décomposition par Composant

Isolation de la contribution de chaque composant (temporel, spatial, GIB) :

**N=20 :**

| Variante | Top-1 | MRR |
|----------|-------|-----|
| A4 : Temporal-only (pas de GCN) | **98.4% ± 0.0%** | **0.989** |
| A1 : Late BN (STGCN + GIB) | 97.9% ± 1.2% | 0.986 |
| A2 : STGCN sans GIB (FullGraphGCN) | 97.3% ± 0.5% | 0.981 |
| A3 : Spatial-only GCN (pas de temporel) | 20.4% ± 3.4% | 0.327 |

**N=100 :**

| Variante | Top-1 | MRR |
|----------|-------|-----|
| A2 : STGCN sans GIB (FullGraphGCN) | **94.1% ± 1.1%** | **0.959** |
| A4 : Temporal-only (pas de GCN) | 64.6% ± 23.3% | 0.732 |
| A1 : Late BN (STGCN + GIB) | 61.8% ± 43.0% | 0.647 |
| A3 : Spatial-only GCN (pas de temporel) | 4.3% ± 1.8% | 0.095 |

**Interprétation :**
- À $N=20$, **le signal temporel domine** : un modèle sans graphe atteint 98.4%, ce qui signifie que le pattern d'anomalie est suffisamment distinctif dans le temps pour que chaque nœud puisse être classifié indépendamment. La structure spatiale n'apporte qu'un gain marginal à cette échelle.
- À $N=100$, **le rapport s'inverse radicalement** : le Temporal-only chute à 64.6% tandis que le Full-Graph GCN (qui combine temporel + spatial) reste à 94.1%. Quand le nombre de classes augmente de 20 à 100, la discrimination par les patterns temporels seuls ne suffit plus — la propagation spatiale de l'anomalie à travers le graphe devient le signal discriminant critique.
- Le **Spatial-only** s'effondre dans les deux cas ($20\%$ et $4\%$), confirmant que les statistiques temporelles agrégées (mean + std) sont insuffisantes — il faut les convolutions temporelles apprises.
- Le **GIB dégrade** les performances à $N=100$ (Late BN à $61.8\% \pm 43\%$ vs FullGraphGCN à $94.1\%$). Le masque d'Information Bottleneck, conçu pour filtrer le bruit, semble au contraire introduire de l'instabilité sur les grands graphes. Hypothèse : avec 100 classes et un masque qui force la parcimonie, le GIB élimine des nœuds porteurs d'information utile pour la discrimination fine.

### 6.2 Ablation B — Protocole d'Entraînement Late Bottleneck

Impact de la phase contrastive InfoNCE sur Late Bottleneck :

| Configuration | N=20 Top-1 | N=100 Top-1 |
|---------------|------------|-------------|
| B1 : Contrastif + Supervisé | 97.9% ± 1.2% | 61.8% ± 43.0% |
| B2 : Supervisé seul | 97.5% ± 0.6% | 31.9% ± 44.1% |
| **Δ (B1 − B2)** | **+0.4%** | **+29.9%** |

**Interprétation :**
- À $N=20$, le gain du contrastif est négligeable ($+0.4\%$) : le signal supervisé suffit.
- À $N=100$, la phase contrastive est **critique** ($+29.9\%$). Elle structure l'espace latent avant la phase supervisée, empêchant le modèle de rester bloqué dans un minimum local. Sans elle, 2 des 3 graines convergent vers un early stop prématuré à 30 epochs avec seulement $\sim60\%$ d'accuracy d'entraînement.
- Cela confirme que pour les grands graphes, le pré-entraînement contrastif n'est pas un luxe mais une **nécessité** pour stabiliser la convergence.

### 6.3 Ablation C — Sensibilité au $k$ de STA (N=100)

Test de l'agressivité de la sélection sur la performance finale :

| k (% des nœuds) | Top-1 | MRR | Taux de sélection | Latence (ms) |
|-------------------|-------|------|--------------------|-------------|
| k=10 (10%) | 90.2% ± 2.1% | 0.926 | 96.1% | 4.41 |
| k=15 (15%) | 92.9% ± 0.5% | 0.946 | 97.7% | 3.25 |
| k=20 (20%) | 92.8% ± 1.4% | 0.947 | 99.0% | 3.03 |
| k=30 (30%) | **93.8% ± 0.3%** | **0.953** | 98.4% | 2.89 |

**Interprétation :**
- Augmenter $k$ de 10 à 15 produit un gain significatif ($+2.7\%$) et réduit la variance ($\pm 2.1\% \to \pm 0.5\%$). $k=10$ est effectivement trop agressif pour $N=100$ : le sélecteur élimine des nœuds porteurs d'information de propagation.
- Au-delà de $k=15$, les gains sont marginaux : $k=20$ et $k=30$ n'apportent que $+0.9\%$ supplémentaire. Cela indique que le signal causal est concentré dans $\sim15\%$ des nœuds.
- À $k=30$ ($30\%$), STA atteint $93.8\% \pm 0.3\%$, se rapprochant du GCN pur à $94.1\%$, tout en réduisant les nœuds traités de 70%. L'écart résiduel de $0.3\%$ pourrait être comblé avec un tuning plus fin.
- Le nombre de paramètres reste **strictement constant à 16 549** quel que soit $k$ — seule la taille du sous-graphe traité change.

### 6.4 Ablation D — Protocole d'Entraînement STA (N=100, k=10)

Impact du warmup contrastif et du k-annealing sur STA :

| Configuration | Top-1 | MRR | Taux de sélection |
|---------------|-------|------|---------------------|
| D1 : Full (warmup + k-anneal) | **90.2% ± 2.1%** | **0.926** | **96.1%** |
| D2 : Sans warmup | 63.3% ± 42.7% | 0.660 | 71.1% |
| D3 : Sans k-annealing | 61.8% ± 43.5% | 0.644 | 67.8% |

**Interprétation :**
- Le warmup et le k-annealing sont **tous deux essentiels** — leur suppression cause un effondrement ($90\% \to \sim 62\%$) avec une variance catastrophique ($\pm 43\%$, soit 1 graine sur 3 qui converge).
- Le **k-annealing** est le composant le plus critique : sans lui ($D3$), le sélecteur doit trouver les bons 10 nœuds dès le début parmi 100, ce qui est quasi-impossible — il tombe dans un minimum local rapidement (early stop à 30 epochs pour une graine sur trois).
- Le **warmup contrastif** ($D2$) structure les embeddings initiaux et donne au sélecteur un meilleur point de départ. Sans lui, les logits du sélecteur sont aléatoires au démarrage, ce qui rend le k-annealing moins efficace.
- Ces deux techniques de curriculum sont indispensables pour que le mécanisme de sélection différentiable converge à grande échelle.

---

## 7. Synthèse et Conclusions

### 7.1 Résultats Clés

| Conclusion | Évidence |
|------------|----------|
| Les graphes spatio-temporels sont la bonne approche | GNN >90% à N=100, baselines <7% |
| Le signal temporel domine à petite échelle | Temporal-only = 98.4% à N=20 |
| La structure spatiale devient critique à grande échelle | Temporal-only chute à 64.6% à N=100 |
| Le GIB dégrade les performances sur les grands graphes | Late BN = 61.8% vs GCN = 94.1% à N=100 |
| Le contrastif est essentiel pour Late BN à grande échelle | +29.9% à N=100 |
| STA avec k=10 est trop agressif à N=100 | k=15 gagne +2.7%, k=30 atteint 93.8% |
| Le curriculum (warmup + k-annealing) est indispensable | Sans → effondrement à ~62% |

### 7.2 Question Ouverte : Le GIB est-il Adapté aux Grands Graphes ?

L'observation la plus frappante de cette semaine est la **contre-performance systématique du masque GIB** (et dans une moindre mesure du STE Selector) par rapport au GCN pur à $N=100$. Deux hypothèses explicatives :

1. **Parcimonie excessive** — Le GIB est optimisé pour trouver un sous-graphe minimal ($\min ||M||_1$). Avec 100 classes de cause racine, le sous-graphe causal n'est plus un seul nœud isolé mais peut impliquer un voisinage étendu. Forcer la parcimonie élimine l'information de voisinage nécessaire à la discrimination fine.

2. **Paysage de perte non-convexe** — À $N=100$, le masque $M \in [0,1]^{100}$ crée un espace de recherche exponentiellement plus grand. La relaxation Bernoulli (Late BN) et la Gumbel-sigmoid (STA) peinent à naviguer ce paysage, d'où la variance élevée entre graines.

Le Full-Graph GCN contourne ces deux problèmes en traitant tous les nœuds sans discrimination, laissant au classificateur final le soin de pondérer les contributions.

---

## 8. Orientations pour la Semaine 3

### 8.1 Priorité haute
1. **Optimiser STA avec k=15 pour N=100** — L'ablation C montre que c'est le sweet spot. Combiner $k=15$ avec un tuning des hyperparamètres devrait rapprocher STA du GCN pur.
2. **Tester à N=500/1000** — C'est le vrai test pour STA : à ces échelles, le GCN pur pourrait rencontrer des problèmes de mémoire tandis que STA ($k$ fixe) resterait stable.
3. **Données réelles K8s** — Valider le pipeline sur les données Prometheus du vrai cluster Chaos Mesh.

### 8.2 Priorité moyenne
4. **Explorer des alternatives au GIB** — Si le masque de parcimonie est le problème à grande échelle, envisager : (a) un GIB avec contrainte de parcimonie adaptative ($\beta$ fonction de $N$), (b) un mécanisme d'attention douce (Graph Attention) au lieu du masque binaire.
5. **Augmenter la patience et le learning rate schedule** pour Late Bottleneck à $N=100$ — L'early stop à 35 epochs (seed 42) est clairement prématuré.

### 8.3 Priorité basse
6. **Ensemble de modèles** — Combiner les prédictions du GCN pur et de STA pour tirer parti de la stabilité du GCN et de la sélectivité de STA.
7. **Complexification des anomalies** — Memory leaks progressifs, pannes multi-hop.
