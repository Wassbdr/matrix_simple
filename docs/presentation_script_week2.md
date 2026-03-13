# Script de Présentation — Semaine 2 : Passage à l'Échelle et Étude d'Ablation

*Note pour le présentateur : Ce script fait suite à la présentation de la Semaine 1. Il couvre les travaux réalisés durant la deuxième semaine : audit du code, correction de bugs critiques, développement d'une deuxième architecture (STA), passage à l'échelle sur des graphes de 20 à 100 nœuds, et étude d'ablation.*

---

## Récapitulatif Semaine 1 (Diapo 1)
**Orateur :** "Avant de commencer, un très bref récapitulatif de ce que j'avais présenté la semaine dernière.

J'avais mis en place un laboratoire Kubernetes complet — 5 services, Chaos Mesh, Prometheus — et développé un premier modèle basé sur les STGCN (réseaux de graphes spatio-temporels) augmenté par le GIB (Graph Information Bottleneck). Les résultats étaient encourageants : environ 90% de Top-1 Accuracy en supervisé, et surtout 88.8% en semi-supervisé, sur une topologie de 5 nœuds.

Cette semaine, j'ai entrepris un travail d'audit, de correction, et d'extension significatif du pipeline. Je vais vous présenter ce qui a changé, les nouveaux résultats, et ce qu'on peut en conclure."

---

## Partie 1 : Audit du Code et Corrections Critiques (Diapo 2)
**Orateur :** "La première chose que j'ai faite cette semaine, c'est corriger les soucis qu'il y avait dans le code, ceux principaux étaient : 

**Mapping Pod-to-Node :** Le mécanisme qui associait les noms de pods Kubernetes aux noms de services utilisait une recherche par sous-chaîne naïve, ce qui créait des ambiguïtés (par exemple, le pod `auth-service-xxx` pouvait matcher à la fois 'auth' et 'auth-service'). J'ai réécrit ce mécanisme avec un matching par segments et une désambiguïsation par clé la plus longue.

**Variance statistique :** Les benchmarks précédents reposaient sur un seul entraînement — une seule graine aléatoire. C'est insuffisant pour tirer des conclusions fiables. J'ai ajouté un système multi-seed (3 graines : 42, 123, 7) avec calcul de la moyenne et de l'écart-type, ce qui permet de quantifier la stabilité de chaque modèle.

---

## Partie 2 : Passage à l'Échelle — Le Générateur Barabási-Albert (Diapo 3)
**Orateur :** "La topologie fixe de 5 nœuds du laboratoire K8s est idéale pour la mise au point, mais elle ne teste pas la capacité du modèle à gérer des architectures réalistes. En production, on parle de dizaines, voire de centaines de microservices.

J'ai donc développé un générateur de datasets simulés basé sur le modèle de **Barabási-Albert**. Pourquoi ce modèle ? Parce que les vraies architectures microservices présentent une distribution de degré en loi de puissance : quelques nœuds critiques (bases de données, message queues) ont beaucoup de connexions, tandis que la majorité des services en ont peu. Le modèle BA reproduit naturellement ce pattern.

Le générateur construit un graphe dirigé BA(N, m) et y injecte des anomalies avec une dynamique réaliste :
- Un signal transitoire au nœud racine (décroissance exponentielle)
- Une propagation multi-hop via une matrice de propagation P
- Des distracteurs (spikes aléatoires avec propagation limitée à 1 hop)

J'ai pu ainsi tester mes modèles sur trois échelles :
- **N=20** avec k=5 (3 000 fenêtres, 20 classes de cause racine)
- **N=50** avec k=8 (3 000 fenêtres, 50 classes)
- **N=100** avec k=10 (3 000 fenêtres, 100 classes)

Cela représente un saut de complexité considérable par rapport aux 5 nœuds initiaux."

---

## Partie 3 : Une Deuxième Architecture — Select-then-Analyze (Diapo 4 et 5)
**Orateur :** "En parallèle du modèle Late Bottleneck que j'avais présenté la semaine dernière, j'ai développé une deuxième architecture que j'appelle **STA** — Select-then-Analyze.

L'idée est d'**inverser la logique** du Late Bottleneck :
- Le Late Bottleneck traite **tout** le graphe (O(N²)), puis filtre avec le masque GIB.
- STA **sélectionne d'abord** un sous-ensemble de k nœuds pertinents (O(N)), puis n'analyse en profondeur que ce sous-graphe (O(k²)).

Concrètement, l'architecture STA comporte 5 étapes :
1. Un **LightEncoder** — un petit réseau GCN + convolution temporelle — encode tous les N nœuds de façon légère.
2. Un **STESelector** — basé sur le Straight-Through Estimator avec bruit de Gumbel — sélectionne les k nœuds les plus pertinents de façon différentiable.
3. Une **extraction de sous-graphe vectorisée** — un seul appel `pyg_subgraph` pour tout le batch.
4. Un **STGCN profond** — convolutions temporelles et spatiales, mais uniquement sur les k nœuds sélectionnés.
5. Un **scorer avec attention sur le contexte** — classifie chaque nœud du sous-graphe.

L'avantage théorique : la complexité est en **O(N + k²)** au lieu de **O(N²)**, et le nombre de paramètres reste **constant à environ 16 500** quelle que soit la taille du graphe. C'est un atout majeur pour la scalabilité.

L'entraînement utilise un **k-annealing** : on commence avec un k grand (N/2) pour que le modèle voie la cause racine avec une haute probabilité, puis on réduit progressivement k jusqu'à la valeur cible. Cela résout le problème de la poule et l'œuf — le sélecteur ne peut pas apprendre à bien sélectionner si le scorer ne voit jamais la bonne réponse."

---

## Partie 4 : Résultats du Benchmark Multi-Échelle (Diapos 6, 7 et 8)
**Orateur :** "Passons aux résultats. J'ai mené un benchmark complet sur les trois échelles, avec 3 graines par configuration et 5 modèles comparés.

### N=20 (petite échelle)
À 20 nœuds, les trois approches basées graphe sont excellentes :

| Modèle | Top-1 | MRR |
|--------|-------|-----|
| **STA (k=5)** | **97.9% ± 0.7%** | 0.983 |
| Full-Graph GCN | 97.3% ± 0.5% | 0.981 |
| Late Bottleneck | 96.6% ± 1.4% | 0.975 |
| Random Forest | 59.3% ± 1.6% | 0.721 |
| MLP | 6.7% ± 0.7% | 0.217 |

Pas de surprise : à petite échelle, tout fonctionne bien. Les baselines classiques s'en sortent déjà beaucoup moins bien.

### N=50 (échelle intermédiaire)
C'est ici que les différences commencent à apparaître :

| Modèle | Top-1 | MRR |
|--------|-------|-----|
| **Late Bottleneck** | **95.4% ± 0.8%** | 0.971 |
| Full-Graph GCN | 94.4% ± 3.2% | 0.961 |
| STA (k=8) | 93.7% ± 1.8% | 0.957 |
| Random Forest | 17.7% ± 1.7% | 0.276 |
| MLP | 1.9% ± 0.5% | 0.096 |

Late Bottleneck prend l'avantage avec une remarquable stabilité. Le Random Forest chute à 17% — il ne gère tout simplement pas la complexité combinatoire.

### N=100 (grande échelle)
Le test le plus exigeant :

| Modèle | Top-1 | MRR |
|--------|-------|-----|
| **Full-Graph GCN** | **94.1% ± 1.1%** | 0.959 |
| STA (k=10) | 90.2% ± 2.1% | 0.926 |
| Late Bottleneck | 90.2% ± 9.0% | 0.925 |
| Random Forest | 7.0% ± 0.9% | 0.149 |
| MLP | 1.7% ± 0.7% | 0.072 |

Plusieurs observations importantes ici :
1. **Toutes les approches graphe maintiennent >90% de précision à 100 nœuds.** C'est un résultat très solide.
2. **Le Full-Graph GCN domine de façon inattendue à N=100.** C'est le modèle le plus simple des trois, et pourtant il bat les deux architectures plus sophistiquées. C'est un résultat qui mérite analyse.
3. **Late Bottleneck montre une variance alarmante de ±9.0%.** Selon la graine, il oscille entre 81% et 99%. Ce n'est pas acceptable pour un déploiement en production.
4. **Les baselines classiques sont en chute libre** : le RF passe de 59% à 7%. Ces méthodes ne sont pas viables à grande échelle.

### Latence
Concernant les temps d'inférence, un point honnête que je dois mentionner :

| N | STA | GCN | Réduction de nœuds |
|---|-----|-----|--------------------|
| 20 | 4.2 ms | 1.5 ms | 75% |
| 50 | 3.5 ms | 1.6 ms | 84% |
| 100 | 3.3 ms | 1.4 ms | 90% |

À ces échelles (N≤100), STA est plus lent que le GCN malgré la réduction de nœuds. La surcharge de la phase de sélection compense le gain du sous-graphe réduit. L'avantage théorique O(N+k²) vs O(N²) ne se matérialisera que sur des topologies beaucoup plus larges (N>500) où la mémoire quadratique deviendra le vrai goulot d'étranglement. En revanche, le nombre de paramètres de STA reste constant à 16 549 quelle que soit N, ce qui est un avantage indéniable pour le déploiement."

---

## Partie 5 : Étude d'Ablation (Diapos 9, 10 et 11)
**Orateur :** "Pour comprendre *pourquoi* on observe ces résultats — et notamment pourquoi le GCN simple bat les architectures plus complexes à N=100 — j'ai mené une étude d'ablation systématique sur quatre axes. Les résultats sont maintenant complets.

### A. Décomposition par composant
J'ai isolé la contribution de chaque composant en testant :
- Un modèle **spatial-only** (GCN sans convolutions temporelles, juste mean+std sur le temps)
- Un modèle **temporal-only** (convolutions temporelles sans GCN, pas de structure de graphe)

À **N=20** :

| Variante | Top-1 |
|----------|-------|
| Temporal-only | **98.4% ± 0.0%** |
| Late BN (STGCN + GIB) | 97.9% ± 1.2% |
| Full-Graph GCN | 97.3% ± 0.5% |
| Spatial-only | 20.4% ± 3.4% |

À petite échelle, **le signal temporel domine** : un modèle sans graphe bat le GCN complet. Le pattern d'anomalie est suffisamment distinctif dans le temps pour que chaque nœud soit classifié indépendamment.

À **N=100**, le tableau s'inverse complètement :

| Variante | Top-1 |
|----------|-------|
| Full-Graph GCN | **94.1% ± 1.1%** |
| Temporal-only | 64.6% ± 23.3% |
| Late BN (STGCN + GIB) | 61.8% ± 43.0% |
| Spatial-only | 4.3% ± 1.8% |

Quand on passe de 20 à 100 classes, la discrimination temporelle seule ne suffit plus — le Temporal-only chute à 64.6%. C'est la **propagation spatiale** de l'anomalie à travers le graphe qui devient le signal discriminant critique. Le GCN pur, qui combine temporel et spatial, maintient 94.1%.

Et le résultat le plus frappant : le **GIB dégrade** les performances. Late Bottleneck s'effondre à 61.8% avec une variance catastrophique de ±43% — une graine sur trois early-stop à 4% d'accuracy après seulement 35 epochs. L'hypothèse : avec 100 classes, le masque de parcimonie du GIB élimine des nœuds porteurs d'information utile pour la discrimination fine.

### B. Protocole d'entraînement Late Bottleneck
La phase contrastive InfoNCE apporte-t-elle quelque chose ?

| | N=20 | N=100 |
|--|------|-------|
| Contrastif + Supervisé | 97.9% | 61.8% |
| Supervisé seul | 97.5% | 31.9% |
| **Δ** | **+0.4%** | **+29.9%** |

Le résultat est spectaculaire : à N=20, le gain est négligeable (+0.4%). Mais à N=100, la phase contrastive apporte **+30 points**. Elle structure l'espace latent avant la phase supervisée et empêche le modèle de rester bloqué dans un minimum local. C'est un résultat important : pour les grands graphes, le pré-entraînement contrastif n'est pas optionnel, il est **indispensable**.

### C. Sensibilité au k de STA (N=100)
J'ai testé k ∈ {10, 15, 20, 30} pour comprendre si k=10 est trop agressif :

| k (% des nœuds) | Top-1 | MRR | Variance |
|-------------------|-------|------|----------|
| k=10 (10%) | 90.2% | 0.926 | ± 2.1% |
| k=15 (15%) | 92.9% | 0.946 | ± 0.5% |
| k=20 (20%) | 92.8% | 0.947 | ± 1.4% |
| k=30 (30%) | **93.8%** | **0.953** | **± 0.3%** |

Oui, k=10 est effectivement trop agressif. Passer à k=15 gagne +2.7% et surtout **divise la variance par 4**. À k=30, STA atteint 93.8% ± 0.3%, très proche du GCN pur à 94.1%. L'écart résiduel de 0.3% pourrait être comblé avec un tuning plus fin. Et le nombre de paramètres reste **strictement constant à 16 549** quel que soit k.

### D. Protocole STA (N=100)
L'impact du warmup et du k-annealing :

| Configuration | Top-1 | Variance |
|---------------|-------|----------|
| Full (warmup + k-annealing) | **90.2%** | **± 2.1%** |
| Sans warmup | 63.3% | ± 42.7% |
| Sans k-annealing | 61.8% | ± 43.5% |

Les deux techniques de curriculum sont **absolument essentielles**. Sans elles, le sélecteur ne converge pas : une graine sur trois tombe dans un minimum local et early-stop à ~4% d'accuracy. Le k-annealing est le composant le plus critique — sans lui, le sélecteur doit trouver les 10 bons nœuds parmi 100 dès le début, ce qui est quasi-impossible."

---

## Partie 6 : Refactoring et Outillage (Diapo 11)
**Orateur :** "Au-delà du travail scientifique, j'ai investi du temps dans la qualité de l'outillage :

1. **Renommage des fichiers :** L'ancienne nomenclature (model_v2, model_v3, train_v3, benchmark_v3) prêtait à confusion. J'ai supprimé les fichiers V2 (devenus obsolètes) et renommé les fichiers V3 de façon explicite : `model_sta.py`, `train_sta.py`, `benchmark.py`. Le modèle original (V1) reste `model.py` — c'est l'approche Late Bottleneck, qui n'est pas obsolète mais complémentaire.

2. **Réécriture des outils :** Les scripts `infer.py`, `evaluate.py` et `visualize.py` étaient codés en dur pour la topologie K8s à 5 nœuds. Je les ai réécrits pour supporter les deux architectures (`--arch late` ou `--arch sta`) sur un nombre de nœuds arbitraire. La visualisation utilise maintenant un layout dynamique (`spring_layout` pour N>5, layout K8s fixe pour 5 nœuds).

3. **Correction du data leakage** dans `graph_builder.py` et du mapping pod-to-node.

4. **Nettoyage de la terminologie** : synthetic → simulated partout dans le code."

---

## Partie 7 : Analyse Critique et Conclusions (Diapo 13)
**Orateur :** "Permettez-moi de synthétiser ce que l'ensemble des résultats — benchmark et ablation — nous apprend.

**Ce qui est positif :**
- La thèse fondamentale est validée : **les approches graphe >90% à N=100** tandis que les baselines s'effondrent (<7%). La modélisation structurelle est la bonne direction.
- L'ablation démontre un résultat élégant : **le signal temporel domine à petite échelle** (Temporal-only = 98.4% à N=20), mais **la structure spatiale devient indispensable à grande échelle** (même modèle chute à 64.6% à N=100). C'est une preuve directe que la fusion spatio-temporelle n'est pas un luxe théorique — elle est nécessaire.
- STA avec k=30 atteint **93.8% ± 0.3%**, quasiment au niveau du GCN pur (94.1%), tout en ne traitant que 30% des nœuds avec un nombre de paramètres constant.
- Le pré-entraînement contrastif apporte **+30 points** à N=100 pour Late BN — ce n'est pas optionnel, c'est structurant.

**Ce qui est préoccupant :**
- Le **GIB est contre-productif à grande échelle** : Late BN à 61.8% ± 43% vs GCN pur à 94.1% ± 1.1%. Le masque de parcimonie élimine des nœuds porteurs d'information quand le nombre de classes augmente. C'est le résultat le plus important de l'ablation.
- Le **curriculum training (warmup + k-annealing) est fragile** : sans eux, STA s'effondre de 90% à ~62% avec une variance de ±43%. Une graine sur trois ne converge pas du tout.
- La **latence de STA** n'est pas encore avantageuse à N≤100 — l'avantage théorique O(N+k²) ne se concrétise pas à ces échelles.

**Conclusion principale :** À l'échelle testée (N≤100), le **Full-Graph GCN** est le modèle optimal — simple, stable, performant. Les mécanismes de sélection/compression (GIB, STE) ajoutent de la complexité sans gain à cette échelle. La question est : cela change-t-il à N=500+ ?"

---

## Partie 8 : Pistes pour la Semaine 3 (Diapo 14)
**Orateur :** "Sur la base de l'ablation complète, mes priorités sont affinées :

1. **STA avec k=15 comme configuration par défaut à N=100** : L'ablation C montre que c'est le sweet spot — +2.7% d'accuracy et variance divisée par 4 par rapport à k=10. Avec un tuning fin des hyperparamètres, on devrait pouvoir atteindre le GCN pur.

2. **Passage à N=500/1000** : C'est le vrai test de la thèse STA. À N=100, le GCN pur gagne, mais sa complexité quadratique pourrait devenir prohibitive à N=500+. Si STA (avec k fixe) rattrape ou dépasse le GCN à ces échelles, l'architecture est validée.

3. **Repenser le GIB pour les grands graphes** : L'ablation A montre clairement que la parcimonie forcée est inadaptée quand le nombre de classes est élevé. Deux pistes : (a) un β adaptatif fonction de N, (b) remplacer le masque binaire par un mécanisme d'attention douce (Graph Attention).

4. **Données réelles K8s** : Valider le pipeline sur les données Prometheus du vrai cluster Chaos Mesh — le benchmark final doit être sur des données réelles, pas simulées.

5. **Stabiliser la convergence** : Le pattern d'early stop catastrophique (1 graine sur 3 qui échoue) observé dans Late BN et STA sans curriculum est un problème fondamental. Explorer des stratégies de convergence plus robustes (warmup cosine, gradient clipping adaptatif).

Je vous remercie de votre attention. Avez-vous des questions ?"
