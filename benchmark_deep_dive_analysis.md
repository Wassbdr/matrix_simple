# Analyse Complète du Benchmark RCA (N=20, N=50, N=100)

Voici l'analyse détaillée des résultats finaux du stress test sur des graphes allant de 20 à 100 nœuds, avec 13 features multimodales et des séries temporelles générées selon les distributions du SOTA v2.

## 1. La Faillite du STGCN+GIB "Late Bottleneck" (V1)
Ce benchmark signe l'arrêt de mort de l'architecture historique qui consistait à traiter tout le graphe avant de filtrer.
*   **N=20** : Top-1 à **35.2%**. Déjà très en deçà.
*   **N=50** : Top-1 tombe à **22.7%**.
*   **N=100** : Top-1 à **31.1%**, mais avec une **variance énorme** ($\pm 11.7%$) et un effondrement complet sur certains runs (souci de convergence de l'Information Bottleneck sur de gros graphes très bruités).
*   **Conclusion** : Le modèle spatial (GCN) dilue le signal critique des root causes ("Gray Failures", "Memory Leaks") en l'agrégeant avec les signaux très forts mais inutiles des nœuds sains voisins. Plus le graphe est grand, plus la dilution est forte.

## 2. La Robustesse du Modèle STA (Select-then-Analyze)
Le nouveau paradigme (filtrer d'abord, analyser ensuite) prouve son efficacité architecturale face à l'échelle.
*   **N=20 (k=5)** : Top-1 à **67.8%** | MRR à **0.76**.
*   **N=50 (k=8)** : Top-1 à **48.6%** | MRR à **0.58**.
*   **N=100 (k=10)** : Top-1 à **41.9%** | MRR à **0.53**.
*   **Scalabilité** : La chute de précision est logique (chercher 1 aiguille parmi 100 est plus dur que 1 parmi 20), mais la performance reste très compétitive et stable. Surtout, la latence reste presque *constante* (~3.8ms) malgré la taille $\times 5$ du cluster, grâce à une réduction des nœuds traités jusqu'à **90%** !

Le module "Data-Centric" affiche des performances très proches (44% contre 42% sur N=100). Cela laisse penser que le curriculum learning ou la pondération par sévérité devront être tunés plus agressivement si l'on veut un écart majeur.

## 3. Le "Mythe" du MLP et Random Forest
C'est le point central de l'audit. Les baselines naïves affichent des performances impressionnantes, voire supérieures (MLP à **75%** sur N=20, et **62%** sur N=100). 
**Pourquoi ces résultats sont-ils anormaux (biaisés) ?**
*   **Transductive Memorization** : Dans ton simulateur, le graphe est toujours le même d'un échantillon à l'autre pour un cluster donné. L'index `0` est toujours le *Load Balancer*, l'index `15` toujours la *Database*, etc.
*   **Les features sont trop riches** : Les 13 features (issues des logs, traces et métriques) contiennent un signal local écrasant pour certains patterns (ex: pic d'erreur CPU sur une métrique = `memory_leak`).
*   **Implementation biaisée** : L'implémentation du MLP et RF "aplatit" tout le cluster (ex: `stats.flatten()`). Sur un graphe $N=100$, le MLP reçoit un vecteur de taille $100 \times 3 \times F$. Le MLP va simplement apprendre à mapper "Si gros chiffre à l'index 1493, alors c'est la classe N°42". **Il ne fait pas d'Analyse de Graphe, il fait de la mémorisation d'indices.**

### Preuve du biais par les paramètres :
*   MLP sur N=20 : 25 924 paramètres.
*   MLP sur N=50 : 63 874 paramètres.
*   MLP sur N=100 : **127 124 paramètres**.
Le MLP gonfle avec $N$, contrairement au GNN/STA qui reste fixe (**18 085 paramètres**) quelle que soit la taille du cluster.

## 4. Comment Fixer les Baselines (MLP / RF) ?
Pour prouver que le STA est la vraie solution SOTA, il faut que tes baselines jouent le jeu selon les mêmes règles. L'objectif est de mesurer si le signal "seul" d'un nœud suffit à le dénoncer, sans tricher sur sa position.

### Solution A : Modèles "Node-Wise" (Recommandé)
Le MLP doit recevoir en entrée seulement les features d'**un seul nœud** à la fois (et non les $N$ concaténées).
*   **Entraînement** : On entraîne le MLP (`input_dim` $= \text{features} \times \text{stats}$) en classification binaire : "Est-ce le root cause ? (0 ou 1)". Les $N$ nœuds d'un batch sont traités comme autant de samples indépendants.
*   **Inférence** : On fait passer les $N$ nœuds dans le MLP; les probabilités obtenues donnent le tri pour le Top-k.

### Solution B : Topological Shuffling
Garder l'architecture globale actuelle mais casser la mémorisation : à chaque passage dans une époque d'entraînement et d'inférence, on **mélange aléatoirement** l'ordre des nœuds de 0 à $N$. Le GNN survivra car l'`edge_index` sera mis à jour. Le MLP global s'effondrera à ~1% de précision, prouvant que sa performance actuele était fausse.

## Conclusion Stratégique
La méthode STA est une victoire mathématique et computationnelle (Top-1 solide en O($k^2$) de latence). Pour que le papier ou la présentation soit inattaquable, **tu dois absolument réécrire l'évaluation des baselines (MLP/RF)** vers une approche *Node-Wise classifier*, seule garante d'une confrontation équitable en contexte inductif.
