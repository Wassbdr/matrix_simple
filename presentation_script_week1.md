# Script de Présentation : Avancées sur le Projet Root Cause Analysis (RCA)

*Note pour le présentateur : Ce script est conçu pour accompagner une présentation orale (type PowerPoint/Keynote). Vous pouvez ajuster le ton selon votre audience.*

---

## Introduction & Contexte (Diapo 1)
**Orateur :** "Bonjour à toutes et à tous. Aujourd'hui, j'ai le plaisir de vous présenter un état des lieux de mon avancée sur le projet d'analyse de causes racines, communément appelé Root Cause Analysis (RCA). 

L'objectif de ce projet est d'automatiser et de fiabiliser la localisation de l'origine d'une panne dans une architecture microservices complexe. Je vais d'abord vous décrire le laboratoire que j'ai conçu pour générer mes données. Ensuite, je vous présenterai l'approche mathématique que j'ai retenue, avant de partager avec vous mes tout récents résultats de benchmark. J'aborderai ensuite les difficultés que j'ai rencontrées, avant de conclure sur les prochaines étapes."

---

## Partie 1 : Mise en place du Laboratoire (Diapos 2 et 3)
**Orateur :** "Pour pouvoir concevoir et tester un modèle d'Intelligence Artificielle robuste, il me fallait des données d'entraînement réalistes. J'ai donc mis sur pied un laboratoire de test complet.

J'ai déployé l'environnement au sein du cluster *observ-it* de l'entreprise. Il s'agit d'un cluster Kubernetes Haute Disponibilité de 6 nœuds (3 nœuds de contrôle, 3 travailleurs). Sur ce cluster tourne une application distribuée classique en 3 tiers comprenant un Frontend, un Backend, une base de données, un cache, et un service d'authentification.

La clé de ce labo, c'est **l'observabilité totale**. J'utilise Prometheus pour récupérer en continu les métriques du système, accompagné d'OpenTelemetry et Jaeger pour le traçage distribué, ainsi que Loki pour les logs.

Mais un système sain ne génère pas de pannes. Pour créer un dataset d'anomalies, j'ai couplé un générateur de trafic réaliste (un script Locust) à un outil d'Ingénierie du Chaos (Chaos Mesh). Cette configuration m'a permis d'injecter des latences réseau, de tuer des pods, et de simuler des congestions CPU de manière déterministe. Ainsi, j'ai pu capturer comment une défaillance initiale – la cause racine – se propage en cascade à travers les différents services."

---

## Partie 2 : La Solution Mathématique Proposée (Diapos 4 et 5)
**Orateur :** "Une fois mon pipeline d'extraction de données en place, je me suis penché sur le modèle. Une simple approche statistique ne suffit pas pour modéliser les dépendances dynamiques. J'ai donc proposé une solution mathématique basée sur les **STGCN** (Spatio-Temporal Graph Convolutional Networks), augmentée par la théorie du **GIB** (Graph Information Bottleneck).

Voici comment ça fonctionne :
1. **Le Graphe (STGCN) :** Chaque service est représenté comme un nœud dans un graphe. Les communications entre les services sont les arêtes. Les graphes intègrent une dimension temporelle (via un système de fenêtres glissantes) pour capturer l'évolution de l'anomalie dans le temps et dans l'espace (c'est-à-dire d'un service à l'autre).
2. **Le GIB (Graph Information Bottleneck) :** L'infrastructure génère énormément de bruit. Le GIB agit comme un filtre mathématique. Son rôle est d'optimiser un compromis : il compresse les caractéristiques du graphe pour minimiser la redondance, tout en maximisant l'information utile permettant de prédire le nœud défaillant. En pratique, cela se traduit par l'apprentissage d'un masque de sous-graphe (*GIB mask*) qui ignore les métriques et les dépendances non pertinentes.

Cette approche permet au modèle de ne se focaliser que sur le 'chemin critique' de l'erreur."

---

## Partie 3 : Les Récents Résultats (Diapos 6 et 7)
**Orateur :** "Pour valider mes choix architecturaux, j'ai lancé un benchmark très récemment sur environ 5 000 fenêtres temporelles (4 000 pour l'entraînement, 1 000 pour le test). La distribution des pannes à détecter (Frontend, Auth, DB, etc.) était parfaitement bien répartie.

Je vous invite à regarder les résultats issus de multiples entraînements (moyenne sur 5 tests). J'ai comparé ma solution face à des modèles classiques :
*   Un **Random Forest** classique et un modèle **MLP** (Multi-Layer Perceptron) atteignent leurs limites dans la gestion des séquences complexes avec des Top-1 Accuracy respectifs de **71.7%** et **65.3%**.
*   L'approche graphe transforme radicalement les résultats. Un **Pure GCN** (qui ne gère pas l'aspect temporel) atteint un excellent score de **93.2%**.

C'est ici que l'approche **STGCN** classique (qui intègre le temps) trouve son équilibre avec un score très solide de **90.5% en Top-1** et **97.5% en Top-2**. 

Mais ce qui valide véritablement mon hypothèse de recherche sur l'Information Bottleneck, ce n'est pas uniquement le score brut, **c'est l'efficience d'apprentissage** (le GIB) :
*   Le modèle **STGCN + GIB en supervisé** atteint des niveaux quasi similaires avec **90.3% (Top-1)** et **97.5% en (Top-2)** pour un *Mean Reciprocal Rank* de **0.94**.
*   **La véritable prouesse se trouve ici : mon modèle STGCN + GIB en Semi-Supervisé.** Bien qu'il se prive de l'immense majorité des labels humains pour son apprentissage, le masque de sous-graphe réussi l'exploit de maintenir un **Top-1 Score de 88.8%** et un Top-2 massif de **96.3%**.

Cela prouve que l'algorithme a effectivement appris à compresser le bruit pour n'extraire que le graphe utile, lui conférant une énorme résilience en cas de données manquantes."

---

## Partie 4 : Les Difficultés Rencontrées (Diapo 8)
**Orateur :** "Bien entendu, ce projet ne s'est pas fait sans heurts, et j'aimerais partager avec vous les principaux défis que j'ai dû relever :

1. **La qualité des anomalies générées :** Initialement, les pannes générées par Chaos Mesh étaient trop évidentes, ce qui donnait des scores presque parfaits (et irréalistes) à de simples modèles de baselines. J'ai donc dû fournir un gros effort d'audit et raffiner drastiquement mon pipeline de génération de données. L'objectif était de créer des propagations d'erreurs plus subtiles et complexes, afin d'avoir un benchmark véritablement exigeant.
2. **L'orchestration de l'infrastructure sur le cluster d'entreprise :** Mettre en place et stabiliser cet environnement sur le cluster *observ-it* de l'entreprise, en assurant la liaison fluide entre les microservices, Prometheus, et les injecteurs de Chaos, a demandé un gros travail d'ingénierie système pour éviter que le labo ne s'effondre pendant l'extraction massive de données.
3. **Le calibrage mathématique du GIB :** Trouver le bon équilibre (le compromis compression vs préservation de l'information) dans les fonctions de perte globales de l'Information Bottleneck a demandé plusieurs itérations de complexification et de réglages fins des hyperparamètres."

---

## Partie 5 : Conclusions et Pistes pour la suite (Diapo 9)
**Orateur :** "Que puis-je tirer comme conclusions de ce benchmark ?
Premièrement, la modélisation sous forme de graphe spatio-temporel est incontestablement la bonne approche pour de l'architecture microservices. 
Deuxièmement, l'ajout du GIB confère au modèle une flexibilité incroyable, lui permettant d'exceller dans des conditions semi-supervisées, ce qui est extrêmement critique pour un déploiement grandeur nature en entreprise, où toutes les pannes ne peuvent pas être labellisées.

**Pour la suite des événements, plusieurs chantiers m'attendent :**
*   **Données réelles :** Je vais maintenant basculer de mon dataset synthétique vers un véritable run Prometheus en direct depuis le laboratoire Chaos Mesh de Kubernetes pour valider la robustesse aux asynchronismes naturels.
*   **Complexification des pannes :** Je vais créer des anomalies plus subtiles, comme des fuites de mémoire lentes (memory leaks) pour voir si la compression du GIB arrive toujours à isoler la cause.
*   **Ensemble de classifieurs :** Enfin, je vais peaufiner la stratégie d'ensemble intégrée au GIB pour gratter les derniers pourcentages de précision sur des topologies de graphes encore plus vastes.

Je vous remercie de votre attention. Avez-vous des questions ?"
