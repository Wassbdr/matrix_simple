# Plan Stratégique d'Amélioration RCA - Alignement SOTA 2026

## 1. Diagnostic Croisé : Benchmark vs. SOTA

L'analyse de vos récents résultats de benchmark face à l'état de l'art (SOTA 2023-2026) met en évidence plusieurs décalages critiques et opportunités d'amélioration :

### A. L'Effondrement à l'Échelle (N=100)
- **Constat Benchmark :** STGCN+GIB passe de 67% (N=20) à 13% (N=100). Le STA chute de 52% (N=50) à 19% (N=100).
- **Analyse sélecteur STA :** À N=100, le sélecteur plafonne à `RC-in-top-15 = 22.98%`. Si la cause racine n'est pas dans le top 15, l'analyseur n'a aucune chance de la trouver.
- **Diagnostic SOTA :** Dans les grands graphes, le phénomène de **Cascading Failures** (SOTA 1.2) crée une "tempête" d'anomalies en aval (ex: *service timeout*). Nos modèles actuels se noient dans ces effets de bord au lieu de remonter à la source.

### B. L'Échec Cuirassant du Modèle "Data-Centric"
- **Constat Benchmark :** Le STA Data-Centric s'effondre littéralement (Top-1 à 5% sur N=50 et N=100).
- **Diagnostic SOTA :** Le *curriculum learning* axé sur la sévérité focalise le modèle sur les anomalies massives (Hard Failures). Or, selon le SOTA (1.1), les **Gray Failures** (dégradations silencieuses) et **Metastable Failures** sont prédominants. En ignorant les signaux faibles, le modèle Data-Centric rate l'essentiel des propagations subtiles.

### C. La Limite de la Topologie BA (Barabási-Albert)
- **Diagnostic SOTA :** Les pannes cloud se propagent souvent via des anti-patterns architecturaux (SOTA 1.5 : *God Services, Shared Databases*). Les graphes aléatoires BA ne modélisent pas fidèlement ces goulots d'étranglement ni les structures de microservices réelles.

---

## 2. Plan d'Action Structuré

Voici un plan détaillé en 4 phases pour ramener le pipeline RCA au niveau de l'état de l'art.

### Phase 1 : Inférence Causale pour Vaincre les Cascades
**Objectif :** Résoudre l'effondrement Top-1 sur N=100 en différenciant la cause de la conséquence.
- **Action 1.1 :** Intégrer un module de **Causal Knowledge Graphs (CKG)** (cf. SOTA 3.3) en pré-traitement. Au lieu de fournir au GCN un simple DAG structurel formel, utiliser un algorithme d'inférence (comme l'algorithme de découverte PC) sur les données temporelles pour orienter les arêtes dynamiquement (Effet $\rightarrow$ Cause).
- **Action 1.2 :** Remanier le récepteur de propagation du STGCN pour empêcher les nœuds en aval fortement anormaux (ex: proxy saturé) de masquer le nœud discret en amont qui cause la panne (ex: base de données lente).

### Phase 2 : Refonte du Data-Centric pour les Gray & Metastable Failures
**Objectif :** Corriger la régression du Data-Centric et le rendre pertinent face aux pannes modernes.
- **Action 2.1 :** Inverser ou nuancer le curriculum learning : au lieu d'apprendre seulement sur les anomalies sévères, introduire une pénalité sur les métriques trop évidentes et forcer l'apprentissage sur les **Gray Failures** (anomalies à faible variance mais haute corrélation causale).
- **Action 2.2 :** Ajouter des "vérificateurs sémantiques" synthétiques imitant le concept du concept **T2C** (SOTA 3.4), en générant des anomalies de type binaire ou silencieux dans nos datasets simulés, sans pic de métrique apparent.

### Phase 3 : Optimisation du Sélecteur STA
**Objectif :** Atteindre +80% de `RC-in-top-k` pour garantir le succès de l'analyseur sur N=100+.
- **Action 3.1 :** Le sélecteur doit intégrer la topologie via l'algorithme Random Walk with Restart (ex: PageRank personnalisé pointant à rebours du flux de requêtes) plutôt qu'un MLP classique qui lisse l'information sur l'ensemble des nœuds.

### Phase 4 : Migration vers les Datasets de Référence
**Objectif :** Valider les modèles sur des données reconnues par la communauté académique.
- **Action 4.1 :** Remplacer / Augmenter le générateur BA(N,m) actuel par les topologies de **TrainTicket** (41 ms) ou **SockShop** (SOTA 2.2).
- **Action 4.2 :** Évaluer le STGCN+GIB sur **RCAEval (WWW'25)** ou **AnoMod** pour démontrer sa supériorité sur des traces réelles multimodales.
