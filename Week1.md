# Rapport de Progression - Semaine 1 : Root Cause Analysis (RCA)

## 1. Constat et Formalisation Mathématique

Le projet s'appuie sur une formalisation mathématique rigoureuse, définissant le RCA non pas comme un simple problème de classification, mais comme la découverte d'entrées exogènes parcimonieuses dans un système dynamique.

### 1.1 Modélisation du Système Dynamique et de l'Anomalie Exogène
Nous considérons l'infrastructure cloud comme un graphe de dépendances $G = (V, E)$, où $V$ représente les microservices et $E$ les interactions réseau (traces). L'état interne du système à l'instant $t$ est noté $X(t) \in \mathbb{R}^{|V| \times d}$.

Le comportement du système est régi par une dynamique d'évolution spatio-temporelle perturbée par des événements exogènes :
$$X(t+1) = f_\theta(A, X(t)) + B U(t)$$

Où :
*   $f_\theta$ modélise la dynamique de propagation normale (apprise via un Spatio-Temporal Graph Convolutional Network - STGCN).
*   $A$ est la matrice d'adjacence issue de la topologie.
*   $B$ est la matrice d'injection des perturbations.
*   $U(t) \in \mathbb{R}^{|V| \times d_u}$ représente le signal exogène. En régime sain, $U(t)$ correspond au trafic utilisateur nominal. En cas de panne, $U(t)$ intègre une perturbation anormale et fortement localisée (l'injection de chaos).

### 1.2 Le Problème Inverse du Root Cause Analysis
Nous n'observons pas directement l'état latent $X(t)$, mais une projection bruitée constituée de séries temporelles (métriques Prometheus) :
$$Y(t) = g_\phi(X(t))$$

L'objectif du RCA est d'identifier le composant source de la panne. Mathématiquement, cela revient à trouver le support minimal de la perturbation exogène $U(t)$ qui explique la dérive observée dans $Y(t)$. Il s'agit d'un problème d'optimisation sous contrainte de parcimonie (*sparsity*) :
$$\min_{U} ||U||_0 \quad \text{sous contrainte} \quad Y(t) \approx \hat{Y}(t)$$

### 1.3 Résolution via Apprentissage Contrastif Spatio-Temporel
Puisque la véritable fonction de dynamique $f_\theta$ et les causes $U(t)$ sont inconnues (absence de labels en production), nous apprenons une représentation latente $H^{(t)}$ de la dynamique saine via une approche auto-supervisée.

Nous utilisons une perte InfoNCE asymétrique temporellement sur des fenêtres glissantes de taille $W$ :
$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(H_{anchor}^{(t)}, H_{positive}^{(t+\Delta t)})/\tau)}{\sum_{k} \exp(\text{sim}(H_{anchor}^{(t)}, H_{negative, k})/\tau)}$$

Cette fonction contraint l'encodeur à modéliser la propagation cohérente de l'information (ancres et positifs sains) et à rejeter dans des régions éloignées de l'espace latent les déviations causées par les perturbations exogènes $U(t)$ (négatifs anormaux).

### 1.4 Isolation Causale par Graph Information Bottleneck (GIB)
Pour résoudre différentiablement la contrainte de parcimonie $||U||_0$ et éviter la propagation du bruit (storm alerts), nous appliquons le principe du Goulot d'Étranglement de l'Information sur l'espace latent appris.

Nous cherchons un sous-graphe compressé $Z \subseteq V$ qui maximise l'information mutuelle avec l'anomalie tout en minimisant l'information conservée du graphe global :
$$\max I(Z; Y_{anomalie}) - \beta I(Z; X)$$

En pratique, un réseau de neurones paramètre une distribution de probabilité pour chaque nœud, dont on échantillonne un masque différentiable $M \in [0, 1]^{|V|}$ via une relaxation continue (ex: Gumbel-Softmax). La représentation finale compressée est :
$$Z^{(t)} = M \odot H^{(t)}$$

La fonction de coût globale du modèle devient alors :
$$\mathcal{L}_{totale} = \mathcal{L}_{InfoNCE} + \beta ||M||_1$$

Où la norme $L_1$ du masque agit comme un proxy différentiable pour forcer la parcimonie du sous-graphe. Lors de l'inférence, le nœud $i$ présentant la valeur $M_i$ la plus proche de $1$ est formellement identifié comme le point d'injection du signal exogène anormal $U(t)$, c'est-à-dire la Cause Racine.

---

## 2. État Actuel des Travaux et Validation Expérimentale

À l'issue de cette première phase de recherche, les fondations infrastructurelles et algorithmiques du pipeline de *Root Cause Analysis* ont été formellement établies et expérimentalement validées. Les résultats obtenus démontrent la viabilité de l'approche reposant sur le Graph Information Bottleneck (GIB) pour l'isolation d'anomalies en régime semi-supervisé.

### 2.1 Environnement d'Expérimentation et Génération du Jeu de Données
Afin de pallier l'absence de jeux de données publics suffisamment complexes et annotés, un banc d'essai (*testbed*) cloud-native a été déployé de novo. L'expérimentation repose sur le cluster Kubernetes d'entreprise `observ-it` (topologie haute disponibilité à 6 nœuds : 3 *control plane*, 3 *workers*).

L'architecture applicative sous-jacente est un système à trois tiers distribué (Frontend, Backend, Database, Auth, Cache). La couche de télémétrie garantit une observabilité exhaustive : collecte continue de métriques *time-series* via Prometheus, traçage distribué par Jaeger/OpenTelemetry, et agrégation de logs via Loki.

La constitution du jeu de données (Dataset) a nécessité l'orchestration d'expériences d'Ingénierie du Chaos (*Chaos Engineering*). Un générateur de charge de travail (Locust) a été couplé à la plateforme Chaos Mesh afin d'injecter des perturbations déterministes (latence réseau sélective, exhaustion CPU, défaillance de *pods*). Ce processus a fait l'objet d'un audit rigoureux pour s'assurer que la propagation des défaillances (effet cascade) produise un signal complexe, interdisant la résolution par de simples heuristiques de détection de seuil. La campagne de génération a abouti à un jeu de données synthétique massif et formalisé, constitué de $N=5000$ fenêtres temporelles de séries multivariées projetées sur des structures de graphes.

### 2.2 Évaluation Algorithmique et Analyse des Performances
L'architecture STGCN couplée à l'optimisation GIB a été évaluée contre des baselines statistiques (Random Forest, Multi-Layer Perceptron) ainsi qu'un modèle Graph Convolutional Network (GCN) atemporel. Afin de garantir la robustesse statistique, les métriques (*Top-1 Accuracy*, *Top-2 Accuracy*, *Mean Reciprocal Rank*) ont été moyennées sur une évaluation multi-seed (5 itérations).

L'analyse des résultats empiriques met en exergue plusieurs conclusions majeures :

1.  **Insuffisance des méthodes perceptroniques et statistiques classiques :** Les modèles ignorant la topologie du graphe de communication (MLP et Random Forest) plafonnent à des *Top-1 Accuracy* respectives de $65.3\% \ (\pm 1.37)$ et $71.7\% \ (\pm 0.97)$. Ces résultats confirment que les informations causales sont majoritairement encodées dans les dépendances structurelles inter-services, que ces architectures ne parviennent pas à modéliser.
2.  **Prépondérance de la structure spatiale (Graphe) :** Le simple ajout de la modélisation topologique via un réseau de convolution sur graphe classique (Pure GCN) engendre un saut drastique des performances asymptotiques, s'établissant à $93.2\% \ (\pm 0.37)$ en *Top-1 Accuracy*.
3.  **Apport du Graph Information Bottleneck (GIB) en régime semi-supervisé :**
    *   L'intégration conjointe de la dimension temporelle (STGCN) et du GIB en régime purement supervisé converge vers un score très solide de $90.3\% \ (\pm 0.00)$ en Top-1, se positionnant légèrement en deçà du GCN atemporel saturé en données.
    *   **Toutefois, la percée scientifique majeure** de cette approche réside dans l'inférence en régime **semi-supervisé**. Alors que la supervision a été volontairement retirée de la grande majorité des échantillons d'entraînement, le masque GIB différenciable parvient à comprimer la représentation latente pour isoler le sous-graphe causal, maintenant ainsi une *Top-1 Accuracy* de $88.8\% \ (\pm 0.00)$ et une *Top-2 Accuracy* de $96.3\% \ (\pm 0.00)$. 

Ces résultats démontrent que l'optimisation par Information Bottleneck agit effectivement comme un régularisateur puissant pour la découverte du *support causal* ($||U||_0$). Elle permet de pallier dramatiquement le problème endémique des environnements de production industriels : l'absence de l'annotation exhaustive des causes racines lors des incidents.

### 2.3 Orientations Futures des Travaux
Les prochaines itérations du projet viseront à renforcer la validité externe du modèle :
1.  **Passage à l'Échelle et Inférence en Temps Réel :** Transition du jeu de données statique vers un flux d'ingestion *live* sondant l'API de Prometheus afin de valider la robustesse du modèle face aux asynchronismes naturels du cluster de production.
2.  **Complexification du Spectre d'Anomalies :** Incorporation d'injections de chaos plus diffuses et progressives (*memory leaks*, dégradation lente de lien réseau) pour éprouver les limites temporelles de la compression GIB.
3.  **Ensembling Latent :** Optimisation de la stabilité de l'inférence du masque GIB sur des macro-topologies via l'intégration de méthodes d'ensemble au niveau de la fonction de décision.
