# Note de synthèse à transmettre - RCA microservices

Date: 2026-04-02  
Auteur: Wassim Badraoui

Je vous partage ici une synthèse claire et directement exploitable de mon travail de stage sur l'analyse de cause racine (RCA) en environnement microservices. Mon objectif, sur cette phase, n'était pas uniquement de produire un bon score, mais de construire une chaîne complète et défendable, du dataset jusqu'à l'audit des résultats.

## 1. Ce que j'ai réalisé concrètement

Pendant le stage, j'ai structuré et industrialisé un pipeline RCA complet:

- formalisation du problème RCA comme problème inverse sous contrainte de parcimonie,
- pipeline unifié collecte -> entraînement -> évaluation -> export,
- comparaison rigoureuse de plusieurs familles de modèles (Late BN STGCN+GIB, Late BN GATv2, STA, variantes multimodales),
- protocoles distincts et reproductibles sur trois volets: simulé, réel RCAEval, et campagnes ObservIT,
- gouvernance des claims via manifeste canonique, matrice de comparabilité, et garde-fous logiciels.

Ce point est important pour moi: le livrable final n'est pas un script isolé, mais un système expérimental traçable, rejouable et auditable.

## 2. Mon analyse des résultats

### 2.1 Résultats simulés (Top-1)

| N | STA | Late BN (STGCN+GIB) | Late BN (GATv2) | Full-Graph GCN |
|---|---:|---:|---:|---:|
| 20 | 0.7691 | 0.8145 | 0.8223 | 0.8379 |
| 50 | 0.6278 | 0.4561 | 0.5752 | 0.5263 |
| 100 | 0.5757 | 0.4208 | 0.5012 | 0.4799 |

Mon interprétation est la suivante:

- à petite échelle (N=20), les modèles full-graph restent très compétitifs,
- quand la taille du graphe augmente (N=50, N=100), STA devient plus robuste.

Autrement dit, je ne constate pas de gagnant universel; je constate un effet d'échelle, ce qui est plus intéressant scientifiquement et plus réaliste pour une mise en production.

### 2.2 Résultats réels RCAEval (metrics-only)

| Dataset | STA Avg@1 | STA MRR | GATv2 Avg@1 | GATv2 MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9667 | 0.9833 | 0.9500 | 0.9750 |
| sockshop | 0.8000 | 0.8775 | 0.7667 | 0.8472 |
| online-boutique | 0.6833 | 0.7953 | 0.7167 | 0.8197 |

Ce que je retiens:

- STA est devant sur 2 datasets sur 3,
- GATv2 reste meilleur sur online-boutique.

Je considère donc qu'un choix unique de modèle pour tous les contextes serait une erreur opérationnelle. Les résultats plaident plutôt pour une stratégie par dataset/corpus.

### 2.3 Résultats réels multimodaux stricts

| Dataset | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9167 | 0.9472 | 0.9167 | 0.9500 |
| online-boutique | 0.6167 | 0.7792 | 0.8000 | 0.8839 |

Le multimodal est prometteur, mais j'insiste sur une limite méthodologique importante: sockshop est en couverture traces incomplète (skipped_with_reason). En conséquence, je n'agrège pas ces résultats en score global MM isopérimètre.

### 2.4 Test de robustesse critique (warmup réel)

- warmup=0 -> STA Avg@1 = 0.7333
- warmup=10 -> STA Avg@1 = 0.0000

Ce test est très structurant pour la suite: le warmup contrastif réel est instable dans cette configuration. C'est la raison pour laquelle j'ai maintenu le guardrail warmup réel à 0 par défaut.

## 3. Limites reconnues (et assumées)

Je préfère être explicite sur les limites plutôt que de sur-vendre les résultats:

1. La puissance statistique reste limitée sur certains benchmarks (souvent n=3 seeds).
2. Les performances sont sensibles à certains choix protocolaire/hyperparamètres (k, warmup, split mode).
3. La comparabilité externe n'est pas totale:
   - metrics-only: partially_comparable,
   - multimodal strict: not_isoperimeter.
4. Certaines briques d'analyse avancée côté réel (ex. failure-type) restent à approfondir.

## 4. Décisions que je recommande

Sur la base des résultats consolidés, je recommande:

1. d'utiliser STA par défaut sur RCAEval metrics-only,
2. de garder GATv2 comme challenger actif (surtout pour online-boutique),
3. d'interdire l'agrégation globale multimodale tant que la couverture n'est pas homogène,
4. de maintenir warmup réel = 0 tant qu'une campagne plus large ne démontre pas le contraire,
5. de lancer une campagne multi-seeds plus large pour consolider les intervalles de confiance.

## 5. Conclusion personnelle

À ce stade, j'estime avoir livré une plateforme RCA microservices techniquement solide, reproductible et exploitable, avec des résultats utiles pour la décision. Ma conclusion principale est qu'il faut raisonner en robustesse contextuelle (par dataset et par périmètre), et non en classement absolu unique des modèles.
