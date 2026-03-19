# Analyse de Biais et Audit des Résultats du Benchmark RCA

Cette analyse objective décompose les raisons pour lesquelles les modèles **MLP** et **Random Forest** affichent des performances "anormalement" élevées par rapport aux architectures GNN (STA, STGCN).

## 1. Le Biais de "Positionnement Global" (Data Leakage structurel)

L'audit du code `benchmark.py` révèle une différence fondamentale dans l'entrée des modèles :

*   **STA / GCN / STGCN** : Traitent le graphe de manière relationnelle. Ils cherchent à apprendre des motifs de propagation.
*   **MLP / Random Forest** : Sont implémentés comme des modèles **Globaux** (Global Classifiers).
    *   `eval_mlp_baseline` (ligne 812) : Aplatit les statistiques de **tous les nœuds** ($N \times 3 \times W$) en un seul vecteur géant.
    *   `eval_rf_baseline` (ligne 837) : Fait de même avec toutes les features brutes ($N \times W \times F$).

### Pourquoi c'est un biais ?
Dans ton environnement de test, le **graphe est fixe** (seed constante pour la topologie). 
Le MLP n'apprend pas la "Root Cause Analysis", il apprend une **classification multi-classe sur des identités fixes**. 
Il "mémorise" que :
> *"Le slot n°5 du vecteur d'entrée correspond au nœud 'Proxy'. S'il y a du délai sur ce slot, c'est lui le coupable."*

C'est ce qu'on appelle du **Transductive Overfitting**. Si tu changeais l'ordre des nœuds ou si tu testais sur un nouveau graphe, le MLP/RF tomberait à 0% de précision, alors que STA (en théorie) survivrait.

## 2. Richesse des Features vs Importance du Graphe

Avec **13 features** (Métriques, Traces, Logs), le signal local est devenu extrêmement dense. 
*   **Hypothèse** : La signature d'une "Gray Failure" ou d'une "Memory Leak" est tellement explicite dans les 13 dimensions d'un nœud que le passage par le graphe (GNN) devient une **source de bruit** (Over-smoothing).
*   C'est pour cela que le **Full-Graph GCN** est parfois moins bon que le MLP : en agrégeant les voisins, il "noie" le signal très clair du coupable dans le silence des nœuds sains.

## 3. Critique Objective de ton modèle (STA)

Tu n'as pas besoin de confort : voici les points faibles de l'approche actuelle mis en lumière par le benchmark.

| Point Faible | Description |
| :--- | :--- |
| **Goulot d'étranglement du Selector** | Si le "Light Selector" rate le nœud coupable dans son top-$k$, le "Scorer" n'a aucune chance. À $N=50$ ou $100$, la probabilité que le selector se trompe augmente, ce qui explique la chute du Top-1. |
| **Instabilité du GIB** | L'Information Bottleneck (GIB) est très sensible aux hyperparamètres (température, $\beta$). S'il n'est pas parfaitement setté, il peut rejeter trop d'informations utiles. |
| **Complexité vs Performance** | Le MLP est 10x plus simple et performe mieux ici. Cela suggère que pour rendre STA compétitif, il faut soit réduire le bruit des features, soit forcer des anomalies plus "diffuses" où le graphe est la *seule* source d'information. |

## 4. Comparaison SOTA (Gray Failures)

Le fait que STA atteigne **~54% de Top-1 sur N=100** reste une prouesse technique, car :
1.  Il le fait avec **10x moins de paramètres** que le MLP global.
2.  Il est **Inductif** : il pourrait (en théorie) être déployé sur un autre cluster sans réentraînement (contrairement au MLP/RF).
3.  Il offre une **Expliquabilité** (via le masque de sélection) que le Random Forest ne peut donner sur de la donnée temporelle brute.

## Conclusion de l'Audit

Les résultats MLP/RF sont **artificiellement gonflés** par la fixité de ta topologie de test. Ils ne font pas de la RCA, ils font de la "mémorisation d'emplacement". 

**Recommandation** : Pour un benchmark "juste", il faudrait tester le MLP en mode **Node-Wise** (uniquement les features d'UN nœud en entrée, sans savoir lequel c'est) pour voir si le signal local suffit, ou faire un **Shuffle** des indices des nœuds à chaque batch pour casser le biais positionnel.
