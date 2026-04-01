# Analyse Multimodale (STA-MM / GATv2-MM)

## 1) Résultat factuel

Comparaison des runs réels `long` (5 seeds, 100 epochs):

| Dataset | STA (metrics-only) | STA-MM (full_mm_strict) | Delta |
|---|---:|---:|---:|
| trainticket | 0.960 | 0.910 | -0.050 |
| online-boutique | 0.650 | 0.650 | +0.000 |
| sockshop | 0.800 | n/a (MM skipped) | n/a |

Source: `rca_pipeline/validation_runs/*__metrics_only.json` et `*__full_mm_strict.json`.

Lecture:

- `trainticket`: dégradation nette en MM.
- `online-boutique`: pas de gain.
- `sockshop`: hors périmètre MM strict dans ce protocole (skipped).

## 2) Diagnostic de couverture MM

Dans les runs MM strict disponibles:

- `trainticket`: `mm_coverage.logs_ratio=1.0`, `traces_ratio=1.0`
- `online-boutique`: `mm_coverage.logs_ratio=1.0`, `traces_ratio=1.0`

Donc la baisse n’est pas due à une simple absence de modalités dans ces deux datasets.

## 3) Hypothèses explicatives

1. **Signal additionnel faible**: logs/traces n’ajoutent pas assez d’information discriminante au-delà des métriques.
2. **Coût d’optimisation**: la fusion multimodale augmente la difficulté d’entraînement (plus de paramètres/chemins), sans gain de signal proportionnel.
3. **Conflit de gradients inter-modalités**: la fusion attentionnelle peut sur-pondérer des modalités corrélées/bruitées.

## 4) Diagnostic complémentaire recommandé

Pour confirmer les hypothèses:

- extraire la variance moyenne de `x_logs` et `x_traces` par dataset;
- tracer la corrélation entre scores unimodaux et score fusionné;
- comparer MM strict vs modes partiels (`traces_only`, `trace_prior`) à seeds identiques.

## 5) Conclusion de contribution

Le résultat négatif MM est robuste sur ce protocole et doit être documenté comme tel:

- l’architecture MM est valide,
- mais les données RCAEval évaluées ici ne montrent pas de bénéfice systématique de la fusion multimodale.

Ce point renforce la crédibilité méthodologique et ouvre une piste claire:

- réévaluer MM sur des données cluster plus riches (ex. observ-it), où logs/traces ont potentiellement un meilleur pouvoir explicatif.
