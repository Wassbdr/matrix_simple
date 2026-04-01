Paquet de résultats — traçabilité et reproduction
==================================================
Date de génération du paquet : 2026-04-01
Commit git (racine dépôt au moment de la préparation) : f22a6ce3cb12afb8e04d296fa5e9d80ddf6a1747
Python : 3.12.3 (adapter si votre venv diffère ; installer le projet avec pip install -e ".[dev]")

Environnement conseillé
-----------------------
cd /chemin/vers/matrix_simple
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
# Données RCAEval : placer ou télécharger sous rcaeval/data/ selon la doc du projet

Commandes types (reproduction indicative — ajuster chemins absolus)
-------------------------------------------------------------------

1) Benchmark simulé (BA) avec export JSON :
   python -m rca_pipeline.benchmark --output-json rca_pipeline/benchmark_full_prepublish.json
   (paramètres par défaut : voir README.md du dépôt, section « Benchmark multi-échelle »)

2) RCAEval — campagne longue métriques (exemple README) :
   python -m rca_pipeline.benchmark_real \
     --dataset all \
     --seeds 42,123,7 \
     --epochs 100 \
     --max-cases 90 \
     --sta-contrastive-warmup 0 \
     --output-json rca_pipeline/benchmark_real_long_prepublish.json

3) RCAEval — multimodal strict (aligné sur les artefacts *mm_strict* ; ajuster flags si besoin) :
   python -m rca_pipeline.benchmark_real \
     --dataset all \
     --seeds 42,123,7 \
     --epochs 100 \
     --max-cases 90 \
     --sta-contrastive-warmup 0 \
     --use-mm \
     --mm-strict \
     --output-json rca_pipeline/benchmark_real_long_mm_strict_prepublish.json

4) Matrice validation_protocol (stubs sockshop MM + runs tt/ob) :
   RCA_VAL_PRESET=long RCA_VAL_STA_WARMUP=0 RCA_VAL_DRIFT=0 \
     python -m rca_pipeline.validation_protocol

5) Ablation simulée — export JSON :
   python -m rca_pipeline.benchmark_ablation --output-json rca_pipeline/ablation_prepublish.json

6) ObservIT — exemple (dossier CSV ou fichier fusionné ; fenêtre selon campagne) :
   python -m rca_pipeline.benchmark_real \
     --observit-csv datasets/observit_runs \
     --observit-window 12 \
     --seeds 42,123,7 \
     --epochs 100 \
     --output-json datasets/observit_runs/benchmark_long_prepublish.json
   Validation de campagne : voir scripts/run_observit_sota_campaign.sh et docs associées.

7) Manifeste canonique de soutenance (source de vérité + SHA256) :
    python -m rca_pipeline.results_manifest \
       --repo-root /chemin/vers/matrix_simple \
       --output results_transmission/canonical_results_manifest.json

8) Matrice d'isopérimètre (comparabilité baselines) :
   python -m rca_pipeline.comparability_matrix \
      --repo-root /chemin/vers/matrix_simple \
      --output-md docs/COMPARABILITY_MATRIX.md \
      --output-json results_transmission/comparability_matrix.json

Documentation projet
--------------------
- docs/BENCHMARK_ARTIFACTS.md  — versionnage des artefacts, bootstrap
- docs/PAPER_BENCHMARKS.md     — périmètre MM, exclusion sockshop MM
- docs/COMPARABILITY_MATRIX.md — verdict isopérimètre claim par claim
- docs/AUDIT_CODE_RESULTS.md   — audit protocole / code

Contenu de l’archive ZIP associée (si présente)
-------------------------------------------------
 Arborescence type : benchmarks/, validation_runs/, observit/, + ce README + INVENTAIRE_JSON.txt + ACCOMPAGNEMENT_MAITRE_STAGE.txt + canonical_results_manifest.json (+ .sha256)

Archive prête à transmettre (sans dépendance zip)
--------------------------------------------------
- results_transmission/SOUTENANCE_PACKAGE_2026-04-01.tar.gz

Note : les chemins rca_pipeline/benchmark*.json et datasets/ sont souvent dans .gitignore ; joindre ce ZIP ou git add -f un sous-ensemble pour versionner les chiffres.
