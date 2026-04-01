# RCA Pipeline — Root Cause Analysis microservices

Plateforme de recherche pour le Root Cause Analysis (RCA) dans des architectures microservices.

Le projet combine:
- un pipeline de donnees simulees multi-echelle (graphes Barabasi-Albert),
- des benchmarks reels RCAEval,
- un flux de collecte ObservIT sur cluster Kubernetes instrumente.

Architectures comparees:
- Late Bottleneck STGCN+GIB (graphe complet puis filtrage),
- STA Select-then-Analyze (selection de k noeuds puis analyse locale),
- variantes GATv2 et multimodales (metrics + logs + traces).

---

## 1. Ce que couvre ce depot

- Simulation RCA a differentes tailles de graphes.
- Evaluation reelle RCAEval (TrainTicket, SockShop, Online Boutique).
- Campagnes reelles ObservIT (Chaos Mesh + Prometheus + Locust).
- Protocole de validation reproductible et artefacts de soutenance.

---

## 2. Installation

Prerequis:
- Python 3.9+
- Environnement virtuel recommande
- Optionnel: kubectl + Helm si vous utilisez le flux K8s ObservIT

Depuis la racine du depot:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Source de verite des dependances: pyproject.toml.

---

## 3. Quickstart experimental

### 3.1 Benchmark simule (canonique)

```bash
python -m rca_pipeline.benchmark \
  --output-json rca_pipeline/benchmark_full_prepublish.json
```

Defaults importants dans benchmark.py:
- sizes=20,50,100
- seeds=42,123,7,1337,2025

### 3.2 Benchmark reel RCAEval (metrics-only)

```bash
python -m rca_pipeline.benchmark_real \
  --dataset all \
  --seeds 42,123,7 \
  --epochs 100 \
  --max-cases 90 \
  --sta-contrastive-warmup 0 \
  --output-json rca_pipeline/benchmark_real_long_prepublish.json
```

### 3.3 Benchmark reel RCAEval (multimodal strict)

```bash
python -m rca_pipeline.benchmark_real \
  --dataset all \
  --seeds 42,123,7 \
  --epochs 100 \
  --max-cases 90 \
  --sta-contrastive-warmup 0 \
  --use-mm \
  --mm-strict \
  --output-json rca_pipeline/benchmark_real_long_mm_strict_prepublish.json
```

### 3.4 Validation protocolaire (smoke, long, paper)

```bash
RCA_VAL_PRESET=long RCA_VAL_STA_WARMUP=0 RCA_VAL_DRIFT=0 \
python -m rca_pipeline.validation_protocol
```

Le script ecrit les sorties dans rca_pipeline/validation_runs.

### 3.5 Ablation

```bash
python -m rca_pipeline.benchmark_ablation \
  --output-json rca_pipeline/ablation_prepublish.json
```

---

## 4. Flux ObservIT (cluster K8s)

### 4.1 Collecte de runs

```bash
./scripts/run_observit_dataset.sh
```

Campagne multi-scenarios:

```bash
./scripts/run_observit_sota_campaign.sh
```

### 4.2 Benchmark ObservIT

```bash
python -m rca_pipeline.benchmark_real \
  --observit-csv datasets/observit_runs \
  --observit-window 12 \
  --seeds 42,123,7 \
  --epochs 100 \
  --output-json datasets/observit_runs/benchmark_long_prepublish.json
```

Equivalent shell helper:

```bash
./scripts/run_observit_benchmark.sh \
  --observit-csv datasets/observit_runs \
  --observit-window 12 \
  --seeds 42,123,7 \
  --epochs 100
```

---

## 5. Resultats consolides (snapshot canonique)

Sources:
- results_transmission/canonical_results_manifest.json
- rca_pipeline/benchmark_real_long_prepublish.json
- rca_pipeline/benchmark_real_long_mm_strict_prepublish.json

### 5.1 Simule (Top-1)

| N | STA | Late STGCN+GIB | Late GATv2 | Full-Graph GCN |
|---|---:|---:|---:|---:|
| 20 | 0.7691 | 0.8145 | 0.8223 | 0.8379 |
| 50 | 0.6278 | 0.4561 | 0.5752 | 0.5263 |
| 100 | 0.5757 | 0.4208 | 0.5012 | 0.4799 |

### 5.2 Reel RCAEval metrics-only (Avg@1, MRR)

| Dataset | STA Avg@1 | STA MRR | GATv2 Avg@1 | GATv2 MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9667 | 0.9833 | 0.9500 | 0.9750 |
| sockshop | 0.8000 | 0.8775 | 0.7667 | 0.8472 |
| online-boutique | 0.6833 | 0.7953 | 0.7167 | 0.8197 |

### 5.3 Reel MM strict (Avg@1, MRR)

| Dataset | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|---|---:|---:|---:|---:|
| trainticket | 0.9167 | 0.9472 | 0.9167 | 0.9500 |
| online-boutique | 0.6167 | 0.7792 | 0.8000 | 0.8839 |

Note: sockshop est exclu du perimetre MM strict (status skipped_with_reason).

### 5.4 Guardrail important sur le reel

Le warmup contrastif STA sur reel est risque sans controle strict.

Comparatif A/B fourni:
- results_transmission/warmup_real_trainticket_w0.json
- results_transmission/warmup_real_trainticket_w10.json

Snapshot trainticket (20 cas, 3 seeds):
- warmup=0: STA Avg@1 = 0.7333
- warmup=10: STA Avg@1 = 0.0000

Recommendation: conserver --sta-contrastive-warmup 0 en benchmark reel par defaut.

---

## 6. Reproductibilite et comparabilite

Artefacts canoniques (soutenance):
- results_transmission/canonical_results_manifest.json
- results_transmission/canonical_results_manifest.json.sha256
- results_transmission/comparability_matrix.json
- docs/COMPARABILITY_MATRIX.md
- results_transmission/AUDIT_FINAL_SOUTENANCE_2026-04-01.md

Regeneration des artefacts:

```bash
python -m rca_pipeline.results_manifest \
  --repo-root . \
  --output results_transmission/canonical_results_manifest.json

python -m rca_pipeline.comparability_matrix \
  --repo-root . \
  --output-md docs/COMPARABILITY_MATRIX.md \
  --output-json results_transmission/comparability_matrix.json
```

Verdicts de comparabilite a retenir:
- Ours vs baselines publiees metrics-only: partially_comparable
- Ours MM strict vs baselines publiees: not_isoperimeter

---

## 7. Deploiement K8s (optionnel)

Scripts:
- scripts/00-check-context.sh
- scripts/02-deploy-observability.sh
- scripts/03-deploy-app.sh
- scripts/04-install-chaos-mesh.sh

Execution type:

```bash
chmod +x scripts/*.sh
./scripts/02-deploy-observability.sh
./scripts/03-deploy-app.sh
./scripts/04-install-chaos-mesh.sh
```

Acces local:

```bash
kubectl port-forward svc/frontend -n rca-sandbox 8080:80
kubectl port-forward svc/rca-grafana -n rca-sandbox 3000:80
kubectl port-forward svc/rca-prometheus-server -n rca-sandbox 9090:80
kubectl port-forward svc/rca-jaeger-query -n rca-sandbox 16686:16686
```

---

## 8. Qualite logicielle

```bash
python -m pytest -q rca_pipeline/tests
python -m ruff check rca_pipeline
```

---

## 9. Structure utile

```text
matrix_simple/
  rca_pipeline/         # modeles, training, benchmark, evaluation
  scripts/              # flux K8s et campagnes observit
  k8s/                  # manifests namespace/app/chaos
  docs/                 # analyses, audit, benchmarks
  results_transmission/ # manifest canonique + comparability + package soutenance
  rcaeval/              # donnees RCAEval (si disponibles localement)
```

---

## 10. Documentation utile

- docs/PAPER_BENCHMARKS.md
- docs/BENCHMARK_ARTIFACTS.md
- docs/ANALYSIS_SIM_VS_REAL.md
- docs/ANALYSIS_MULTIMODAL.md
- docs/AUDIT_CODE_RESULTS.md
- docs/COMPARABILITY_MATRIX.md
- docs/Week1.md
- docs/Week2.md
- docs/Week3.md
- docs/Week4.md
- docs/rapport_stage.md

---

## 11. Nettoyage cluster (optionnel)

```bash
kubectl delete networkchaos,stresschaos,podchaos --all -n rca-sandbox
helm uninstall chaos-mesh rca-grafana rca-loki rca-prometheus rca-jaeger -n rca-sandbox
kubectl delete -f k8s/app/ -n rca-sandbox
kubectl delete namespace rca-sandbox
```