# STA: A Select-Then-Analyze Framework for Scalable Root Cause Analysis in Microservice Architectures

[![CI](https://github.com/Wassbdr/matrix_simple/actions/workflows/ci.yml/badge.svg)](https://github.com/Wassbdr/matrix_simple/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Wassbdr/matrix_simple/branch/main/graph/badge.svg)](https://codecov.io/gh/Wassbdr/matrix_simple)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Nouveau sur le projet ?** Commencez par [`HANDOVER.md`](HANDOVER.md) — installation,
> ce qui est reproductible hors ligne, et où sont les résultats canoniques.
> Index de la documentation : [`docs/README.md`](docs/README.md).

## Abstract

Identifying the root cause of a failure in a microservice mesh within seconds is hard because
the dependency graph is large, alerts are noisy, and faults propagate non-linearly.
We propose a spatio-temporal GNN pipeline that combines a Late-Bottleneck STGCN with a
Graph Information Bottleneck (GIB) for full-graph analysis, and a complementary
Select-Then-Analyze (STA) architecture for sublinear inference on large topologies.
This is a **deliberately honest, reproducible study**, and it makes no SOTA claim. Across three
RCAEval benchmarks, Barabási-Albert simulations (N = 20-100), a live 14-service Kubernetes
cluster, and a 50-service synthetic-but-real deployment, STA keeps inference latency **flat**
as N grows and holds a **modest** accuracy edge at large N (**+2.4 pp** Avg@1 over GATv2 at
N = 100, 10 seeds, down from +6.3 pp at 5 seeds). At small graph scale (N ≤ 14) STA, GATv2
and an **isoperimetric BARO** baseline are within noise. BARO even **wins on Online Boutique**.
We also report a robust **negative result** on multimodal fusion and a fully **deployed**
operational stack (HPO → bundle → FastAPI service + live alerting on the cluster). All code,
benchmark scripts, chaos manifests, 10-seed and isoperimetric-baseline JSONs (SHA-256-verified)
and Docker images are released to support reproducibility.

---

## Key Results

> **Publication-grade results** below come from `scripts/run_publication_suite.sh`
> (5 seeds: 42, 123, 7, 1337, 2025; deterministic; 150 supervised epochs simulated, 100 epochs real).
> All numbers are reported as **mean ± 1σ across 5 seeds**; per-seed and bootstrap data live in
> the JSON artifacts under `results_transmission/`. SHA-256 of every artifact is in
> `results_transmission/paper_benchmark.sha256`.

### RCAEval: Real-World Datasets (metrics-only, 5 seeds)

Source: `results_transmission/paper_benchmark_real.json`

| Dataset          | STA Avg@1     | STA MRR       | GATv2 Avg@1   | GATv2 MRR     |
|------------------|--------------:|--------------:|--------------:|--------------:|
| TrainTicket      | 96.0% ± 3.7%  | 98.0% ± 1.9%  | 95.0% ± 4.5%  | 97.5% ± 2.2%  |
| SockShop         | 85.0% ± 8.9%  | 90.8% ± 5.8%  | 81.0% ± 8.0%  | 87.6% ± 5.8%  |
| Online Boutique  | 65.0% ± 9.5%  | 77.9% ± 5.3%  | 68.0% ± 5.1%  | 78.8% ± 4.2%  |

> **Comparability note (read this before citing a "beats baselines" claim).** The
> published RCAEval baseline numbers (BARO 71%, CausaRank 52%, MicroRCA 49%) use a
> *different* split/preprocessing. We re-ran BARO **isoperimetrically**, same cases,
> candidate node set, per-seed split and metric functions (`baseline_baro.py`,
> `results_transmission/baselines_rcaeval_ourprotocol.json`). Under identical
> conditions BARO is far stronger (Avg@1 **0.80 / 0.84 / 0.86** on TT / SS / OB): our
> models **win on TrainTicket, tie on SockShop, and lose on Online Boutique**. We make
> **no uniform SOTA claim** over BARO. See `docs/COMPARABILITY_MATRIX.md`.

### RCAEval: Multimodal Strict (metrics + traces, 5 seeds)

Source: `results_transmission/paper_benchmark_real_mm.json`

| Dataset         | STA-MM Avg@1   | STA-MM MRR     | GATv2-MM Avg@1 | GATv2-MM MRR   |
|-----------------|---------------:|---------------:|---------------:|---------------:|
| TrainTicket     | 91.0% ± 2.0%   | 94.3% ± 1.0%   | 88.0% ± 5.1%   | 92.4% ± 3.4%   |
| Online Boutique | 65.0% ± 15.5%  | 79.4% ± 8.3%   | 79.0% ± 8.6%   | 87.6% ± 4.5%   |

> SockShop is intentionally **skipped** in MM-strict mode: the public RCAEval release of SockShop
> ships with 0% trace coverage, and `--mm-strict` requires a real signal on every modality. The
> JSON records this as `mm_dataset_status: "skipped_with_reason"`.

### Simulated Benchmark: Barabási-Albert Graphs (5 seeds, N = 20 / 50 / 100)

Source: `results_transmission/paper_benchmark_sim.json`

| Model                     | N=20 Top-1     | N=50 Top-1     | N=100 Top-1    | N=100 latency (ms) |
|---------------------------|---------------:|---------------:|---------------:|-------------------:|
| STA Select-then-Analyze   | 80.5% ± 12.2%  | 58.9% ± 4.9%   | 55.2% ± 4.6%   | 3.4 ± 0.8          |
| Late Bottleneck STGCN+GIB | 82.3% ± 4.7%   | 46.9% ± 6.8%   | 47.1% ± 9.7%   | 2.5 ± 1.4          |
| Late Bottleneck GATv2     | 82.8% ± 3.9%   | 57.1% ± 2.3%   | 49.0% ± 2.9%   | 3.0 ± 1.3          |
| Full-Graph GCN (baseline) | 83.0% ± 2.9%   | 52.6% ± 2.6%   | 46.6% ± 4.2%   | 1.8 ± 0.4          |
| Random Forest (baseline)  | 68.5% ± 2.7%   | 31.2% ± 1.6%   |  6.5% ± 0.9%   | n/a                |
| MLP (baseline)            |  8.4% ± 2.4%   |  2.9% ± 0.5%   |  1.1% ± 0.4%   | n/a                |

> STA scales gracefully: at N = 100 it leads top-1 by **+5 to +9 pts** over O(N²) full-graph
> models while keeping inference under **3.5 ms**.

### ObservIT Live Cluster: Sock-Shop with Chaos Mesh (5 seeds, 5 chaos scenarios)

Source: `results_transmission/paper_benchmark_observit.json`, `paper_benchmark_observit_mm.json`

| Configuration                  | Avg@1         | MRR           |
|--------------------------------|--------------:|--------------:|
| STA (metrics-only)             | 79.6% ± 4.3%  | 86.8% ± 3.3%  |
| GATv2 (metrics-only)           | 76.2% ± 3.4%  | 86.0% ± 3.0%  |
| Late-STGCN+GIB (metrics-only)  | 73.6% ± 6.8%  | 84.4% ± 3.7%  |
| **STA-MM (metrics + Loki logs)** | **77.2% ± 2.1%** | **87.7% ± 1.0%** |
| GATv2-MM (metrics + logs)      | 76.2% ± 1.6%  | 84.5% ± 3.5%  |

> 11 chaos runs (CPU stress, memory stress, network delay, network loss, pod kill) on a real
> K8s cluster; logs queried live from Loki via port-forward. Multimodal logs **halve
> seed-to-seed variance** for STA (σ 4.3 → 2.1 on Avg@1).

### Reproducibility caveats: read before citing

- **5 seeds primary; 10-seed robustness check released.** The headline tables use 5 deterministic seeds, for which `ci95.<metric>.seed_level_bootstrap_reliable` is `false` (too tight to publish), so the tables report **±1σ across seeds**. We also ran the 10-seed set (`42,123,7,1337,2025,314,555,888,999,4242`, the `paper` preset in `validation_protocol.py`); there the flag is **`true`** for all metrics. Artifacts: `results_transmission/paper_benchmark_*_10seed.json` (hashes in `paper_benchmark_10seed.sha256`). At 10 seeds, STA's synthetic N=100 lead over GATv2 **shrinks from +6.2 to +2.4 pp** (still positive), STA still clearly beats Late-STGCN+GIB, and TrainTicket tightens to a ~1 pp STA/GATv2 tie, consistent with the *within-noise at small N* claim. We keep the 5-seed numbers as primary for continuity with the frozen canonical artifacts.
- **SockShop multimodal is skipped, not failed.** The public RCAEval SockShop release has 0% trace coverage; `--mm-strict` honors that and refuses to silently substitute zeros. This is logged transparently in `paper_benchmark_real_mm.json` under `sockshop.config.mm_dataset_status`.
- **Older reference values in legacy artifacts are stale.** Some `reference_top1` fields under `rca_pipeline/benchmark_full_prepublish.json` come from a short-epoch run and disagree with this 5-seed deterministic run. The numbers above are the canonical ones; please cite `paper_benchmark_*.json` only.
- **Determinism is per-seed, not bit-for-bit across machines.** The pipeline runs with `torch.use_deterministic_algorithms(True)` and CuDNN deterministic mode on, so within one machine each seed is reproducible; absolute numbers may shift by <0.5pp on different CUDA / PyTorch versions.

---

## Architecture

```
Input: time-series windows per microservice node
  [N nodes × T timesteps × F features]

        ┌─────────────────────────────────────┐
        │         Late-Bottleneck path         │
        │                                     │
        │  STGCN (spatial conv + GRU)         │
        │    → node embeddings [N × H]        │
        │  GIB mask (soft node selection)     │
        │    → bottlenecked subgraph          │
        │  Ranking head → root-cause scores   │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │            STA path                  │
        │                                     │
        │  STE selector (Straight-Through)    │
        │    → top-k nodes [k ≪ N]            │
        │  Local STGCN on subgraph            │
        │    → node embeddings [k × H]        │
        │  Contrastive warmup (sim only)      │
        │  Ranking head → root-cause scores   │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │       Multimodal extension           │
        │                                     │
        │  Metric features  (above)           │
        │  + Log features   (TF-IDF / BERT)   │
        │  + Trace features (span aggregates) │
        │  → late-fusion concat → ranking     │
        └─────────────────────────────────────┘

Graph construction:  Prometheus time-series → correlation + topology priors
Failure injection:   Chaos Mesh (CPU stress, network delay/loss, pod kill)
Load generation:     Locust
```

**Complexity:** Late Bottleneck O(N²), STA O(N + k²) with k = 10-15% of N.
**Parameters:** Late Bottleneck ~8 600, STA ~15 500 (N=20 scale).

---

## MLOps Pipeline

Beyond the research code, the model ships with a production-grade lifecycle
(full runbook in [`docs/OPERATIONS.md`](docs/OPERATIONS.md)):

```
HPO ─▶ train_and_persist ─▶ MLflow run + model registry (sta-sockshop@challenger)
                                      │
              promote.py (manual gate) ▼
                              sta-sockshop@champion
                                      │
   serve initContainer fetches @champion ▼      drift CronJob (Evidently/PSI)
              FastAPI /predict  ◀── bundle      shadow deploy (champion vs challenger)
                     │                                   │
                     └────────▶ Prometheus + Grafana SLO dashboard ◀──┘
```

- **Tracking + registry**: `rca_pipeline/tracking.py` (no-op offline) logs
  params/metrics and registers the bundle *as the artifact*; the run_id is
  cross-linked into the bundle and surfaced by `/info`.
- **Serving**: `serving/serve.py` (FastAPI, structured JSON logs, Prometheus
  metrics) pulls the model by registry alias via an initContainer.
- **CT**: retrain CronJob → `@challenger` → `promote.py` gate → `@champion`.
- **Monitoring**: Evidently drift job, shadow comparison, Grafana SLOs.
- **Delivery**: GHCR images (`release.yml`), Helm chart `deploy/helm/rca-stack`,
  self-hosted CD (`deploy.yml`).

Validated end-to-end on a live cluster: a real 5-seed champion
(Avg@1 0.786 / MRR 0.871) was trained, registered, promoted, and fetched by
alias. Post-defense model experiments are in
[`docs/ANALYSIS_MODEL_IMPROVEMENTS.md`](docs/ANALYSIS_MODEL_IMPROVEMENTS.md).

---

## Quick Start

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0 with matching `torch_geometric` ≥ 2.4
- For K8s data collection: `kubectl`, `helm`, Chaos Mesh cluster-scoped

### Install

```bash
git clone https://github.com/Wassbdr/matrix_simple.git
cd matrix_simple
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Smoke test: verify the install (< 2 min)

```bash
python -m pytest -q rca_pipeline/tests
python -m ruff check rca_pipeline
```

### Reproduce simulated benchmark

```bash
python -m rca_pipeline.benchmark \
    --output-json /tmp/bench_sim.json
```

### Reproduce real-world RCAEval benchmark

> Requires RCAEval data under `rcaeval/data/`. See [Data setup](#data-setup).

```bash
python -m rca_pipeline.benchmark_real \
    --dataset all \
    --seeds 42,123,7 \
    --epochs 100 \
    --sta-contrastive-warmup 0 \
    --output-json /tmp/bench_real.json
```

### Configuration files (YAML)

Every entry point also accepts `--config <yaml>` with pydantic validation
(`rca_pipeline/config.py`); explicit CLI flags override YAML values. Versioned
configs live in [`configs/`](configs/), e.g. the RCAEval run above is:

```bash
python -m rca_pipeline.benchmark_real \
    --config configs/benchmark_real.yaml \
    --output-json /tmp/bench_real.json
```

Unknown keys fail loudly, and the real-data warmup guardrail (§ below) is
enforced at config load time. See `docs/OPERATIONS.md` for the full runbook.

---

## Reproducing Results

Three preset levels are available via `validation_protocol.py`:

| Preset  | Seeds            | Epochs | Cases | Purpose                              |
|---------|------------------|-------:|------:|--------------------------------------|
| `smoke` | 42, 123          |     20 |    20 | CI sanity check (< 5 min)            |
| `long`  | 42,123,7,1337,2025 |   100 |   all | Reference run (stable CIs)           |
| `paper` | 10 seeds         |    100 |   all | Paper-grade statistical power        |

```bash
# Smoke (CI)
RCA_VAL_PRESET=smoke RCA_VAL_STA_WARMUP=0 \
    python -m rca_pipeline.validation_protocol

# Long reference
RCA_VAL_PRESET=long RCA_VAL_STA_WARMUP=0 \
    python -m rca_pipeline.validation_protocol

# Paper
RCA_VAL_PRESET=paper RCA_VAL_STA_WARMUP=0 \
    python -m rca_pipeline.validation_protocol
```

Outputs are written to `rca_pipeline/validation_runs/`.

### Ablation study

```bash
python -m rca_pipeline.benchmark_ablation \
    --output-json /tmp/ablation.json
```

Covers: GIB contribution, spatial vs temporal components, STA k-sensitivity,
contrastive warmup effect.

### Verifying canonical artifacts

SHA-256 hashes for the pre-publication snapshots are stored in
`results_transmission/canonical_results_manifest.json.sha256`.

```bash
sha256sum -c results_transmission/canonical_results_manifest.json.sha256
```

---

## Data Setup

### RCAEval (TrainTicket, SockShop, Online Boutique)

Follow the [RCAEval](https://github.com/NetManAIOps/RCAEval) instructions to
download raw CSVs, then place them under `rcaeval/data/`.
The loader at `rca_pipeline/loaders/rcaeval_loader.py` handles parsing and
graph construction automatically.

### ObservIT: Live Kubernetes collection

Requires a running cluster with Prometheus, Loki, Jaeger, and Chaos Mesh
in cluster-scoped mode. Bootstrap scripts:

```bash
chmod +x scripts/*.sh
./scripts/02-deploy-observability.sh
./scripts/03-deploy-app.sh
./scripts/04-install-chaos-mesh.sh
./scripts/05-enable-chaos-mesh-cluster-scope.sh   # required for chaos
./scripts/run_sockshop_dataset.sh                  # collect Sock Shop dataset
```

Chaos manifests are under `k8s/chaos/sockshop/` (CPU stress, memory stress,
network delay/loss, pod kill).

---

## Project Structure

```
.
├── rca_pipeline/              # Core package (source of truth)
│   ├── model_sta.py           # STA: Select-then-Analyze
│   ├── model_late_gat.py      # Late Bottleneck STGCN+GIB (GATv2 variant)
│   ├── model_sta_mm.py        # STA multimodal
│   ├── model_late_gat_mm.py   # Late Bottleneck multimodal
│   ├── benchmark.py           # Simulated benchmark (BA graphs)
│   ├── benchmark_real.py      # RCAEval / ObservIT benchmark
│   ├── benchmark_ablation.py  # Ablation study
│   ├── validation_protocol.py # Preset-based reproducibility harness
│   ├── dataset_builder.py     # Window extraction, graph construction
│   ├── graph_builder.py       # Topology-aware edge construction
│   ├── topologies.py          # Service topologies (custom_app, sock_shop)
│   ├── loaders/               # RCAEval and ObservIT data loaders
│   │   ├── rcaeval_loader.py
│   │   └── observit_loader.py
│   ├── losses.py              # InfoNCE, GIB, cascade-aware losses
│   ├── training_utils.py      # Training loop utilities
│   ├── log_feature_extractor.py   # Log → feature vectors
│   ├── trace_feature_extractor.py # Trace span → feature vectors
│   ├── prometheus_connector.py    # Live Prometheus scraper
│   ├── loki_connector.py          # Live Loki log fetcher
│   ├── metrics_rcaeval.py         # Avg@k, MRR, NDCG, bootstrap CI
│   ├── drift_detector.py          # K8s concept-drift detection (online ADWIN)
│   ├── ensemble.py                # Post-hoc STA+GATv2 vote + calibration
│   ├── cascade_prior.py           # Inference-time cascade propagation prior
│   ├── config.py                  # YAML + pydantic config layer
│   ├── tracking.py                # MLflow wrapper (no-op offline)
│   ├── promote.py                 # Challenger→champion promotion gate
│   ├── train_and_persist.py       # Train + register the serving bundle
│   ├── model_bundle.py            # Portable bundle (model/scaler/metadata)
│   ├── monitoring/drift_job.py    # Batch data-drift (Evidently/PSI)
│   ├── common.py                  # Seeds, splits, env info
│   └── tests/                     # pytest test suite
├── serving/                       # FastAPI service + watcher
│   ├── serve.py                   # /predict /health /ready /info /metrics
│   ├── rca_watcher.py             # Polling loop + alerts + shadow compare
│   ├── fetch_bundle.py            # Pull model by registry alias (initContainer)
│   ├── logging_config.py          # Structured JSON logging
│   └── Dockerfile*                # serve + watcher images
├── deploy/                        # Phase-2 MLOps manifests
│   ├── mlflow/                    # MLflow on K8s (SQLite + PVC)
│   ├── helm/rca-stack/            # Serve/watcher/CronJobs Helm chart
│   └── grafana/rca-model-slo.json # Model-SLO dashboard
├── docker/train.Dockerfile        # Training/retraining/initContainer image
├── configs/                       # Versioned YAML run configs
├── results_transmission/          # Canonical pre-publication artifacts
│   ├── canonical_results_manifest.json       # SHA-256 verified snapshot
│   ├── canonical_results_manifest.json.sha256
│   └── comparability_matrix.json             # Comparability verdicts vs. baselines
├── docs/
│   ├── OPERATIONS.md              # MLOps runbook (Phase 2)
│   ├── ANALYSIS_MODEL_IMPROVEMENTS.md  # Post-defense model experiments
│   ├── PAPER_BENCHMARKS.md        # Detailed benchmark commentary
│   ├── ANALYSIS_SIM_VS_REAL.md
│   ├── ANALYSIS_MULTIMODAL.md
│   └── COMPARABILITY_MATRIX.md
├── k8s/
│   ├── app/                       # Application manifests
│   └── chaos/sockshop/            # Chaos Mesh fault injection manifests
├── .github/workflows/            # CI, GHCR release, self-hosted deploy
├── scripts/                       # K8s cluster bootstrap and campaign scripts
├── rcaeval/                       # RCAEval data (not versioned, see .gitignore)
└── pyproject.toml                 # Single dependency source
```

---

## Comparability Notes

Results reported here use our own train/test splits and may differ from
published RCAEval baselines due to:
- Different split ratios and random seeds
- No cherry-picked threshold tuning on the test set
- `--mm-strict` mode requires *both* logs and traces to be present per window

Comparability verdicts (generated by `rca_pipeline/comparability_matrix.py`):

| Comparison                             | Verdict               |
|----------------------------------------|-----------------------|
| Ours (metrics-only) vs RCAEval baselines | `partially_comparable` |
| Ours (MM strict) vs RCAEval baselines    | `not_isoperimeter`     |

See `docs/COMPARABILITY_MATRIX.md` for the full analysis.

---

## Guardrail: Contrastive Warmup on Real Data

STA's contrastive warmup (`--sta-contrastive-warmup`) **must be set to 0**
for real-world benchmarks. With warmup = 10, TrainTicket Avg@1 drops from
73.3% to 0.0% (3 seeds, 20 cases).
Evidence: `results_transmission/warmup_real_trainticket_w{0,10}.json`.

All benchmark commands above already include `--sta-contrastive-warmup 0`.

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{badraoui2026sta,
  title  = {STA: A Select-Then-Analyze Framework for Scalable Root Cause
             Analysis in Microservice Architectures},
  author = {Badraoui, Wassim},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/Wassbdr/matrix_simple}
}
```

---

## License

MIT. See [LICENSE](LICENSE).

---

## Contact

Wassim Badraoui, wassim.badraoui@epita.fr
