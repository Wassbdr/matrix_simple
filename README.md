# GNN-RCA: Graph Neural Network Root Cause Analysis for Kubernetes

[![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper: coming soon](https://img.shields.io/badge/paper-coming%20soon-lightgrey.svg)](#citation)

## Abstract

Identifying the root cause of a failure in a microservice mesh within seconds is hard because
the dependency graph is large, alerts are noisy, and faults propagate non-linearly.
We propose a spatio-temporal GNN pipeline that combines a Late-Bottleneck STGCN with a
Graph Information Bottleneck (GIB) for full-graph analysis, and a complementary
Select-Then-Analyze (STA) architecture for sublinear inference on large topologies.
Evaluated on three RCAEval real-world datasets (TrainTicket, SockShop, Online Boutique)
and on multi-scale Barabási-Albert simulations (N = 20–100), our models reach
**Top-1 Avg@1 96.7%** on TrainTicket (3 seeds, 90 test cases), with STA inference
latency under **4 ms** at N = 20 and competitive accuracy at N = 100 where
O(N²) full-graph models degrade.
The full pipeline, benchmark scripts, chaos manifests, and SHA-256-verified result
artifacts are released to support reproducibility.

---

## Key Results

### RCAEval — Real-World Datasets (metrics-only, 3 seeds: 42, 123, 7)

Source: `rca_pipeline/benchmark_real_long_prepublish.json`

| Dataset          | STA Avg@1 | STA MRR | GATv2 Avg@1 | GATv2 MRR | Random baseline |
|------------------|----------:|--------:|------------:|----------:|----------------:|
| TrainTicket      |     96.7% |   98.3% |       95.0% |     97.5% |           ~3.3% |
| SockShop         |     80.0% |   87.7% |       76.7% |     84.7% |           ~5.0% |
| Online Boutique  |     68.3% |   79.5% |       71.7% |     82.0% |           ~5.0% |

> Avg@1 = RCAEval top-1 accuracy; MRR = Mean Reciprocal Rank.
> 3 seeds only — bootstrap CIs are wide; see `seed_level_bootstrap_reliable` flag in JSON.

### RCAEval — Multimodal Strict (metrics + traces, 3 seeds)

Source: `rca_pipeline/benchmark_real_long_mm_strict_prepublish.json`

| Dataset         | STA-MM Avg@1 | STA-MM MRR | GATv2-MM Avg@1 | GATv2-MM MRR |
|-----------------|-------------:|-----------:|---------------:|-------------:|
| TrainTicket     |        91.7% |      94.7% |          91.7% |        95.0% |
| Online Boutique |        61.7% |      77.9% |          80.0% |        88.4% |

> SockShop excluded from MM strict (insufficient trace coverage — see `docs/PAPER_BENCHMARKS.md`).

### Simulated Benchmark — Barabási-Albert Graphs (3 seeds)

Source: `rca_pipeline/benchmark_full_prepublish.json`

| Model                     |  N=20 Top-1 |  N=20 Top-2 |  N=50 Top-1 | N=100 Top-1 |
|---------------------------|------------:|------------:|------------:|------------:|
| STA Select-then-Analyze   |       76.9% |       81.7% |       62.8% |       57.6% |
| Late Bottleneck STGCN+GIB |       81.5% |       86.1% |       45.6% |       42.1% |
| Late Bottleneck GATv2     |       82.2% |       87.8% |       57.5% |       50.1% |
| Full-Graph GCN (baseline) |       83.8% |       88.6% |       52.6% |       48.0% |
| Random Forest (baseline)  |       68.7% |       85.9% |         n/a |         n/a |
| MLP (baseline)            |        8.3% |       18.2% |        2.6% |        1.3% |

> STA maintains competitive accuracy at N = 100 while STGCN+GIB degrades due to O(N²) scaling.
> All results are mean over 3 seeds; stddev available in the JSON files.

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

**Complexity:** Late Bottleneck O(N²), STA O(N + k²) with k = 10–15% of N.
**Parameters:** Late Bottleneck ~8 600, STA ~15 500 (N=20 scale).

---

## Quick Start

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0 with matching `torch_geometric` ≥ 2.4
- For K8s data collection: `kubectl`, `helm`, Chaos Mesh cluster-scoped

### Install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Smoke test — verify the install (< 2 min)

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

> Requires RCAEval data under `rcaeval/data/` — see [Data setup](#data-setup).

```bash
python -m rca_pipeline.benchmark_real \
    --dataset all \
    --seeds 42,123,7 \
    --epochs 100 \
    --sta-contrastive-warmup 0 \
    --output-json /tmp/bench_real.json
```

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

### ObservIT — Live Kubernetes collection

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
│   ├── drift_detector.py          # K8s concept-drift detection
│   ├── common.py                  # Seeds, splits, env info
│   └── tests/                     # pytest test suite (14 test files)
├── results_transmission/          # Canonical pre-publication artifacts
│   ├── canonical_results_manifest.json       # SHA-256 verified snapshot
│   ├── canonical_results_manifest.json.sha256
│   └── comparability_matrix.json             # Comparability verdicts vs. baselines
├── docs/
│   ├── PAPER_BENCHMARKS.md        # Detailed benchmark commentary
│   ├── ANALYSIS_SIM_VS_REAL.md
│   ├── ANALYSIS_MULTIMODAL.md
│   └── COMPARABILITY_MATRIX.md
├── k8s/
│   ├── app/                       # Application manifests
│   └── chaos/sockshop/            # Chaos Mesh fault injection manifests
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
@misc{badraoui2026gnnrca,
  title  = {GNN-RCA: Spatio-Temporal Graph Neural Networks for Root Cause
             Analysis in Kubernetes Microservices},
  author = {Badraoui, Wassim},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

Wassim Badraoui — wassim.badraoui@epita.fr
