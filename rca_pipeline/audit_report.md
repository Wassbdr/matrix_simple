# Audit Final Hybrid — RCA Pipeline

Date: 2026-03-25  
Scope: simulated pipeline + real benchmark pipeline + SOTA modules (2023-2026)

---

## Executive status

- Global state improved versus previous audit.
- Core gaps closed:
  - failure taxonomy now includes `hard`, `gray`, `cascade`
  - RCA metrics can be split by failure type
  - STA fusion no longer uses naive additive blend
  - causal graph output (`edge_weights`) is consumed by training losses
  - drift detector is integrated in both benchmark scripts
  - multimodal model variants are integrated in real benchmark script
- Remaining risks are mostly integration depth and protocol comparability, not missing files.

---

## Findings by severity

## CRITIQUE

1. Real multimodal path can run in degraded mode without true multimodal signal.
   - Evidence: `rca_pipeline/benchmark_real.py`, `rca_pipeline/loaders/rcaeval_loader.py`, `rca_pipeline/model_sta_mm.py`, `rca_pipeline/model_late_gat_mm.py`
   - Detail: `--use-mm` is wired, but `RCAEvalLoader` currently provides metric tensors only; logs/traces are often absent, so MM models fallback to zeros (`None` path).
   - Impact: benchmark output may be labeled "MM" while effectively metric-only.
   - Corrective action:
     - Add explicit runtime flag in results: `mm_effective=false/true` based on presence of `x_logs` and `x_traces`.
     - For strict protocol, fail fast when `--use-mm` and modalities are missing (or add `--allow-mm-fallback`).

2. Drift filtering may alter evaluation distribution without explicit protocol guardrails.
   - Evidence: `rca_pipeline/benchmark.py`, `rca_pipeline/benchmark_real.py`, `rca_pipeline/drift_detector.py`
   - Detail: samples in cooldown are skipped before training/evaluation, potentially changing class balance and case distribution.
   - Impact: scores can become non-comparable with non-drift baselines.
   - Corrective action:
     - Report pre/post sample counts per split and per root-cause class.
     - Add paired reporting: `raw_metrics` and `drift_filtered_metrics`.

## MAJEUR

3. STA does not yet consume causal cascade regularization (only Late BN / Late GAT).
   - Evidence: `rca_pipeline/benchmark.py`, `rca_pipeline/cascade_aware_loss.py`
   - Detail: `CausalCascadeAwareLoss` is integrated into `train_v1_compact` and `train_gat_compact`, not `train_sta_compact`.
   - Impact: method-to-method comparison includes asymmetric causal regularization.
   - Corrective action: add optional causal regularizer to STA scorer phase with its own weight flag.

4. RCAEval split protocol still uses deterministic case slicing without seeded shuffle in `_split`.
   - Evidence: `rca_pipeline/benchmark_real.py`
   - Impact: seeds mostly affect model optimization, less the data split.
   - Corrective action: per-seed shuffled split for honest variance estimates.

5. Gray-failure detector module exists but is not orchestrated in benchmark flow.
   - Evidence: `rca_pipeline/gray_failure_detector.py`, `rca_pipeline/benchmark.py`, `rca_pipeline/benchmark_real.py`
   - Impact: SOTA detector coverage is partial at pipeline level.
   - Corrective action: optional post-score adjustment or diagnostic channel using gray indicators.

## MINEUR

6. Simulated anomaly ratios (`gray_failure_ratio`, `hard_failure_ratio`) are not exposed via benchmark CLI.
   - Evidence: `rca_pipeline/simulated_data_scalable.py`, `rca_pipeline/benchmark.py`
   - Corrective action: add CLI args and persist values in output JSON.

7. Failure-type metrics are attached for STA, but not fully homogenized for all model families in benchmark aggregate output.
   - Evidence: `rca_pipeline/benchmark.py`
   - Corrective action: factor a shared helper for all `eval_*` methods and aggregate per-type consistently.

---

## SOTA checklist mapping (code-backed)

- Failure taxonomy hard/gray/cascade:
  - `rca_pipeline/simulated_data_scalable.py`
  - `rca_pipeline/failure_type_utils.py`
  - `rca_pipeline/evaluate.py`

- Selector/scorer robust STA fusion:
  - `rca_pipeline/sta_fusion_utils.py`
  - `rca_pipeline/benchmark.py`

- Causal graph and weighted cascade regularization:
  - `rca_pipeline/causal_graph_builder.py`
  - `rca_pipeline/cascade_aware_loss.py`
  - `rca_pipeline/benchmark.py`

- Drift-aware suppression logic:
  - `rca_pipeline/drift_detector.py`
  - `rca_pipeline/benchmark.py`
  - `rca_pipeline/benchmark_real.py`

- Multimodal modeling path:
  - `rca_pipeline/model_sta_mm.py`
  - `rca_pipeline/model_late_gat_mm.py`
  - `rca_pipeline/benchmark_real.py`
  - Connectors/extractors available in:
    - `rca_pipeline/prometheus_connector.py`
    - `rca_pipeline/log_feature_extractor.py`
    - `rca_pipeline/trace_feature_extractor.py`

- RCAEval metrics and baseline comparison:
  - `rca_pipeline/metrics_rcaeval.py`
  - `rca_pipeline/benchmark_real.py`
  - `rca_pipeline/loaders/rcaeval_loader.py`

---

## Priority corrective plan

P0 (immediate):
- Enforce truthful multimodal reporting (`mm_effective`) and protocol mode in `benchmark_real.py`.
- Publish raw vs drift-filtered metrics side-by-side.

P1 (next):
- Add seeded split in real benchmark for proper multi-seed variance.
- Extend causal cascade regularization to STA (optional flag, default off).

P2 (quality):
- Expose hard/gray ratios via CLI in `benchmark.py`.
- Unify per-type RCA metrics across all benchmark model outputs.

---

## Final assessment

- SOTA alignment is now materially better and operational in the main pipeline.
- The remaining concerns are not missing components; they are evaluation rigor and integration strictness.
- With P0 + P1 completed, the benchmark will be substantially more defensible for SOTA-style comparisons.

---

## Validation checklist (README ↔ code)

### Dependencies & environment

- **Install**: `pip install -r requirements.txt` succeeds.
- **Key imports**:
  - `python -c "import torch, torch_geometric, numpy, pandas, sklearn, networkx, matplotlib, requests, locust"` succeeds.

### Ports & service access (ObservIT cluster)

- **Frontend**: `kubectl port-forward svc/frontend -n rca-sandbox 8080:80`
- **Prometheus**: `kubectl port-forward svc/rca-prometheus-server -n rca-sandbox 9090:80`
- **Sanity**:
  - Frontend reachable at `http://127.0.0.1:8080`
  - Prometheus health responds: `http://127.0.0.1:9090/-/healthy`

### ObservIT dataset generation → CSV

- **Command** (from repo root):
  - `./scripts/run_observit_dataset.sh`
- **Expected outputs**:
  - `datasets/observit_runs/dataset_<run_id>.csv`
  - `datasets/observit_runs/dataset_<run_id>.meta.json`
- **CSV minimal schema**:
  - columns: `timestamp,pod,cpu_usage,ram_usage,network_tx,network_rx,error_rate,label,root_cause,failure_type,scenario`
  - optional: `latency_p99`

### ObservIT load (CSV or directory) → windows

- **Single CSV**:
  - `python -c "from rca_pipeline.loaders.observit_loader import load_observit; d,n,ei=load_observit('datasets/observit_runs/dataset_<run_id>.csv', window_size=12); print(len(d), len(n), ei.size())"`
- **Directory merge**:
  - `python -c "from rca_pipeline.loaders.observit_loader import load_observit; d,n,ei=load_observit('datasets/observit_runs', window_size=12); print(len(d))"`
- **Expected behavior**:
  - if directory: creates/uses `datasets/observit_runs/_merged_observit.csv`
  - drops baseline-only windows without `root_cause`

### Benchmarks

- **Simulated**:
  - `cd rca_pipeline && python3 benchmark.py --sizes 20 --seeds 42 --num-samples 400 --supervised-epochs 40 --late-contrastive-epochs 10 --baseline-epochs 40`
  - Expected: prints summary tables + exits 0.
- **RCAEval real** (requires `rcaeval/` data layout):
  - `python3 rca_pipeline/benchmark_real.py --dataset trainticket --seeds 42 --epochs 30 --output-json rca_pipeline/benchmark_real_smoke.json`
- **ObservIT real**:
  - `python3 rca_pipeline/benchmark_real.py --observit-csv datasets/observit_runs --observit-window 12 --seeds 42 --epochs 30 --output-json datasets/observit_runs/benchmark_smoke.json`
