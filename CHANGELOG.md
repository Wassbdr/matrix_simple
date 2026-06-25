# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — Phase 2 (MLOps industrialization)
- `rca_pipeline/tracking.py`: MLflow wrapper that is a complete no-op when
  `MLFLOW_TRACKING_URI` is unset (research/CI stay offline-capable). New
  `[tracking]` extra. `train_and_persist` logs params/metrics, logs the
  bundle dir *as the artifact* (no pyfunc), registers `sta-sockshop` with
  alias `challenger`, and cross-links the run_id into `metadata.json`.
- `rca_pipeline/promote.py`: challenger-vs-champion promotion gate on logged
  metrics; promotion is manual (`--apply` moves the `champion` alias).
- `rca_pipeline/monitoring/drift_job.py`: batch data-drift (Evidently
  `DataDriftPreset` with a built-in PSI fallback) vs a training-time
  `reference_windows.parquet`; pushes `rca_drift_share` to Pushgateway.
  New `[monitoring]` extra. Layered with the existing online ADWIN detector.
- `serving/fetch_bundle.py`: resolve `models:/<model>@<alias>` to a version
  and download the bundle by run_id (serve initContainer).
- `deploy/mlflow/mlflow.yaml`: MLflow on K8s (SQLite + PVC, `Recreate`,
  basic-auth, single gunicorn worker, default StorageClass).
- `deploy/helm/rca-stack/`: Helm chart — serve (+ fetch-bundle initContainer),
  watcher, PrometheusRule + MLOps alerts, retrain & drift CronJobs, optional
  shadow deployment.
- `docker/train.Dockerfile` + `.dockerignore`; `.github/workflows/release.yml`
  (GHCR build/push) and `deploy.yml` (self-hosted CD, never on `pull_request`).
- `deploy/grafana/rca-model-slo.json`: model-SLO dashboard.
- Shadow comparison in `serving/rca_watcher.py`
  (`rca_watcher_shadow_top1_agree_total` / `_score_delta`).

### Added — Phase 1A (model improvements, post-defense)
- `rca_pipeline/ensemble.py`: post-hoc STA+GATv2 soft/confidence vote,
  temperature calibration, ECE, routing-headroom diagnostics. Honest negative
  result on RCAEval (see `docs/ANALYSIS_MODEL_IMPROVEMENTS.md`).
- `rca_pipeline/cascade_prior.py`: inference-time propagation prior for cascade
  failures (`benchmark_real --cascade-prior-gamma`). +4.8 pp on the ObservIT
  cascade scenario at γ=0.5 (10 seeds), no regression on hard scenarios.
- `losses.class_aware_info_nce_loss` + the real-warmup-collapse fix: the §5.4
  collapse was GIB NaN-poisoning at `k==N` (float32 clamp no-op), not InfoNCE;
  fixed in `variational_gib_loss` + a `k>=N` skip in `benchmark_real`.

### Fixed — runtime integration bugs (found by end-to-end cluster validation)
- `serving/fetch_bundle.py`: `download_artifacts("models:/name@alias/sub")`
  404s on the 2.x server — resolve to a version then download by run_id.
- `deploy/mlflow/mlflow.yaml`: hard-coded StorageClass `standard` (Pending on
  Harvester) → use the cluster default; OOMKilled at 512Mi/4 workers →
  `--workers=1` + 1Gi; basic-auth secret aligned on the server's admin user.
- `rca_pipeline/monitoring/drift_job.py`: current sample was a single window
  (~14 rows) vs a thousands-row reference → slice the lookback into
  sub-windows and stack, matching the reference's unit.
- `deploy/helm/rca-stack/templates/prometheusrule.yaml`: PodMonitor scoped to
  `variant=champion` so shadow metrics don't pollute champion SLO alerts.

### Validated end-to-end on the cluster
- Full lifecycle on a real model: `train_and_persist` (5 seeds, sock-shop)
  → registered `sta-sockshop` v2 (Avg@1 0.786 / MRR 0.871) → promoted to
  `@champion` → fetched by alias with run_id cross-link intact.

## [0.2.0] — 2026-06-12

Phase 1B "engineering quality" milestone: tests on the production path,
config-as-code, reproducible images, canonical dependency pins.

### Added
- `serving/tests/` + `rca_pipeline/tests/test_model_bundle.py`: FastAPI
  contract tests (`/predict`, probes, `/info`), watcher webhook-history
  tests, bundle round-trip tests. Coverage gate in CI (`fail_under`,
  Codecov upload + badge).
- `rca_pipeline/config.py`: YAML + pydantic config layer for all CLI entry
  points (`--config`, precedence CLI > YAML > defaults, `extra="forbid"`).
  Versioned configs in `configs/`. The real-data contrastive-warmup
  guardrail (README §5.4) is now enforced at config load time.
- `serving/logging_config.py`: stdlib-only structured JSON logging
  (`LOG_FORMAT=json|text`), request-id middleware and per-request latency
  logging in `serve.py`, `rca_model_info` Prometheus gauge (bundle
  provenance for dashboards/shadow comparisons).
- `requirements/constraints-cpu.txt`: single source of truth for the CPU
  ABI pins (torch 2.0.1 / torch_geometric 2.4.0 / numpy<2), consumed by CI,
  both Dockerfiles and serving requirements.
- `docs/OPERATIONS.md`: operations runbook (config, lifecycle, serving,
  logs/metrics, rollback).
- CI: pip caching, actions pinned by commit SHA, docker build job for both
  serving images (no push), smoke-benchmark JSON uploaded as artifact.

### Changed
- `serving/Dockerfile` no longer bakes the model bundle into the image; the
  bundle is mounted at `MODEL_BUNDLE` (k8s volume now, registry download
  later). Docker builds work from a clean checkout.
- ntfy `Title` headers no longer contain emoji (HTTP headers are latin-1);
  watcher dashboard gains a webhook-driven alert history.
- `pydantic` and `pyyaml` move to core dependencies.

### Removed
- `requirements-frozen.txt` (froze a CUDA dev env — misleading for the CPU
  serving stack; superseded by `requirements/constraints-cpu.txt`).

## [0.1.0]

### Added
- MIT `LICENSE`, `CITATION.cff`, `SECURITY.md`.
- `CONTRIBUTING.md` and GitHub issue / PR templates.
- `Makefile` with `install`, `dev`, `test`, `lint`, `format`, `audit`,
  `benchmark-smoke`, `benchmark-paper`, `clean` targets.
- `.pre-commit-config.yaml` (ruff, large-file guard, private-key detection).
- `scripts/pre_publish_audit.sh` — repeatable repo audit before public push.
- `scripts/run_paper_benchmark.sh` — publication-grade benchmark suite
  (5 seeds, deterministic, simulated + RCAEval metrics-only + multimodal strict).
- `requirements-frozen.txt` — exact dependency pins for reproducibility.
- `.env.example` — environment-variable template (Loki, Prometheus, kubeconfig).

### Changed
- `pyproject.toml` declares MIT license, authors, keywords, PyPI classifiers.
- `pyproject.toml` moves `locust` to optional `[k8s]` extras (it is a load
  generator, not required for model code).
- README rewritten in English with sourced result tables, ASCII architecture
  diagram, badges, and BibTeX citation.
- Internal supervisor-facing documents moved to `docs/internal/`
  (`Week1.md` … `Week5.md`, `rapport_stage.md`, `resume_rapport.md`,
  `ACCOMPAGNEMENT_MAITRE_STAGE.txt`, `README_resultats.txt`).

### Removed
- Untracked generated artifacts: `rca_pipeline/graph_heatmap.png`,
  `rca_pipeline/simulated_dataset.csv`. Files remain on disk;
  removed from git index and added to `.gitignore`.

### Security
- Documented sandbox-only `mypassword` placeholder in `k8s/app/{db,backend}.yaml`
  (`SECURITY.md`).
- Confirmed via git history audit that no real secrets, kubeconfig, or
  credential files were ever committed.
