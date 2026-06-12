# Changelog

All notable changes to this project will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
